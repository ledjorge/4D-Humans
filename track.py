import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import os
import hydra
import torch
import torch.nn as nn
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from hmr2.utils import recursive_to

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
import phalp.trackers.PHALP as phalp_tracker_module
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
from phalp.configs.base import CACHE_DIR
import phalp.models.hmar.hmar as phalp_hmar_module
import phalp.models.predictor.pose_transformer_v2 as pose_transformer_v2_module

from hmr2.datasets.utils import expand_bbox_to_aspect_ratio

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

def _resolve_runtime_device(preferred: Optional[str] = None) -> torch.device:
    """Select the best available device and patch `.cuda()` calls to map to it when needed."""
    candidate_order = []
    if preferred is not None:
        candidate_order.append(preferred)
    candidate_order.extend(["cuda", "mps", "cpu"])

    target_device = None
    for candidate in candidate_order:
        if candidate == "cuda" and torch.cuda.is_available():
            target_device = torch.device("cuda")
            break
        if candidate == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            target_device = torch.device("mps")
            break
        if candidate == "cpu":
            target_device = torch.device("cpu")
            break

    if target_device is None:
        raise RuntimeError("Unable to select a runtime device. Please check your PyTorch installation.")

    if target_device.type != "cuda":
        # Remap .cuda() calls so the upstream PHALP code runs on the selected backend.
        def _tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
            return torch.Tensor.to(
                self,
                target_device if device is None else torch.device(target_device),
                non_blocking=non_blocking,
                memory_format=memory_format,
            )

        def _module_cuda(self, device=None):
            return nn.Module.to(self, target_device if device is None else torch.device(target_device))

        setattr(torch.Tensor, "cuda", _tensor_cuda)
        setattr(nn.Module, "cuda", _module_cuda)

    return target_device

RUNTIME_DEVICE = _resolve_runtime_device()

if RUNTIME_DEVICE.type != "cuda":
    def _load_weights_cpu(self, path):
        checkpoint_file = torch.load(path, map_location=RUNTIME_DEVICE)
        state_dict_filt = {}
        for k, v in checkpoint_file['model'].items():
            if ("encoding_head" in k or "texture_head" in k or "backbone" in k or "smplx_head" in k):
                state_dict_filt.setdefault(k[5:].replace("smplx", "smpl"), v)
        self.load_state_dict(state_dict_filt, strict=False)

    phalp_hmar_module.HMAR.load_weights = _load_weights_cpu

    def _pose_transformer_load_weights_cpu(self, path):
        checkpoint_file = torch.load(path, map_location=RUNTIME_DEVICE, weights_only=False)
        checkpoint_file_filtered = {
            k.replace("encoder.", ""): v for k, v in checkpoint_file['state_dict'].items()
        }
        self.encoder.load_state_dict(checkpoint_file_filtered, strict=False)

    pose_transformer_v2_module.Pose_transformer_v2.load_weights = _pose_transformer_load_weights_cpu

LOCAL_DETECTRON_CKPT = Path(__file__).resolve().parent / "weights" / "detectron2" / "model_final_f05665.pkl"

if LOCAL_DETECTRON_CKPT.exists():
    _orig_setup_detectron2 = PHALP.setup_detectron2

    def _setup_detectron2_with_local(self):
        log.info("Loading Detection model...")
        if self.cfg.phalp.detector == 'maskrcnn':
            model_zoo = phalp_tracker_module.model_zoo
            DefaultPredictor_Lazy = phalp_tracker_module.DefaultPredictor_Lazy

            self.detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            self.detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            self.detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(self.detectron2_cfg)
            self.class_names = self.detector.metadata.get('thing_classes')
        elif self.cfg.phalp.detector == 'vitdet':
            from detectron2.config import LazyConfig
            import phalp

            cfg_path = Path(phalp.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            self.detectron2_cfg = LazyConfig.load(str(cfg_path))
            self.detectron2_cfg.train.init_checkpoint = str(LOCAL_DETECTRON_CKPT)
            for i in range(3):
                self.detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
            self.detector = phalp_tracker_module.DefaultPredictor_Lazy(self.detectron2_cfg)
        else:
            _orig_setup_detectron2(self)
            return

        if getattr(self.cfg, "use_gt", False):
            self.setup_detectron2_with_RPN()

    PHALP.setup_detectron2 = _setup_detectron2_with_local

_orig_setup_visualizer = PHALP.setup_visualizer

def _setup_visualizer_conditional(self):
    if not getattr(self.cfg.render, "enable", True):
        log.info("Skipping visualizer setup because rendering is disabled.")
        self.visualizer = None
        return
    _orig_setup_visualizer(self)

PHALP.setup_visualizer = _setup_visualizer_conditional

class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model.to(RUNTIME_DEVICE)
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        batch = recursive_to(batch, RUNTIME_DEVICE)
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out
    
class HMR2023TextureSampler(HMR2Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Model's all set up. Now, load tex_bmap and tex_fmap
        # Texture map atlas
        bmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/bmap_256.npy'))
        fmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/fmap_256.npy'))
        self.register_buffer('tex_bmap', torch.tensor(bmap, dtype=torch.float))
        self.register_buffer('tex_fmap', torch.tensor(fmap, dtype=torch.long))

        self.img_size = 256         #self.cfg.MODEL.IMAGE_SIZE
        self.focal_length = 5000.   #self.cfg.EXTRA.FOCAL_LENGTH

        try:
            import neural_renderer as nr
        except ModuleNotFoundError:
            nr = None
            log.warning("neural_renderer not available; falling back to simplified UV rendering (no occlusion handling).")

        self._renderer_device = torch.device("cpu")
        if nr is not None:
            self.neural_renderer = nr.Renderer(
                dist_coeffs=None,
                orig_size=self.img_size,
                image_size=self.img_size,
                light_intensity_ambient=1,
                light_intensity_directional=0,
                anti_aliasing=False,
            ).to(self._renderer_device)
        else:
            self.neural_renderer = None

    def forward(self, x):
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        batch = recursive_to(batch, RUNTIME_DEVICE)
        model_out = self.model(batch)

        # from hmr2.models.prohmr_texture import unproject_uvmap_to_mesh

        def unproject_uvmap_to_mesh(bmap, fmap, verts, faces):
            # bmap:  256,256,3
            # fmap:  256,256
            # verts: B,V,3
            # faces: F,3
            valid_mask = (fmap >= 0)

            fmap_flat = fmap[valid_mask]      # N
            bmap_flat = bmap[valid_mask,:]    # N,3

            face_vids = faces[fmap_flat, :]  # N,3
            face_verts = verts[:, face_vids, :] # B,N,3,3

            bs = face_verts.shape
            map_verts = torch.einsum('bnij,ni->bnj', face_verts, bmap_flat) # B,N,3

            return map_verts, valid_mask

        pred_verts = model_out['pred_vertices'] + model_out['pred_cam_t'].unsqueeze(1)
        device = pred_verts.device
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long, device=device)
        map_verts, valid_mask = unproject_uvmap_to_mesh(self.tex_bmap, self.tex_fmap, pred_verts, face_tensor) # B,N,3

        # Project map_verts to image using K,R,t
        # map_verts_view = einsum('bij,bnj->bni', R, map_verts) + t # R=I t=0
        focal = self.focal_length / (self.img_size / 2)
        map_verts_proj = focal * map_verts[:, :, :2] / map_verts[:, :, 2:3] # B,N,2
        map_verts_depth = map_verts[:, :, 2] # B,N

        img_rgba = torch.cat([batch['img'], batch['mask'][:,None,:,:]], dim=1) # B,4,H,W
        img_rgba_at_proj = torch.nn.functional.grid_sample(img_rgba, map_verts_proj[:,None,:,:]) # B,4,1,N
        img_rgba_at_proj = img_rgba_at_proj.squeeze(2) # B,4,N
        if self.neural_renderer is not None:
            # Render Depth. Annoying but we need to create this
            K = torch.eye(3, device=device)
            K[0, 0] = K[1, 1] = self.focal_length
            K[1, 2] = K[0, 2] = self.img_size / 2  # Because the neural renderer only support squared images
            K = K.unsqueeze(0)
            R = torch.eye(3, device=device).unsqueeze(0)
            t = torch.zeros(3, device=device).unsqueeze(0)
            rend_depth = self.neural_renderer(
                pred_verts.to(self._renderer_device),
                face_tensor[None].expand(pred_verts.shape[0], -1, -1).to(self._renderer_device).int(),
                mode='depth',
                K=K.to(self._renderer_device),
                R=R.to(self._renderer_device),
                t=t.to(self._renderer_device),
            ).to(device)

            rend_depth_at_proj = torch.nn.functional.grid_sample(
                rend_depth[:, None, :, :], map_verts_proj[:, None, :, :]
            )  # B,1,1,N
            rend_depth_at_proj = rend_depth_at_proj.squeeze(1).squeeze(1)  # B,N
            visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4)  # B,N
        else:
            visibility_mask = torch.ones_like(map_verts_depth, dtype=torch.bool)

        img_rgba_at_proj[:,3,:][~visibility_mask] = 0

        # Paste image back onto square uv_image
        uv_image = torch.zeros((batch['img'].shape[0], 4, 256, 256), dtype=torch.float, device=device)
        uv_image[:, :, valid_mask] = img_rgba_at_proj

        out = {
            'uv_image':  uv_image,
            'uv_vector' : self.hmar_old.process_uv_image(uv_image),
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam':  model_out['pred_cam'],
        }
        return out

class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2023TextureSampler(self.cfg)
        self.HMAR.to(RUNTIME_DEVICE)

    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        (
            pred_bbox, pred_bbox, pred_masks, pred_scores, pred_classes, 
            ground_truth_track_id, ground_truth_annotations
        ) =  super().get_detections(image, frame_name, t_, additional_data, measurments)

        # Pad bounding boxes 
        pred_bbox_padded = expand_bbox_to_aspect_ratio(pred_bbox, self.cfg.expand_bbox_shape)

        return (
            pred_bbox, pred_bbox_padded, pred_masks, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        )
    
    def get_uv_distance(self, t_uv, d_uv):
        t_uv_tensor = torch.from_numpy(t_uv).to(RUNTIME_DEVICE).float()
        d_uv_tensor = torch.from_numpy(d_uv).to(RUNTIME_DEVICE).float()
        d_mask = d_uv_tensor[3:, :, :] > 0.5
        t_mask = t_uv_tensor[3:, :, :] > 0.5

        mask_dt = torch.logical_and(d_mask, t_mask)
        mask_dt = mask_dt.repeat(4, 1, 1)
        mask_ = torch.logical_not(mask_dt)

        t_uv_tensor[mask_] = 0.0
        d_uv_tensor[mask_] = 0.0

        with torch.no_grad():
            t_emb = self.HMAR.autoencoder_hmar(t_uv_tensor.unsqueeze(0), en=True)
            d_emb = self.HMAR.autoencoder_hmar(d_uv_tensor.unsqueeze(0), en=True)
        t_emb = t_emb.view(-1) / 10**3
        d_emb = d_emb.view(-1) / 10**3
        return (
            t_emb.cpu().numpy(),
            d_emb.cpu().numpy(),
            torch.sum(mask_dt).cpu().numpy() / 4 / 256 / 256 / 2,
        )

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    expand_bbox_shape: Optional[Tuple[int]] = (192,256)
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""

    cfg.device = RUNTIME_DEVICE.type
    log.info(f"Using runtime device: {cfg.device}")

    phalp_tracker = HMR2_4dhuman(cfg)

    phalp_tracker.track()

if __name__ == "__main__":
    main()
