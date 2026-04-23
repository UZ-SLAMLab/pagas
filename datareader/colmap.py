# License: CC BY-NC 4.0 - see /LICENSE
import os
from typing import Any, Dict, Optional

import imageio.v2 as imageio
import numpy as np
import torch
import pycolmap

from rich.console import Console
from .normalize import (
    similarity_from_cameras,
    transform_cameras,
)
from .downscale import downscale_data
from .view_selection import build_neighbor_dict, write_views_file
from .utils import quaternion_to_rotation_matrix, get_rel_paths, get_mask_from_path, get_indices, ensure_colmap_sparse_model, save_colmap
from .image_ops import compute_smoothing_weight_color


console = Console()


class Parser:
    """COLMAP parser."""
    def __init__(
        self,
        data_dir: str,
        masks_name: str = 'masks',
        views_file: str = 'views.cfg',
        gt_depth_folder: str = '',
        scale_factors: list = [1],
        normalize: bool = False,
        depth_init_type: str = "est",
        num_context_views: int = -1,
        target_view: Optional[int] = None,
        starting_view: Optional[int] = 0,
        use_color_grad_weight: bool = False,
        result_dir: str = "",
        colmap_results_dir: str = "",
    ):
        self.data_dir = data_dir
        self.masks_name = masks_name
        self.gt_depth_folder = gt_depth_folder
        self.scale_factors = sorted(scale_factors, reverse=True)
        self.normalize = normalize
        self.depth_init_type = depth_init_type
        self.target_view = target_view
        self.starting_view = starting_view if target_view is None else 0
        self.use_color_grad_weight = use_color_grad_weight

        biggest_scale_factor = max(self.scale_factors)

        colmap_dir = os.path.join(data_dir, "sparse/0")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse\\0")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."
        self.colmap_dir = colmap_dir

        ensure_colmap_sparse_model(colmap_dir)
        self.rec = pycolmap.Reconstruction(colmap_dir)

        # Extract extrinsic matrices in world-to-camera format
        w2c_mats = []
        camera_ids = []
        K_dict = dict()
        imsize_dict = dict()  # width, height
        self.K_dict = dict()  # Dict of camera_id -> Dict of scale_factor -> K
        self.imsize_dict =  dict()  # Dict of camera_id -> Dict of scale_factor -> (width, height)
        self.rotated_image = dict()
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        same_intrinsics = False
        if len(self.rec.images) > 1 and len(self.rec.cameras) == 1:
            same_intrinsics = True
            cam = next(iter(self.rec.cameras.values()))

        for i, camera_id in enumerate(self.rec.images):
            if not (len(self.rec.images) > 1 and len(self.rec.cameras) == 1):
                cam = self.rec.cameras[camera_id]
            img = self.rec.images[camera_id]
            pose_cw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            rot = pose_cw.rotation.quat  #xyzw
            rot = quaternion_to_rotation_matrix(rot)
            trans = pose_cw.translation.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            camera_ids.append(camera_id)

            # Camera intrinsics
            fx, fy, cx, cy = cam.focal_length_x, cam.focal_length_y, cam.principal_point_x, cam.principal_point_y
            # Crop starting from the top left corner, so cx and cy are not affected
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  
            K_dict[camera_id] = K

            # Crop to be divisible by the biggest scale_factor
            imsize_dict[camera_id] = (cam.width//biggest_scale_factor*biggest_scale_factor, cam.height//biggest_scale_factor*biggest_scale_factor)  

            # Get distortion parameters            
            mask_dict[camera_id] = None
            self.K_dict[camera_id] = dict()  # Dict of camera_id -> Dict of scale_factor -> K
            # Dict of camera_id -> Dict of scale_factor -> (width, height)
            self.imsize_dict[camera_id] = dict()

        if len(self.rec.images) == 0:
            raise ValueError("No images found in COLMAP.")
        if cam.model.name not in [0, "SIMPLE_PINHOLE", 1, "PINHOLE", 2]:
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")
        if self.target_view is None:
            console.log(f"[bold blue]🎬 Running on scene {os.path.basename(data_dir)} with {len(self.rec.images) - self.starting_view} views")
        else:
            console.log(f"[bold blue]🎬 Running on scene {os.path.basename(data_dir)}, only view {self.target_view}")
        
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworld = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to image names anymore.
        image_names = [self.rec.images[camera_id].name for camera_id in self.rec.images]

        # Sort
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworld = camtoworld[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Normalize the world space
        if normalize:
            T1, scale = similarity_from_cameras(camtoworld)  # T1 includes change of scale
            camtoworld = transform_cameras(T1, camtoworld)
            transform = T1
        else:
            transform = np.eye(4)

        colmap_image_dir = os.path.join(data_dir, "images")
        assert os.path.exists(colmap_image_dir), f"Image folder {colmap_image_dir} does not exist."
        num_images = len(image_names)

        self.image_names = image_names  # List[str], (num_images,)
        self.camtoworld = camtoworld  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.transform = transform  # np.ndarray, (4, 4)
        self.normalization_scale = scale
        self.image_paths = [{} for _ in range(num_images)]
        self.mask_paths = [{} for _ in range(num_images)]
        self.depth_init_paths = [{} for _ in range(num_images)]
        self.gt_depth_paths = [{} for _ in range(num_images)]

        # If some scales are skipped, we still need them since the downscaling starts at 1 and halves each time
        all_scale_factors = [2**i for i in range(int(np.log2(biggest_scale_factor)), -1, -1)]
        for scale_factor in all_scale_factors:
            image_dir, mask_dir, depth_init_dir, gt_depth_dir = \
                downscale_data(scale_factor, data_dir, depth_init_type, masks_name, gt_depth_folder, num_images)

            # Downsampled images may have different names vs images used for COLMAP,
            # so we need to map between the two sorted lists of files.
            colmap_files = sorted(get_rel_paths(colmap_image_dir))
            image_files = sorted(get_rel_paths(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))
            image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
            [sub_dict.update({scale_factor: sub_path}) for sub_dict, sub_path in zip(self.image_paths, image_paths)]
            if mask_dir is not None:
                mask_files = sorted(get_rel_paths(mask_dir))
                colmap_to_mask = dict(zip(colmap_files, mask_files))
                mask_paths = [os.path.join(mask_dir, colmap_to_mask[f]) for f in image_names]
                [sub_dict.update({scale_factor: sub_path}) for sub_dict, sub_path in zip(self.mask_paths, mask_paths)]
            else:
                self.mask_paths = None  
            if depth_init_dir is not None:
                depth_init_files = sorted(get_rel_paths(depth_init_dir))
                colmap_to_depth_init = dict(zip(colmap_files, depth_init_files))
                depth_init_paths = [os.path.join(depth_init_dir, colmap_to_depth_init[f]) for f in image_names]
                [sub_dict.update({scale_factor: sub_path}) for sub_dict, sub_path in zip(self.depth_init_paths, depth_init_paths)]
            if os.path.exists(gt_depth_dir):
                gt_depth_files = sorted(get_rel_paths(gt_depth_dir))
                colmap_to_gt_depth = dict(zip(colmap_files, gt_depth_files))
                gt_depth_paths = [os.path.join(gt_depth_dir, colmap_to_gt_depth[f]) for f in image_names]
                [sub_dict.update({scale_factor: sub_path}) for sub_dict, sub_path in zip(self.gt_depth_paths, gt_depth_paths)]

            # We assume all images have the same resolution and some can be rotated 90º
            actual_image = imageio.imread(self.image_paths[0][scale_factor])[..., :3]  
            actual_height, actual_width = actual_image.shape[:2]  # after crop and resize
            colmap_width, colmap_height = imsize_dict[self.camera_ids[0]]  # after crop and before resize
            s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
            for camera_id, K in K_dict.items():
                K = K.copy()
                width, height = imsize_dict[camera_id]
                if height <= width:
                    self.rotated_image[camera_id] = False
                    K[0, :] *= s_width
                    K[1, :] *= s_height
                else:  # We rotate the vertical images
                    self.rotated_image[camera_id] = True
                    K[0, :] *= s_height
                    K[1, :] *= s_width
                self.imsize_dict[camera_id][scale_factor] = (max(actual_width, actual_height), min(actual_width, actual_height))  # Dimensions already rotated so width > height
                self.K_dict[camera_id][scale_factor] = K  # No rotated
                      
        # Load closest views indices
        closest_views_file = os.path.join(data_dir, views_file)
        if not os.path.exists(closest_views_file):
            console.print(f"[bold blue]Closest views file {closest_views_file} does not exist. Generating views.cfg")
            views_file = 'views.cfg'
            n_neighbors = 10
            self.indices = build_neighbor_dict(
                poses=[cam for cam in camtoworld], 
                n_neighbors=n_neighbors,
                intrinsics=[self.K_dict[i][1] for i in sorted(self.K_dict)], 
                im_sizes=[self.imsize_dict[i][1] for i in sorted(self.imsize_dict)]
            )            
            write_views_file(self.indices, image_names, os.path.join(data_dir, views_file))
            if num_context_views not in [-1, n_neighbors]:
                self.indices = get_indices(closest_views_file, num_context_views)
        else:
            console.print(f"[bold blue]👀 Loading closest views from {views_file}")
            self.indices = get_indices(closest_views_file, num_context_views)
        write_views_file(self.indices, image_names, os.path.join(result_dir, views_file), set_first_index_to_0=True)

        save_colmap(self, output_dir=colmap_results_dir, scale_factor=1, same_intrinsics=same_intrinsics)


class Dataset:
    """A simple dataset class."""
    def __init__(
        self,
        parser: Parser,
    ):
        self.parser = parser
        self.scale_factors = parser.scale_factors
        self.indices = parser.indices
        self.target_view = parser.target_view
        self.starting_view = parser.starting_view
        self.rotated_image = parser.rotated_image
        self.use_color_grad_weight = parser.use_color_grad_weight

        self.rotation_90 = np.array([
            [0., -1., 0., 0.],
            [1.,  0., 0., 0.],
            [0.,  0., 1., 0.],
            [0.,  0., 0., 1.]
        ], dtype=parser.camtoworld.dtype)

    def __len__(self):
        return len(self.indices) - self.starting_view if self.target_view is None else 1

    def __getitem__(self, item: int) -> Dict[str, Any]:        
        if self.target_view is not None:
            item = self.target_view
        else:
            item = item + self.starting_view
        context_item = self.indices[item]  # indices always start at 0
        camera_id = self.parser.camera_ids[item]  # colmap index, normally starts at 1
        context_camera_id = [self.parser.camera_ids[i] for i in context_item]

        # Get rotated image items
        rotated_image = self.rotated_image[camera_id]
        rotated_context_image = {i for i, key in enumerate(context_camera_id) if self.rotated_image[key]}
        rotated_images = [x + 1 for x in rotated_context_image]
        rotated_images = [0] + rotated_images if rotated_image else rotated_images

        # Camera poses
        camtoworld = self.parser.camtoworld[item]        
        if rotated_image:
            camtoworld = camtoworld @ self.rotation_90
        context_camtoworld = []
        for i, cam_id in enumerate(context_item):
            context_camtoworld_i = self.parser.camtoworld[cam_id]
            if i in rotated_context_image:
                context_camtoworld_i = context_camtoworld_i @ self.rotation_90
            context_camtoworld.append(context_camtoworld_i)
        context_camtoworld = np.stack(context_camtoworld)

        imsize = self.parser.imsize_dict[self.parser.camera_ids[item]]

        data = {
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "context_camtoworld": torch.from_numpy(context_camtoworld).float(),
            "image_id": item,
            "context_image_id": torch.tensor(context_item),
            "imsize": imsize,  # width, height
        }

        for scale_factor in self.scale_factors:
            # Images
            image = imageio.imread(self.parser.image_paths[item][scale_factor])[..., :3]
            if image.shape[0] > image.shape[1]:
                image = np.ascontiguousarray(np.rot90(image))
            context_image_path = [self.parser.image_paths[i][scale_factor] for i in context_item]
            all_image = [image]
            for i, path in enumerate(context_image_path):
                img = imageio.imread(path)[..., :3]
                if img.shape[0] > img.shape[1]:
                    # Rotate 90 degrees counter-clockwise
                    img = np.rot90(img)
                all_image.append(img)
            all_image = np.stack(all_image)

            # Intrinsics
            K = self.parser.K_dict[camera_id][scale_factor].copy()  # undistorted K
            if rotated_image:
                K = np.array([
                    [K[1, 1], 0, K[1, 2]],
                    [0, K[0, 0], image.shape[0] - 1 - K[0, 2]],  # cy = old width - 1 - cx
                    [0, 0, 1]
                ])
            context_K = []
            for i, cam_id in enumerate(context_camera_id):
                K_i = self.parser.K_dict[cam_id][scale_factor].copy()
                if i in rotated_context_image:
                    K_i = np.array([
                    [K_i[1, 1], 0, K_i[1, 2]],
                    [0, K_i[0, 0], all_image[i + 1].shape[0] - 1 - K_i[0, 2]],
                    [0, 0, 1]
                ])
                context_K.append(K_i)
            context_K = np.stack(context_K)

            # Initialization depth
            depth_init = np.load(self.parser.depth_init_paths[item][scale_factor])["arr_0"] * self.parser.normalization_scale  # H x W
            if rotated_image:
                depth_init = np.ascontiguousarray(np.rot90(depth_init))
            if (depth_init > 0.0).sum() == 0:
                raise ValueError(f"   --> Depth used for initialization is all zeros: {self.parser.depth_init_paths[item][scale_factor]}")
            all_depth_init = np.stack([depth_init] + [np.rot90(np.load(self.parser.depth_init_paths[idx][scale_factor])["arr_0"] * self.parser.normalization_scale) if i in rotated_context_image else np.load(self.parser.depth_init_paths[idx][scale_factor])["arr_0"] * self.parser.normalization_scale for i, idx in enumerate(context_item)])  # B x H x W
            depth_init = np.clip(depth_init, a_min=0.0, a_max=None)  # All depth must be >= 0.
            all_depth_init = np.clip(all_depth_init, a_min=0.0, a_max=None)

            # Masks
            if self.parser.mask_paths is None:
                mask = np.ones_like(depth_init[..., None], dtype=bool)                
                all_mask = np.ones_like(all_depth_init[..., None], dtype=bool)
            else:
                if scale_factor in self.parser.mask_paths[item]:
                    mask = get_mask_from_path(filepath=self.parser.mask_paths[item][scale_factor])
                    if rotated_image:
                        mask = np.ascontiguousarray(np.rot90(mask))
                    all_mask = np.stack([mask] + [np.rot90(get_mask_from_path(filepath=self.parser.mask_paths[idx][scale_factor])) if i in rotated_context_image else get_mask_from_path(filepath=self.parser.mask_paths[idx][scale_factor]) for i, idx in enumerate(context_item)])  # B x H x W x 1
            # Masks are a combination of the loaded mask and the depth init valid values
            mask = np.logical_and(mask, depth_init[..., None] > 0.)
            all_mask = np.logical_and(all_mask, all_depth_init[..., None] > 0.)

            # GT depth
            if scale_factor in self.parser.gt_depth_paths[item]:
                gt_depth = np.load(self.parser.gt_depth_paths[item][scale_factor])["arr_0"]  # H x W
                gt_depth *= self.parser.normalization_scale
                if rotated_image:
                    gt_depth = np.ascontiguousarray(np.rot90(gt_depth))
            
            all_image[~all_mask.squeeze()] = 0.0
            if 'gt_depth' in locals():
                gt_depth[~mask.squeeze()] = 0.0

            # Color gradient based weight
            inv_color_grad_weight = None
            if self.use_color_grad_weight:
                inv_color_grad_weight = compute_smoothing_weight_color(all_image / 255., np.squeeze(all_mask), min_color_grad=0.02, max_color_grad=0.1)

            data_i = {
                "K": torch.from_numpy(K).float(),
                "context_K": torch.from_numpy(context_K).float(),
                "all_image": torch.from_numpy(all_image).float(),
                "depth_init": torch.from_numpy(depth_init).float(),
                "all_mask": torch.from_numpy(all_mask)
            }

            if scale_factor in self.parser.gt_depth_paths[item]:
                data_i["gt_depth"] = torch.from_numpy(gt_depth).float()

            if inv_color_grad_weight is not None:
                data_i["inv_color_grad_weight"] = torch.from_numpy(inv_color_grad_weight).float()

            data[scale_factor] = data_i

        data["rotated_image"] = rotated_image
        data["rotated_images"] =  torch.tensor(rotated_images)

        return data
