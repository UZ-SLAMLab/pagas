# License: CC BY-NC 4.0 - see /LICENSE

import math
import os
import cv2
import shutil
from rich.console import Console
from typing import List, Tuple
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from typing_extensions import Literal


console = Console()


def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False):
    """A context manager that does nothing if verbose is True. Otherwise it hides logs under a message.

    Args:
        msg: The message to log.
        spinner: The spinner to use.
        verbose: If True, print all logs, else hide them.
    """
    if verbose:
        return nullcontext()
    return console.status(msg, spinner=spinner)


def downsample_depth(image: np.ndarray, exclude_value: float) -> np.ndarray:
    # Check input shape and type
    assert image.ndim == 2 and image.dtype == np.float32
    h, w = image.shape
    assert h % 2 == 0 and w % 2 == 0, "Try deleting downscaled_data folder and running again."
    h_down, w_down = h // 2, w // 2

    # Extract subimages for downsampling
    image00 = image[::2, ::2]
    image01 = image[::2, 1::2]
    image10 = image[1::2, ::2]
    image11 = image[1::2, 1::2]

    # Create a mask to identify valid pixels
    mask00 = image00 != exclude_value
    mask01 = image01 != exclude_value
    mask10 = image10 != exclude_value
    mask11 = image11 != exclude_value

    # Count valid pixels in full image
    image_count = np.zeros((h_down, w_down), dtype=np.float32)
    image_count += mask00.astype(np.float32)
    image_count += mask01.astype(np.float32)
    image_count += mask10.astype(np.float32)
    image_count += mask11.astype(np.float32)

    # Generate downsampled image by averaging subimages
    return (image00 * mask00 + image01 * mask01 + image10 * mask10 + image11 * mask11) / image_count.clip(min=1.0)


def downscale_files(
    input_dir: str,
    output_dir: str,
    file_type: Literal["color-png", "mask-png", "depth-npz"],
    downscale_factor: int = None,
    num_images: int = 0,
    shapes: tuple = None,  # height, width
):
    with status(msg="[bold yellow]Downscaling data...", spinner="growVertical"):

        if file_type in ["color-png", "mask-png"]:
            paths = list(Path(input_dir).glob("*.png"))
        elif file_type == "depth-npz":
            paths = list(Path(input_dir).glob("*.npz"))
        assert len(paths) == num_images, f"Expected {num_images} files, but found {len(paths)}."

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if file_type == "color-png":
            shapes = []
            for path in paths:
                img = cv2.imread(path)
                for _ in range(int(math.log2(downscale_factor))):
                    img = cv2.pyrDown(img)
                path_out = os.path.join(output_dir, path.name)
                cv2.imwrite(path_out, img)
                shapes.append(img.shape[:2])
            return shapes

        elif file_type == "mask-png":
            threshold = 255//2
            for path in paths:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                for _ in range(int(math.log2(downscale_factor))):
                    img = cv2.pyrDown(img)
                img[img > threshold] = 255
                img[img <= threshold] = 0
                path_out = os.path.join(output_dir, path.name)
                cv2.imwrite(path_out, img)

        elif file_type == "depth-npz":
            for path in paths:
                arr = next(iter(np.load(path).values()))  # Load with any kind of key name
                for _ in range(int(math.log2(downscale_factor))):
                    arr = downsample_depth(arr, 0.0)
                path_out = os.path.join(output_dir, path.name)
                np.savez(path_out, arr)
        else:
            raise ValueError("Invalid file type. Files must be 'png' or 'npz'.")


def get_target_shapes(input_dir, divisable_factor=2):
    shapes = []
    image_paths = list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.jpeg"))
    for path in image_paths:
        img = cv2.imread(path)
        shapes.append(img.shape[:2])

    assert shapes != [], "No images found in the input directory."

    # modify shapes to be divisible by scale factor
    target_shapes = [(shape[0]//divisable_factor*divisable_factor, shape[1]//divisable_factor*divisable_factor) for shape in shapes]  # height, width

    if shapes == target_shapes: # no need to crop
        shapes = None

    return target_shapes, shapes 


def _resize_array(arr: np.ndarray, new_shape: Tuple[int, int], interp) -> np.ndarray:
    """Resize a 2D or 3D array to (H, W) using the requested OpenCV interpolation."""
    h, w = new_shape
    return cv2.resize(arr, (w, h), interpolation=interp)


def fit_images_to_shape(
    input_dir: str,
    output_dir: str,
    file_type: Literal[
        "color-png-jpg",
        "mask-png-jpg",
        "depth-npz",
    ],
    num_images: int = 0,
    shapes: List[Tuple[int, int]] | None = None,
    original_shapes: List[Tuple[int, int]] | None = None,
):
    """
    Copy, resize, or crop files so they match requested shapes.

    If *original_shapes* is provided, each file is first resized to the
    corresponding entry in *original_shapes*, then cropped to the entry in *shapes*.

    If *original_shapes* is not provided, the first file decides the action:
      * When the requested shape is larger than the source, the data are resized.
      * When the shape matches, the file is copied.
      * When the shape is smaller, the data are cropped.
    """
    with status(msg="[bold yellow]Adjusting data to input images shape...", spinner="growVertical"):
        # Locate paths
        if file_type in {"color-png-jpg", "mask-png-jpg"}:
            paths = sorted(
                list(Path(input_dir).glob("*.png")) + \
                list(Path(input_dir).glob("*.jpg")) + \
                list(Path(input_dir).glob("*.jpeg"))
            )
        else:
            paths = sorted(Path(input_dir).glob("*.npz"))

        assert len(paths) == num_images, f"Expected {num_images} files, found {len(paths)} in {input_dir}."
        if shapes is None:
            raise ValueError("Argument shapes must be provided")
        if original_shapes is not None and len(original_shapes) != num_images:
            raise ValueError("original_shapes length must match num_images")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Decide action only when original_shapes is absent
        if original_shapes is None:
            first_path = paths[0]
            if file_type == "color-png-jpg":
                first_img = cv2.imread(str(first_path), cv2.IMREAD_UNCHANGED)
            elif file_type == "mask-png-jpg":
                first_img = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE)
            else:
                first_img = next(iter(np.load(first_path).values()))
            src_h, src_w = first_img.shape[:2]
            tgt_h, tgt_w = shapes[0]

            if tgt_h > src_h or tgt_w > src_w:
                action = "resize"
            elif tgt_h == src_h and tgt_w == src_w:
                action = "copy"
            else:
                action = "crop"
        else:
            action = "resize_then_crop"  # special path when original_shapes is supplied

        # Process every file
        for i, path in enumerate(paths):
            # Always save images with .png for images sources
            save_name = (
                os.path.splitext(path.name)[0] + ".png"
                if file_type in {"color-png-jpg", "mask-png-jpg"}
                else path.name
            )
            path_out = os.path.join(output_dir, save_name)

            # Simple copy when possible
            if action == "copy":
                shutil.copy2(path, path_out)
                continue

            # Load
            if file_type == "color-png-jpg":
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            elif file_type == "mask-png-jpg":
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            else:
                img = next(iter(np.load(path).values()))

            # Interpolation
            interp = cv2.INTER_NEAREST if file_type == "mask-png-jpg" else cv2.INTER_LINEAR
            interp_fn = lambda arr, size: _resize_array(arr, size, interp)
            
            # Transformation pipeline
            if action == "resize":
                img_proc = interp_fn(img, shapes[i])
            elif action == "crop":
                h, w = shapes[i]
                img_proc = img[:h, :w] if img.ndim == 2 else img[:h, :w, ...]
            elif action == "resize_then_crop":
                # Step 1: resize to original_shapes[i]
                img_proc = interp_fn(img, original_shapes[i])
                # Step 2: crop to shapes[i]
                h, w = shapes[i]
                img_proc = img_proc[:h, :w] if img_proc.ndim == 2 else img_proc[:h, :w, ...]

            # Save
            if file_type in {"color-png-jpg", "mask-png-jpg"}:
                cv2.imwrite(path_out, img_proc)
            else:
                np.savez(path_out, img_proc)


def get_shapes(image_dir):
    shapes = []
    # some images can be rotated, so we need to check all images
    for path in list(Path(image_dir).glob("*.png")):
        img = cv2.imread(path)
        shapes.append(img.shape[:2])
    return shapes


def downscale_data(downscale_factor: int, data_dir: str, depth_init_type: str, masks_name: str, gt_depth_folder: str, num_images: int = 0):
    """Downscale data by a factor of downscale_factor"""
    assert downscale_factor >= 1, "Downscale factor must be equal or greater than 1"

    image_dir_suffix = f"_{downscale_factor}"
    subfolder_name = "downscaled_data"

    image_dir = os.path.join(data_dir, subfolder_name, "images" + image_dir_suffix)
    mask_dir = os.path.join(data_dir, subfolder_name, masks_name + image_dir_suffix)
    if depth_init_type in ["random", "zero"]:
        depth_init_folder_name = None
    else:
        depth_init_folder_name = depth_init_type
    depth_init_dir = os.path.join(data_dir, subfolder_name, depth_init_folder_name + image_dir_suffix) if depth_init_folder_name is not None else None
    gt_depth_dir = os.path.join(data_dir, subfolder_name, gt_depth_folder + image_dir_suffix) if gt_depth_folder != '' else ''

    shapes = None
    target_shapes = None
    original_shapes = None
    image_dir_original = os.path.join(data_dir, "images")
    if not os.path.exists(image_dir):
        image_dir_original = os.path.join(data_dir, subfolder_name, "images_1")
        if not os.path.exists(image_dir_original):
            image_dir_original = os.path.join(data_dir, "images")
            if not os.path.exists(image_dir_original):
                raise ValueError(f"Image folder {image_dir_original} does not exist.")
            else:
                target_shapes, original_shapes = get_target_shapes(input_dir=image_dir_original, divisable_factor=downscale_factor)
                fit_images_to_shape(
                    input_dir=image_dir_original,
                    output_dir=os.path.join(data_dir, subfolder_name, "images_1"),
                    file_type="color-png-jpg",
                    num_images=num_images,
                    shapes=target_shapes,
                    original_shapes=original_shapes,
                )
                console.print("[bold green]:tada: Done adjusting images 1x")
            image_dir_original = os.path.join(data_dir, subfolder_name, "images_1")
        if downscale_factor != 1:
            shapes = downscale_files(
                input_dir=image_dir_original,
                output_dir=image_dir,
                file_type="color-png",
                downscale_factor=downscale_factor,
                num_images=num_images,
            )
            console.print(f"[bold green]:tada: Done downscaling images {downscale_factor}x")
        image_dir_original = os.path.join(data_dir, "images")

    if not os.path.exists(mask_dir) and masks_name != "":
        mask_dir_original = os.path.join(data_dir, subfolder_name, masks_name + "_1")
        if not os.path.exists(mask_dir_original):
            mask_dir_original = os.path.join(data_dir, masks_name)
            if not os.path.exists(mask_dir_original):
                console.print(f"[bold yellow]Warning: Mask folder {mask_dir_original} does not exist. No loading masks.")
                mask_dir = None
            else:
                target_shapes, original_shapes = get_target_shapes(input_dir=image_dir_original, divisable_factor=downscale_factor)
                fit_images_to_shape(
                    input_dir=mask_dir_original,
                    output_dir=os.path.join(data_dir, subfolder_name, masks_name + "_1"),
                    file_type="mask-png-jpg",
                    num_images=num_images,
                    shapes=target_shapes,
                    original_shapes=original_shapes,
                )
                console.print("[bold green]:tada: Done adjusting masks 1x")
            mask_dir_original = os.path.join(data_dir, subfolder_name, masks_name + "_1")
        if downscale_factor != 1 and os.path.exists(mask_dir_original):
            shapes = get_shapes(image_dir) if shapes is None else shapes
            downscale_files(
                input_dir=mask_dir_original,
                output_dir=mask_dir,
                file_type="mask-png",
                downscale_factor=downscale_factor,
                num_images=num_images,
                shapes=shapes,
            )
            console.print(f"[bold green]:tada: Done downscaling masks {downscale_factor}x")
    if masks_name == "":
        mask_dir = None

    if depth_init_folder_name is not None:
        if not os.path.exists(depth_init_dir):
            depth_init_dir_original = os.path.join(data_dir, subfolder_name, depth_init_folder_name + "_1")
            if not os.path.exists(depth_init_dir_original):
                depth_init_dir_original = os.path.join(data_dir, depth_init_folder_name)
                if not os.path.exists(depth_init_dir_original):
                    raise ValueError(f"Depth folder {depth_init_dir_original} does not exist.")
                else:
                    target_shapes, original_shapes = get_target_shapes(input_dir=image_dir_original, divisable_factor=downscale_factor)
                    fit_images_to_shape(
                        input_dir=depth_init_dir_original,
                        output_dir=os.path.join(data_dir, subfolder_name, depth_init_folder_name + "_1"),
                        file_type="depth-npz",
                        num_images=num_images,
                        shapes=target_shapes,
                        original_shapes=original_shapes,
                    )
                    console.print("[bold green]:tada: Done adjusting initial depths 1x")
                depth_init_dir_original = os.path.join(data_dir, subfolder_name, depth_init_folder_name + "_1")
            if downscale_factor != 1:
                shapes = get_shapes(image_dir) if shapes is None else shapes
                downscale_files(
                    input_dir=depth_init_dir_original,
                    output_dir=depth_init_dir,
                    file_type="depth-npz",
                    downscale_factor=downscale_factor,
                    num_images=num_images,
                    shapes=shapes,
                )
                console.print(f"[bold green]:tada: Done downscaling initial depths {downscale_factor}x")

    if not os.path.exists(gt_depth_dir) and gt_depth_folder != '':
        gt_depth_dir_original = os.path.join(data_dir, subfolder_name, gt_depth_folder + "_1")
        if not os.path.exists(gt_depth_dir_original):
            gt_depth_dir_original = os.path.join(data_dir, gt_depth_folder)
            if not os.path.exists(gt_depth_dir_original):
                raise ValueError(f"GT depth folder {gt_depth_dir_original} does not exist.")
            else:
                target_shapes, original_shapes = get_target_shapes(input_dir=image_dir_original, divisable_factor=downscale_factor)
                fit_images_to_shape(
                    input_dir=gt_depth_dir_original,
                    output_dir=os.path.join(data_dir, subfolder_name, gt_depth_folder + "_1"),
                    file_type="depth-npz",
                    num_images=num_images,
                    shapes=target_shapes,
                    original_shapes=original_shapes,
                )
                console.print("[bold green]:tada: Done adjusting GT depths 1x")
            gt_depth_dir_original = os.path.join(data_dir, subfolder_name, gt_depth_folder + "_1")
        if downscale_factor != 1:
            shapes = get_shapes(image_dir) if shapes is None else shapes
            downscale_files(
                input_dir=gt_depth_dir_original,
                output_dir=gt_depth_dir,
                file_type="depth-npz",
                downscale_factor=downscale_factor,
                num_images=num_images,
                shapes=shapes,
            )
            console.print(f"[bold green]:tada: Done downscaling GT depths {downscale_factor}x")

    return image_dir, mask_dir, depth_init_dir, gt_depth_dir
