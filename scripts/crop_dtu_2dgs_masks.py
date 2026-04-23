# License: CC BY-NC 4.0 - see /LICENSE
import os
import argparse
from PIL import Image
from tqdm import tqdm


def crop_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    for filename in tqdm(png_files, desc=f"Cropping {os.path.basename(input_folder)}"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with Image.open(input_path) as img:
            if img.width != 1600 or img.height != 1200:
                raise ValueError(f"Unexpected image size for {filename}: {img.width}x{img.height}. Expected 1600x1200.")

            # Crop from top and left
            left = 46
            top = 38
            right = img.width
            bottom = img.height
            cropped_img = img.crop((left, top, right, bottom))  # Resulting size: 1554x1162

            cropped_img.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop masks in all DTU scans.")
    parser.add_argument('--data_folder', type=str, required=True, help='Path to DTU dataset folder preprocessed by 2DGS.')
    args = parser.parse_args()

    scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69',
              'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

    for scene in tqdm(scenes, desc="Processing scenes"):
        mask_input_path = os.path.join(args.data_folder, scene, 'mask')
        mask_output_path = os.path.join(args.data_folder, scene, 'masks')

        if os.path.exists(mask_input_path):
            crop_images_in_folder(mask_input_path, mask_output_path)
        else:
            tqdm.write(f"Folder not found: {mask_input_path}")

        # Change name of original folder 'mask' to 'masks_unncropped'
        if os.path.exists(mask_input_path):
            os.rename(mask_input_path, os.path.join(args.data_folder, scene, 'masks_uncropped'))
