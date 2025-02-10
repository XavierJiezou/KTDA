from PIL import Image, ImageFile
import os
from natsort import natsorted
from rich.progress import track
import argparse
from glob import glob
from typing import List, Tuple

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def get_args() -> Tuple[str, str,str]:
    parser = argparse.ArgumentParser(description="Invert and crop images")
    parser.add_argument("--input_folder", type=str, default="data/output",help="Path to the input folder containing the images")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Path to the output folder where the inverted and cropped images will be saved",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="res.pdf",
        help="Filename of the image to be inverted and cropped",
    )
    args = parser.parse_args()
    return args.input_folder, args.output_folder, args.filename


def invert_crop_images(input_folder:str,patch_size=256,work_name="Working")->Image.Image:
    filenames = natsorted(glob(os.path.join(input_folder, "*.png")))
    rows = [
        int(filename.split(os.path.sep)[-1].split(".")[0].split("_")[1])
        for filename in filenames
    ]
    cols = [
        int(filename.split(os.path.sep)[-1].split(".")[0].split("_")[2])
        for filename in filenames
    ]
    max_row = max(rows)
    max_col = max(cols)
    res_img = Image.new("RGB", (max_row * patch_size, max_col * patch_size))
    for filename in track(filenames, total=len(filenames),description=work_name):
        img = Image.open(filename)
        row = int(filename.split(os.path.sep)[-1].split(".")[0].split("_")[1])
        col = int(filename.split(os.path.sep)[-1].split(".")[0].split("_")[2])
        res_img.paste(img, (row * patch_size, col * patch_size))
    return res_img


def main():
    input_folder, output_folder,filename = get_args()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    img = invert_crop_images(input_folder, output_folder,filename)
    img.save(os.path.join(output_folder, filename))

if __name__ == "__main__":
    main()
