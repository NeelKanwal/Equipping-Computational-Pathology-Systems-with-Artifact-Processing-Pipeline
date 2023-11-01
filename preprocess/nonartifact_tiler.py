## This script create patches for artifact_free class.
import time
import os
os.environ["PATH"] = "E:\\Histology\\WSIs\\vips-dev-8.11\\bin" + ";" + os.environ["PATH"]
import pyvips as vips
import openslide
print("Pyips: ", vips.__version__)
print("Openslide: ", openslide.__version__)
from PIL import Image
from histolab.slide import Slide
from histolab.tiler import RandomTiler
from histolab.tiler import GridTiler
from histolab.masks import BinaryMask
import matplotlib.pyplot as plt
import numpy as np
num_tiles = 200 # Fix number of files for each WSI.
mag_level = 8
crop = False

initiate = time.time()
class MyMask(BinaryMask):
    def _mask(self, slide):
        # thumb = slide.thumbnail
        # thumb = slide.scaled_image(100)
        # temp = np.array([0,1])
        my_mask = np.asarray(Image.open(os.path.join(dataset_dir, mask_path)))
        # print(f"Size of my_mask is {my_mask.shape[1]}")
        if thumb.size[0] == my_mask.shape[1]:
            return my_mask
        else:
            print("Thumbnail and mask have different size\n")

tile_size = (224,224) # (256,256)
# dataset_dir = os.path.join(os.getcwd(), "train")   # "train/"  , "validation/"  #os.getcwd()
dataset_dir = "E:\\Histology\\WSIs\\excluded"
t_files = os.listdir(dataset_dir)
total_wsi = [f for f in t_files if f.endswith("ndpi")]
print(f"Total files in {dataset_dir} directory are {len(total_wsi)}")
total_masks = [f for f in t_files if f.endswith("png") and f.split("_")[1].split(".")[0] == "nonartifactmask"]
print(f"Total Non-Artifact masks in {dataset_dir} directory are {len(total_masks)}.")

# assert len(total_masks) == len(total_wsi)
count = 0
print("#####################################################################################\n")
print (f"Processing non-artifact patches from {dataset_dir} dataset.\n")

# for mask in total_masks:
for mask in ["CZ465.TP.I.I-9.ndpi_nonartifactmask.png"]:
    start = time.time()
    fname = mask.split("_")[0]
    curr_slide = Slide(os.path.join(dataset_dir, fname), os.path.join(dataset_dir, "Processed", "artifact_free"), autocrop=crop)
    thumb = curr_slide.scaled_image(100)
    mask_path = mask
    bin_mask = MyMask()
    print(f"Dimensions of file {fname} at level {mag_level}: {curr_slide.level_dimensions(level=mag_level)}.\n")
    #     grid_tiles_extractor = GridTiler(tile_size = tile_size, level=0, check_tissue=True,  tissue_percent= 30.0, pixel_overlap=0, prefix=f"{fname}_{label}_", suffix=".png")
    random_tiles_extractor = RandomTiler(
        tile_size=tile_size, n_tiles=num_tiles, level=mag_level, seed=42, check_tissue=True, tissue_percent=95, prefix=f"{fname}", suffix=".png")
    if not os.path.exists(os.path.join(dataset_dir, "Processed", "artifact_free")):
        print(f"Directory Created.\n")
        os.mkdir(os.path.join(dataset_dir, "Processed", "artifact_free"))

    print(f"Creating patches.....\n")
    # print(os.getcwd())
    random_tiles_extractor.extract(curr_slide, extraction_mask=bin_mask)
    # saves tiles in this format {prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}
    count = count + 1
    end = time.time()
    minutes = (end-start)/60
    print(f"Total time Elapsed: {minutes:.2f} minutes\n")

if count == len(total_masks):
    print(f"{dataset_dir} dataset processed successfully with total of {count} masks.\n")
    minutes_f = (time.time() - initiate)/60
    print(f"Total time consumed for {dataset_dir} dataset is {minutes_f:.2f} minutes.\n")