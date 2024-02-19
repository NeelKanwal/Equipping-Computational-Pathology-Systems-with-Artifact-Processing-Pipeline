import time
import os
os.environ["PATH"] = "E:\\Histology\\WSIs\\vips-dev-8.11\\bin" + ";" + os.environ["PATH"]
import pyvips as vips
import openslide
print("Pyips: ", vips.__version__)
print("Openslide: ", openslide.__version__)
from PIL import Image
from histolab.slide import Slide
from histolab.tiler import GridTiler
from histolab.masks import BinaryMask, BiggestTissueBoxMask, TissueMask
import matplotlib.pyplot as plt
import numpy as np

label_map = {'Blood': "blood",
             'Cauterized':  "damage",
             'Folding': "fold",
             'Blurry': "blur",
             'Others': "bubble"}

initiate = time.time()
class MyMask(BinaryMask):
    def _mask(self, slide):
        # thumb = slide.thumbnail
        # thumb = slide.level_dimensions(level=0)
        thumb = slide.scaled_image(100)
        thumb_size =thumb.size
        my_mask = np.asarray(Image.open(os.path.join(dataset_dir, mask_path)).convert('L'))
        # print(f"Size of my_mask is {my_mask.shape[1]}")
        if thumb_size[0] == my_mask.shape[1]:
            return my_mask
        else:
            print("Thumbnail and mask have different size\n")

crop = False
tile_size = (224, 224) # (256,256)
fname = 'CZ219'
dataset_dir = f"D:\\mask_from_xml\\qunatitative_test\\Inference\\{fname}"
total_masks = [f for f in t_files if f.endswith("png") and f.split("_")[-1].split(".")[0] != "thumbnail" and f.split("_")[-1].split(".")[0] != "artifact_free" and f.split("_")[-1].split(".")[0] != "merged" and f.split("_")[-1].split(".")[0] != "tissue"]
print(f"Total masks in {dataset_dir} directory are {len(total_masks)}")
count = 0
print("#####################################################################################\n")
print (f"Processing {dataset_dir} dataset.\n")

# for mask in total_masks:
for mask in total_masks:
    start = time.time()
    fname = mask.split("_")[0]
    label = label_map[mask.split("_")[-1].split(".")[0]]
    curr_slide = Slide(os.path.join(dataset_dir, fname), os.path.join(dataset_dir, "Processed", label), autocrop=crop)
    thumb = curr_slide.thumbnail
    mask_path = mask
    bin_mask = MyMask()
    print(f"Dimensions of file {fname} at level 7: {curr_slide.level_dimensions(level=1)}.\n")

    if not os.path.exists(os.path.join(dataset_dir, "Processed")):
        os.mkdir(os.path.join(dataset_dir, "Processed"))
        print(f"Directory Created.\n")
        if not os.path.exists(os.path.join(dataset_dir, "Processed", label)):
            os.mkdir(os.path.join(dataset_dir, "Processed", label))
            print(f"Directory for {label} Created.\n")
    
    grid_tiles_extractor = GridTiler(tile_size=tile_size, level=1, check_tissue=True,  tissue_percent=70.0, pixel_overlap=0, prefix=f"{fname}_{label}_", suffix=".png")
        

    print(f"Creating patches for \"{label}\" label.\n")
    grid_tiles_extractor.extract(curr_slide, extraction_mask= bin_mask)
    # saves tiles in this format {prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}
    count = count + 1
    end = time.time()
    minutes = (end-start)/60
    print(f"Total time Elapsed: {minutes:.2f} minutes\n")

if count == len(total_masks):
    print(f"{dataset_dir} dataset processed successfully with total of {count} masks.\n")
    minutes_f = (time.time() - initiate)/60
    print(f"Total time consumed for {dataset_dir} dataset is {minutes_f:.2f} minutes.\n")