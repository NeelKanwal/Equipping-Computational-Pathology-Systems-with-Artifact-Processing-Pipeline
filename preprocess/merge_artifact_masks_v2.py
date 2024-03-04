import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore all FutureWarning warnings that might flood the console log
warnings.simplefilter(action='ignore', category=DeprecationWarning) # Ignore all DeprecationWarning warnings that might flood the console log

import os
import json
from PIL import Image
# from histolab.slide import Slide
import matplotlib.pyplot as plt
from skimage.draw import polygon
import numpy as np
import cv2
import time
from skimage.morphology import closing, opening, dilation, square

def sav_fig(path, img, sav_name, cmap="RGB"):
    plt.clf()
    plt.axis("off")
    plt.title(None)
    if cmap == "gray":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.savefig(os.path.join(path, f"{sav_name}.png"), bbox_inches='tight', pad_inches=0)


def merge_masks(path, image_name, listofmasks_):
    # listofmasks = os.listdir(path)
    
    shape = Image.open(os.path.join(path, listofmasks_[0])).size
    output_mask = np.full((shape[1], shape[0]), False)
    for img in listofmasks_:
        mask = Image.open(os.path.join(path, img)).convert("L")
        mask = mask.resize(shape)
        output_mask = output_mask | np.asarray(mask)

    merged_mask = dilation(output_mask, square(2))
    merged_mask = merged_mask.astype(int) * 255
    merged_mask = Image.fromarray(merged_mask.astype(np.uint8))
    sav_fig(path, merged_mask ,sav_name=f"{image_name}_merged", cmap='gray')

# sav_dir = "D:\\mask_from_xml\\qunatitative_test\\Inference\\s6"
sav_dir = "/path_to/new_WSIs/Inference/s5/"
directory = sav_dir

t_files = os.listdir(directory)
# list_ofmasks = [f for f in t_files if f.endswith("png")]
listofmasks_ = [f for f in t_files if f.endswith("png") and not (f.endswith("tissue.png") or f.endswith("merged.png") or f.endswith("thumbnail.png") or f.endswith("artifactfree.png"))]
print(f"Total masks {len(listofmasks_)}")
print(listofmasks_)


# image_mask_dict = {image_name: [mask for mask in list_ofmasks if mask.startswith(image_name)] for image_name in set(mask.split('.')[0] for mask in list_ofmasks)}
# print(image_mask_dict)
# for image_name, masks in image_mask_dict.items():
#     # print(image_name,"+", masks)
#     # Merge the masks for the current image using your merge function
merge_masks(sav_dir, "s6", listofmasks_)     


print("### FINISHED ######")