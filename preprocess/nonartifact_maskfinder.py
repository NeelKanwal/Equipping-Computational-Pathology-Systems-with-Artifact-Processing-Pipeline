# This script uses output from mask_fuser.py (overll produced artifact mask) to create artifact free mask, later used by nonartifact_tiler.py to produce artifact-free regions.

import numpy as np
import os
from PIL import Image
import time
from matplotlib import pyplot as plt

def mask_deducer(listofmasks):
    assert len(listofmasks) == 2
    binarymask = [f for f in listofmasks if f.split("_")[1].split(".")[0] =="binarymask"][0]
    binarymask = np.asarray(Image.open(os.path.join(dataset_dir, binarymask))).astype("float")
    # print(binarymask)
    fusedmask = [f for f in listofmasks if f.split("_")[1].split(".")[0] =="fusedmask"][0]
    fusedmask = np.asarray(Image.open(os.path.join(dataset_dir, fusedmask)).convert("L")).astype("float")
    # print(fusedmask)
    output_mask = binarymask - fusedmask
    # print(output_mask)
    # plt.imshow(Image.fromarray(output_mask))
    return Image.fromarray(output_mask)

initiate = time.time()
dataset_dir = os.path.join(os.getcwd(),"train")
t_files = os.listdir(dataset_dir)
total_wsi = [f for f in t_files if f.endswith("ndpi")]
print(f"Total files in {dataset_dir} directory are {len(total_wsi)}")
total_fusedmasks = [f for f in t_files if f.endswith("png") and f.split("_")[1].split(".")[0] == "fusedmask"]
total_binarymasks = [f for f in t_files if f.endswith("png") and f.split("_")[1].split(".")[0] == "binarymask"]
print(f"Total fused  masks in {dataset_dir} directory are {len(total_fusedmasks)}")
print(f"Total binary masks in {dataset_dir} directory are {len(total_binarymasks)}")
total_masks = total_binarymasks + total_fusedmasks

for fi in total_wsi:
    print(f"Finding non-artifact mask for {fi}")
    working_list = [mask for mask in total_masks if mask.split("_")[0]==fi]
    k = mask_deducer(working_list)
    k.convert('L').save(f"{dataset_dir}/{fi}_nonartifactmask.png")
#
minutes_f = (time.time() - initiate)/60
print(f"Total time consumed for creating non-artifact masks in {dataset_dir} dataset is {minutes_f:.2f} minutes.\n")