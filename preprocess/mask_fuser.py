## This script, takes all artifact masks and fused them to get overall artifact mask. Which will be best to subtract from binary-mask and produce artifact free mask for nonartifact_tiler.py

import numpy as np
import os
from PIL import Image
import time

def mask_fuse(listofmasks):
    shape = Image.open(os.path.join(dataset_dir,listofmasks[0])).size
    output_mask = np.full((shape[1],shape[0]),False)
    for img in listofmasks:
        mask =  Image.open(os.path.join(dataset_dir,img))
        output_mask += np.asarray(mask)
    return Image.fromarray(output_mask)

initiate = time.time()
dataset_dir = os.path.join(os.getcwd(),"train")
t_files = os.listdir(dataset_dir)
total_wsi = [f for f in t_files if f.endswith("ndpi")]
print(f"Total files in {dataset_dir} directory are {len(total_wsi)}")
total_masks = [f for f in t_files if f.endswith("png") and f.split("_")[1].split(".")[0] != "thumbnail"]
print(f"Total masks in {dataset_dir} directory are {len(total_masks)}")

# unique_files = list(set([f.split("_")[0] for f in total_masks]))
for fi in total_wsi:
    print(f"Combining mask for {fi}")
    fusing_list = [mask for mask in total_masks if mask.split("_")[0]==fi]
    k = mask_fuse(fusing_list)
    k.save(f"{dataset_dir}/{fi}_fusedmask.png")
#
minutes_f = (time.time() - initiate)/60
print(f"Total time consumed for fusing masks in {dataset_dir} dataset  is {minutes_f:.2f} minutes.\n")
# plt.imshow(k)
# plt.axis("off")