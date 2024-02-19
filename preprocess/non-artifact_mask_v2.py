import numpy as np
import os
from PIL import Image
import time
from matplotlib import pyplot as plt


fname = "CZ542"
dataset_dir = f"D:\\mask_from_xml\\qunatitative_test\\Inference\\{fname}"
t_files = os.listdir(dataset_dir)


total_fusedmasks = [f for f in t_files if f.endswith("png")]
binarymask = [f for f in total_fusedmasks if f.split(".")[-2].split("_")[-1] == "tissue"][0]
print(binarymask)
binarymask_img = Image.open(os.path.join(dataset_dir, binarymask)).convert("L")
shape = binarymask_img.size
binarymask = np.asarray(binarymask_img).astype("float")


fusedmask = [f for f in total_fusedmasks if f.split(".")[-2].split("_")[-1] == "merged"][0]
fusedmask_r = Image.open(os.path.join(dataset_dir, fusedmask)).convert("L")
fusedmask_r = fusedmask_r.resize(shape)
fusedmask_r = np.asarray(fusedmask_r).astype("float")


output_mask = binarymask - fusedmask_r

Image.fromarray(output_mask).convert('L').save(f"{dataset_dir}/{fname}_artifactfree.png")
  
