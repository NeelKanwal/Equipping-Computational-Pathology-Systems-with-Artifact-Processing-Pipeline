import os
# os.environ["PATH"] = "E:\\Histology\\WSIs\\vips-dev-8.11\\bin" + ";" + os.environ["PATH"] # Use this while working on windows
import pyvips as vips
import openslide
print("Pyips: ",vips.__version__)
print("Openslide: ",openslide.__version__)
from histolab.slide import Slide
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

dataset_dir = os.path.join(os.getcwd(),"test")
t_files = os.listdir(dataset_dir)
total_wsi = [f for f in t_files if f.endswith("ndpi")]
print(f"Total files in {dataset_dir} directory are {len(total_wsi)}")

count = 0
print("####################################################\n")
print (f"Creating basic binary masks for {dataset_dir} dataset.\n")

for fname in total_wsi:
    curr_slide = Slide(os.path.join(dataset_dir,fname),os.path.join(dataset_dir,fname))
    tissue_mask = curr_slide.scaled_image(100) # Reduce the binary mask size 100 times to easy to transport.
    tissue_mask = cv2.cvtColor(np.array(tissue_mask), cv2.COLOR_RGBA2RGB)
    tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2HSV)
    tissue_mask = tissue_mask[:, :, 1]
    _, tissue_mask = cv2.threshold(tissue_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    Image.fromarray(tissue_mask).save(f"{dataset_dir}/{fname}_binarymask.png")