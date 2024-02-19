import time
import os
os.environ["PATH"] = "E:\\Histology\\WSIs\\vips-dev-8.11\\bin" + ";" + os.environ["PATH"]
import pyvips as vips
import openslide
print("Pyips: ", vips.__version__)
print("Openslide: ", openslide.__version__)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp

label_map = {'artifactfree': "artifact_free",
                'Blood': "blood",
             'Cauterized':  "damage",
             'Folding': "fold",
             'Blurry': "blur",
             'Others': "bubble"}

def read_vips(file_path, level=0):
    if file_path.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
        # print("MRXS file, loading file at 40x")
        try:
            img_400x = vips.Image.new_from_file(file_path, level=level+1,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_path, page=level+1,
                                                autocrop=True).flatten()
    else:
        try:
            img_400x = vips.Image.new_from_file(file_path, level=level,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_path, page=level,
                                                autocrop=True).flatten()
    return img_400x

def crop(region, patch_size, x, y):
    return region.crop(patch_size * x, patch_size * y, patch_size, patch_size)

def extract_and_save_patch(y_cord, file_path, file_name, mask_path,
                           patch_folder, patch_size=224, mask_overlap=70.0):
    slide =  read_vips(file_path)
    mask = vips.Image.new_from_file(mask_path)
    resized_mask = mask.resize(slide.width/mask.width,
                               vscale=slide.height/mask.height, kernel="nearest")
    n_across = int(slide.width/ patch_size)
    for x_cord in range(n_across):
        patch_mask = crop(resized_mask, patch_size, x_cord, y_cord)
        if patch_mask.avg()/2.55 > mask_overlap:
            # print(f"average overlap of patch {patch_mask.avg()/2.55}")
            patch = crop(slide, patch_size, x_cord, y_cord)

            fname = file_name.split(".")[0]
            x_start, y_start = x_cord*patch_size, y_cord*patch_size

            base_name = f"{fname}_{x_start}_{y_start}.png"
            patch.write_to_file(os.path.join(patch_folder, base_name))

def create_patches(location, file, mask_path, label,
                   workers=1, patch_size=224, mask_overlap=70.0):

    print(f"Creating patches for {label}, using {workers} CPU out of {mp.cpu_count()}")
    file_path = os.path.join(location, file)
    st = time.time()

    img_400x = read_vips(file_path)
    w, h = img_400x.width, img_400x.height
    n_down = int(h/patch_size)
    # mask_path = os.path.join(path, "#tissue.png")

    
    sav_loc = os.path.join(location, "Processed", label)

    params = [(y, file_path, file, mask_path, sav_loc, patch_size, mask_overlap)
              for y in range(0, n_down)]

    with mp.Pool(processes=workers) as p:
        result = p.starmap(extract_and_save_patch, params)

    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"Patches created for {file} with {label} label in {minutes:.2f} minutes.")

tile_size = (224, 224) 
fname = 'CZ565'
dataset_dir = f"/nfs/student/neel/full_artifact_pipeline/new_WSIs/Inference/{fname}/"
t_files = os.listdir(dataset_dir)
print("Total masks", len(t_files))
total_masks = [f for f in t_files if f.endswith("png") and f.split("_")[-1].split(".")[0] != "thumbnail" and f.split("_")[-1].split(".")[0] != "merged" and f.split("_")[-1].split(".")[0] != "tissue"]
print(total_masks)
print(f"Total masks in {dataset_dir} directory are {len(total_masks)}")
count = 0
print("#####################################################################################\n")
print (f"Processing {dataset_dir} dataset.\n")

# for mask in total_masks:
for mask in total_masks:
    print(f"On mask {mask}")
    mask_path = os.path.join(dataset_dir, mask)
    label = label_map[mask.split("_")[-1].split(".")[0]]

    if not os.path.exists(os.path.join(dataset_dir, "Processed")):
        os.mkdir(os.path.join(dataset_dir, "Processed"))
        print(f"Directory Created.\n")
    if not os.path.exists(os.path.join(dataset_dir,"Processed", label)):
        sav_loc = os.path.join(dataset_dir,"Processed", label)
        os.mkdir(sav_loc)
        print(f"Directory for {label} Created.\n")

    create_patches(dataset_dir, fname+".mrxs", mask_path, label, workers=40)
    count += 1


if count == len(total_masks):
    print(f"{dataset_dir} dataset processed successfully with total of {count} masks.\n")
    