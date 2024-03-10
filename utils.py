
""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""
# This file provides helpful functions for preprocessing, training, inference and post-processing.


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Ignore all FutureWarning warnings that might flood the console log
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, brier_score_loss, \
        accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score, precision_score
from scikitplot.metrics import plot_roc, plot_precision_recall, plot_lift_curve, plot_ks_statistic, \
        plot_calibration_curve
import copy

import os # Useful when running on windows.
os.environ["PATH"] = "path_to/openslide-win64-20171122/bin/" + ";" + os.environ["PATH"]
# os.environ["PATH"] = "path_to\\vips-dev-8.11\\bin" + ";" + os.environ["PATH"]
os.environ["PATH"] = "path_to/full_artifact_pipeline/vips-dev-8.11/bin/" + ";" + os.environ["PATH"]

import pyvips as vips
import openslide
# print("Pyips: ", vips.__version__)
# print("Openslide: ", openslide.__version__)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import time
from skimage.morphology import closing, opening, dilation, square, disk
from skimage.measure import label, regionprops
from PIL import Image
import json
import pprint
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from torchstat import stat
from torchvision import models
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.patches as mpatches
import timm

import sys
import numpy as np
import seaborn as sns
from tqdm import tqdm
import numpy as np
import torch
import os
from datetime import datetime
import torch
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
# from pytorchtools import EarlyStopping
# from gpytorch.variational import UnwhitenedVariationalStrategy
from torch.autograd import Variable
from torchvision import models
import torch
import torch.nn.functional as F
from torch import nn
import timm
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib

import torch.multiprocessing as mp
# mp.get_context('forkserver') # fixes problems with CUDA and fork ('forkserver')
# from multiprocessing import Pool
#from concurrent.futures import ThreadPoolExecutor
# from multiprocess import Pool
# from multiprocessing.pool import ThreadPool as Pool
# from pathos.multiprocessing import ProcessingPool as Pool
# Pool methods which allows tasks to be offloaded to the worker processes in a few different ways.
# mp.get_context('forkserver') # fixes problems with CUDA and fork ('forkserver')

#from thop import profile
#from thop import clever_format
#from deepspeed.profiling.flops_profiler import get_model_profile
#from deepspeed.profiling.flops_profiler import FlopsProfiler
#from pthflops import count_ops
#from flopth import flopth
#from fvcore.nn import FlopCountAnalysis
from numerize import numerize
#from ptflops import get_model_complexity_info

font = {'family': 'serif',
        'weight': 'normal',
        'size': 28}
plt.rc('font', **font)

colors = {"tissue": (0.9, 0.9, 0.9),
          "blood": (0.99, 0, 0),
          "damage": (0, 0.5, 0.8),
          "airbubbles": (0, 0.1, 0.5),
          "fold": (0, 0.9, 0.1),
          "blur": (0.99, 0.0, 0.50),
          "artifact": (0.3,0.5,0.8),
          "artifactfree": (0.6,0.3,0.5)}

colors_new = {"tissue": "brown",
          "blood": "red",
          "damage": "green",
          "airbubbles": "yellow",
          "fold": "cyan",
          "blur": "blue",
          "artifact": "purple",
          "artifactfree": "maroon"}

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,}

test_transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def sav_fig(path, img, sav_name, cmap="RGB"):
    plt.clf()
    plt.axis("off")
    plt.title(None)
    if cmap == "gray":
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.savefig(os.path.join(path, f"{sav_name}.png"), bbox_inches='tight', pad_inches=0)

def remove_small_regions(img, size2remove=100):
    img = closing(img, square(2))
    label_image = label(img)

    # Run through the image labels and set the small regions to zero
    props = regionprops(label_image)
    for region in props:
        if region.area < size2remove:
            minY, minX, maxY, maxX = region.bbox
            img[minY:maxY, minX:maxX] = 0
    return img

def create_binary_mask(wsi_dir, f, sav_path, downsize = 50):
    # using pyvips
    print("\n##########################################")
    print(f"Creating basic binary masks for {f}")
    st = time.time()
    file_pth = os.path.join(wsi_dir, f)
    img_400x = read_vips(file_pth)
    w, h = img_400x.width, img_400x.height
    print(f"Original image width: {w}, height: {h} ")
    if "#tissue#mask.png" not in os.listdir(sav_path):
        thumbnail = img_400x.resize(1/downsize)
        sav_fig(sav_path, thumbnail, sav_name="#thumbnail")
        # tissue_mask = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGBA2RGB)
        img_hsv = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2HSV)
        mask_HSV = cv2.inRange(img_hsv, (100, 10, 50), (180, 255, 255))
        mask = remove_small_regions(mask_HSV)
        maskInv = cv2.bitwise_not(mask)
        maskInv_closed = remove_small_regions(maskInv)
        binarymask = cv2.bitwise_not(maskInv_closed)
        binarymask_resized = cv2.resize(binarymask, (thumbnail.width, thumbnail.height))
        sav_fig(sav_path, binarymask_resized, sav_name="#tissue#mask", cmap='gray')
        # Image.fromarray(tissue_mask).save(f"{dataset_dir}/{fname}_binarymask.png")
        print(f"Time taken for creating binary mask is {time.time()-st:.2f} seconds")
    return w, h

def fetch(region, patch_size, x, y):
    return region.fetch(patch_size * x, patch_size * y, patch_size, patch_size)

def crop(region, patch_size, x, y):
    return region.crop(patch_size * x, patch_size * y, patch_size, patch_size)

def read_vips(file_path, level=0):
    if file_path.endswith("mrxs"): # mrxs are scanned with
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

def convert_wsi(path,file):
    # The access="sequential" will let libvips stream the image from the source JPEG,
    # rather than decoding the whole thing in advance.
    image = vips.Image.new_from_file(os.path.join(path, file))
    image.write_to_file(f"{file}.tif", tile=True, pyramid=True, compression="jpeg")
    #Pyramidal TIFF images store their levels in one of two ways -- either as pages of the document,
    # or using subifds. A page-based TIFF pyramid stores the pyramid with the full
    # resolution image in page 0, the half-resolution level in page 1, and so on. Use something like:

def create_patches_v2(location, file, path, patch_folder, workers=1,
                   patch_size=224, mask_overlap= 95.0):

    global extract_and_save_patch
    print(f"Creating patches for {file}, using {workers} CPUs out of {mp.cpu_count()}")
    # "C:\files" + "/" + "wsi_files"
    file_pth = os.path.join(location, file)
    st = time.time()
    if file.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
        # import pyvips as vips
        # print("MRXS file, loading file at 40x")
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=1,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=1,
                                                autocrop=True).flatten()

    else:
        try:
            img_400x = vips.Image.new_from_file(file_pth, level=0,   # 40x
                                                autocrop=True).flatten() #RGBA to RGB img_400x = img_400[:3]
        except:
            img_400x = vips.Image.new_from_file(file_pth, page=0,
                                                autocrop=True).flatten()

    w, h = img_400x.width, img_400x.height
    n_across = int(w/patch_size)
    n_down = int(h/patch_size)
    # n_across * n_down
    mask_path = os.path.join(path, "#tissue#mask.png")

    def extract_and_save_patch(x_cord, y_cord, file_path=file_pth, file_name=file, mask_path=mask_path,
                               patch_folder=patch_folder, patch_size=patch_size, mask_overlap=mask_overlap):
        if file.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
            print("MRXS file, loading file at 40x")
            try:
                img_400x = vips.Image.new_from_file(file_path, level=1,
                                                    autocrop=True).flatten()
            except:
                img_400x = vips.Image.new_from_file(file_path, page=1,
                                                    autocrop=True).flatten()
        else:
            try:
                img_400x = vips.Image.new_from_file(file_path, level=0,
                                                    autocrop=True).flatten()
            except:
                img_400x = vips.Image.new_from_file(file_path, page=0,
                                                    autocrop=True).flatten()

        w, h = img_400x.width, img_400x.height
        mask_w, mask_h = Image.open(mask_path).size
        resized_mask = vips.Image.new_from_file(mask_path).flatten() \
            .colourspace("b-w").resize(w/mask_w, vscale=h/mask_h)
        img_400x = vips.Region.new(img_400x)
        resized_mask = vips.Region.new(resized_mask)
        # fetch, crop, extract_area,
        patch_mask = fetch(resized_mask, patch_size, x_cord, y_cord)
        patch_mask = np.ndarray(buffer=patch_mask, dtype=np.uint8, shape=[patch_size, patch_size])
        if np.mean(patch_mask/255)*100 > mask_overlap:
            patch = fetch(img_400x, patch_size, x_cord, y_cord)
            patch = np.ndarray(buffer=patch, dtype=np.uint8, shape=[patch_size, patch_size, 3])
            fname = file_name.split(".")[0]
            x_start, y_start = x_cord*patch_size, y_cord*patch_size
            base_name = f"{fname}_{x_start}_{y_start}.png"
            patch_pil = Image.fromarray(patch)
            patch_pil.save(os.path.join(patch_folder, base_name))

    # if workers == 1:
    #     for y in range(0, n_down):
    #         for x in range(0, n_across):
    #             extract_and_save_patch(x, y)
    # else:
    list_cord = [(x, y) for x in range(0, n_across) for y in range(0, n_down)]
    with mp.Pool(processes=workers) as p: # multiprocessing.cpu_count()  # check available CPU counts
        p.starmap(extract_and_save_patch, list_cord)
        # args = [(file, img_400x, resized_mask, patch_folder,
        #          cord_tuple, patch_size, mask_overlap) for cord_tuple in list_cord]

    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"Patches created for {file} in {minutes:.2f} minutes.")
    # return tile_counter

def extract_and_save_patch(y_cord, file_path, file_name, mask_path,
                           patch_folder, patch_size=224, mask_overlap=95.0):
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

def create_patches(location, file, path, patch_folder,
                   workers=1, patch_size=224, mask_overlap=95.0):

    print(f"Creating patches for {file}, using {workers} CPU out of {mp.cpu_count()}")
    file_path = os.path.join(location, file)
    st = time.time()

    img_400x = read_vips(file_path)
    w, h = img_400x.width, img_400x.height
    n_down = int(h/patch_size)
    mask_path = os.path.join(path, "#tissue#mask.png")

    params = [(y, file_path, file, mask_path, patch_folder, patch_size, mask_overlap)
              for y in range(0, n_down)]

    with mp.Pool(processes=workers) as p:
        result = p.starmap(extract_and_save_patch, params)

    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"Patches created for {file} in {minutes:.2f} minutes.")

def data_generator(patch_folder, test_transform, batch_size=32, worker=1):
    print(f"\nLoading patches...........")
    # test_images = datasets.ImageFolder(root=patch_folder, transform= test_transform)
    test_images = custom_data_loader(patch_folder, test_transform)
    test_loader = DataLoader(dataset=test_images, batch_size=batch_size, shuffle=False, num_workers=worker, pin_memory=True)
    total_patches = len(test_images)
    print(f"total number of patches are {total_patches}")
    return test_loader, total_patches

def load_cnn_model(weight_loc, weights_name, num_classes=2, dropout=0.2):
    model = models.mobilenet_v3_large()
    model.classifier = custom_classifier(960, num_classes, dropout=dropout)
    best_model_wts = os.path.join(weight_loc, weights_name)
    model.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])
    model.eval()
    return model

def load_vit_model(weight_loc, weights_name, num_classes=2):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
    best_model_wts = os.path.join(weight_loc, weights_name)
    model.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])
    model.eval()
    return model

class custom_data_loader(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_dir = img_path
        self.transform = transform
        self.data_path = []
        file_list = os.listdir(self.img_dir)
        for img in file_list:
            self.data_path.append(os.path.join(self.img_dir, img))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image = Image.open(self.data_path[idx])
        label = 0
        if self.transform is not None:
            return self.transform(image), label
        else:
            return image, label

class custom_classifier(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.2):
        super(custom_classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # fully connected layer 1
        x = self.dropout(x)
        feat = F.relu(self.fc2(x)) # fully connected layer 2
        x = self.dropout(x)
        x = self.fc3(feat)   #fully connected layer 3
        return x, feat

def infer_multiclass(model, test_loader, use_prob_threshold = None):
    y_preds, probs, artifact_free, blood, blur, bubble, damage, fold = [], [], [], [], [], [], [], []
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            # ID to classes  {0: 'artifact_free', 1: 'blood', 2: 'blur',
            # 3: 'bubble', 4: 'damage', 5: 'fold'}
        try:
            output, _ = model(data)
        except:
            output = model(data)

        probabilities = F.softmax(output, dim=1)
        if use_prob_threshold is not None:
            preds = (probabilities >= use_prob_threshold)
            _, preds = torch.max(preds, 1)
        else:
            _, preds = torch.max(output, 1)
        probs.append(list(np.around(probabilities.detach().cpu().numpy(), decimals=3)))
        y_pred = preds.cpu().numpy()

        artifact_free.append(list((y_pred == 0).astype(int)))
        blood.append(list((y_pred == 1).astype(int)))
        blur.append(list((y_pred == 2).astype(int)))
        bubble.append(list((y_pred == 3).astype(int)))
        damage.append(list((y_pred == 4).astype(int)))
        fold.append(list((y_pred == 5).astype(int)))

        y_preds.append(list(y_pred))

    artifact_free = convert_batch_list(artifact_free)
    blood = convert_batch_list(blood)
    blur = convert_batch_list(blur)
    bubble = convert_batch_list(bubble)
    damage = convert_batch_list(damage)
    fold = convert_batch_list(fold)
    probs = convert_batch_list(probs)
    y_preds = convert_batch_list(y_preds)

    return y_preds, artifact_free, blood,  blur, bubble, damage, fold, probs

def infer_cnn(model, test_loader, use_prob_threshold = None):
    y_pred, probs = [], []
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
        output, _ = model(data)
        probabilities = F.softmax(output, dim=1)
        if use_prob_threshold is not None:
            preds = (probabilities >= use_prob_threshold)
            _, preds = torch.max(preds, 1)
        else:
            _, preds = torch.max(output, 1)
        probs.append(list(np.around(probabilities[:,1].detach().cpu().numpy(), decimals=3)))
        y_pred.append(list(preds.cpu().numpy()))
    return convert_batch_list(y_pred), convert_batch_list(probs)

def infer_vit(model, test_loader, use_prob_threshold = None):
    y_preds, probs = [], []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            if use_prob_threshold is not None:
                preds = (probabilities >= use_prob_threshold)
                _, preds = torch.max(preds, 1)
            else:
                _, preds = torch.max(output, 1)
            # probabilities = F.softmax(output, dim=1).detach().cpu().numpy()
            probs.append(list(np.around(probabilities[:,1].detach().cpu().numpy(), decimals=3)))
            y_preds.append(list(preds.cpu().numpy()))
        return convert_batch_list(y_preds),  convert_batch_list(probs)

def convert_batch_list(lst_of_lst):
    return sum(lst_of_lst, [])

def post_process_masks(dataf, mask_saving_path, wsi_shape, downsize=50, blur=True, blood=True,
                       damage=True, fold=True, airbubble=True, merged=True):
    # dataframe can be loaded from excel sheet instead
    # dataf = pd.read_excel(path_to_excel, engine='openpyxl')
    if blood:
        print("-----Producing masks for blood-------------")
        blood_df = dataf[dataf['blood'] == 1]
        mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize)) # h,w
        blood_mask = np.full(mask_shape, False)
        for name in blood_df['files'].to_list():
            # for patch naming style SUShud37_30_8064_6048.png
            x_cord = int(name.split(".")[0].split("_")[-2])
            y_cord = int(name.split(".")[0].split("_")[-1])
            blood_mask[int(y_cord/downsize), int(x_cord/downsize)] = True
            # blood_mask = dilation(blood_mask,square(2)).astype(int) * 255
            blood_mask = blood_mask.astype(int) * 255
        sav_fig(mask_saving_path, Image.fromarray(blood_mask.astype(np.uint8)).convert("L"), sav_name="#blood#mask", cmap='gray')
    if blur:
        print("-----Producing masks for blur-------------")
        blur_df = dataf[dataf['blur'] == 1]
        mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize)) # h,w
        blur_mask = np.full(mask_shape, False)
        for name in blur_df['files'].to_list():
            # for patch naming style SUShud37_30_8064_6048.png
            x_cord = int(name.split(".")[0].split("_")[-2])
            y_cord = int(name.split(".")[0].split("_")[-1])
            blur_mask[int(y_cord/downsize), int(x_cord/downsize)] = True
            # blur_mask = dilation(blur_mask,square(2)).astype(int) * 255
            blur_mask = blur_mask.astype(int) * 255
        sav_fig(mask_saving_path, Image.fromarray(blur_mask.astype(np.uint8)).convert("L"), sav_name="#blur#mask", cmap='gray')
    if damage:
        print("-----Producing masks for damaged tissue--")
        damage_df = dataf[dataf['damage'] == 1]
        mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize)) # h,w
        damage_mask = np.full(mask_shape, False)
        for name in damage_df['files'].to_list():
            # for patch naming style SUShud37_30_8064_6048.png
            x_cord = int(name.split(".")[0].split("_")[-2])
            y_cord = int(name.split(".")[0].split("_")[-1])
            damage_mask[int(y_cord/downsize), int(x_cord/downsize)] = True
            # damage_mask = dilation(damage_mask,square(2)).astype(int) * 255
            damage_mask = damage_mask.astype(int) * 255
        sav_fig(mask_saving_path, Image.fromarray(damage_mask.astype(np.uint8)).convert("L"), sav_name="#damage#mask", cmap='gray')
    if fold:
        print("-----Producing masks for folded tissue---")
        fold_df = dataf[dataf['fold'] == 1]
        mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize)) # h,w
        fold_mask = np.full(mask_shape, False)
        for name in fold_df['files'].to_list():
            # for patch naming style SUShud37_30_8064_6048.png
            x_cord = int(name.split(".")[0].split("_")[-2])
            y_cord = int(name.split(".")[0].split("_")[-1])
            fold_mask[int(y_cord/downsize), int(x_cord/downsize)] = True
            # fold_mask = dilation(fold_mask,square(2)).astype(int) * 255
            fold_mask = fold_mask.astype(int) * 255
        sav_fig(mask_saving_path, Image.fromarray(fold_mask.astype(np.uint8)).convert("L"), sav_name="#fold#mask", cmap='gray')
    if airbubble:
        print("-----Producing masks for airbubbles-------")
        airbubble_df = dataf[dataf['bubble'] == 1]
        mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize)) # h,w
        airbubbles_mask = np.full(mask_shape, False)
        for name in airbubble_df['files'].to_list():
            # for patch naming style SUShud37_30_8064_6048.png
            x_cord = int(name.split(".")[0].split("_")[-2])
            y_cord = int(name.split(".")[0].split("_")[-1])
            airbubbles_mask[int(y_cord/downsize), int(x_cord/downsize)] = True

            # airbubbles_mask = dilation(airbubbles_mask,square(2)).astype(int) * 255
            airbubbles_mask = airbubbles_mask.astype(int) * 255
        sav_fig(mask_saving_path, Image.fromarray(airbubbles_mask.astype(np.uint8)).convert("L"),
                sav_name="#airbubbles#mask", cmap='gray')
    if merged:
        print("-----Producing masks a merged mask--------")
        merge_masks_new(mask_saving_path)

def merge_masks(path):
    listofmasks = os.listdir(path)
    listofmasks = [f for f in listofmasks if f.endswith("png") and not (f.startswith("#tissue") or f.startswith("#merged")
                                        or f.startswith("#artifact") or f.startswith("#thumb") or f.startswith("#segmentation"))]
    shape = Image.open(os.path.join(path, listofmasks[0])).size
    output_mask = np.full((shape[1], shape[0]), False)
    for img in listofmasks:
        mask = Image.open(os.path.join(path, img)).convert("L")
        mask = mask.resize(shape)
        output_mask = output_mask | np.asarray(mask)

    merged_mask_o = output_mask.astype(int) * 255    
    merged_mask_o = Image.fromarray(merged_mask_o.astype(np.uint8))
    sav_fig(path, merged_mask_o ,sav_name="#original#merged#mask", cmap='gray')

    merged_mask = dilation(output_mask, square(3))
    merged_mask = merged_mask.astype(int) * 255
    merged_mask = Image.fromarray(merged_mask.astype(np.uint8))
    sav_fig(path, merged_mask ,sav_name="#merged#mask", cmap='gray')

def merge_masks_new(path):
    listofmasks = os.listdir(path)
    listofmasks = [f for f in listofmasks if f.endswith("png") and (f.startswith("#bl") or f.startswith("#bubble")
                                        or f.startswith("#fol") or f.startswith("#dam"))]
    shape = Image.open(os.path.join(path, listofmasks[0])).convert("L").size
    # output_mask = np.full((shape[1], shape[0]), False)
    resized_masks = []
    for img in listofmasks:
        mask = Image.open(os.path.join(path, img)).convert("L")
        mask = mask.resize(shape, resample=Image.LANCZOS)
        resized_masks.append(np.asarray(mask))

    output_mask = np.any(resized_masks, axis=0)
    merged_mask_o = output_mask.astype(int) * 255    
    merged_mask_o = Image.fromarray(merged_mask_o.astype(np.uint8))
    sav_fig(path, merged_mask_o ,sav_name="#original#merged#mask", cmap='gray')

    merged_mask = dilation(output_mask, square(2))
    merged_mask = merged_mask.astype(int) * 255
    merged_mask = Image.fromarray(merged_mask.astype(np.uint8))
    sav_fig(path, merged_mask ,sav_name="#merged#mask", cmap='gray')


def segmentation_color_mask_with_df(df, sav_path, wsi_shape, downsize=50):
    mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize), 3)# h,w
    segmentation_mask = np.zeros(mask_shape, dtype=np.uint8)
    for i, row in df.iterrows():

        name = row['files']
        x_cord = int(name.split(".")[0].split("_")[-2])
        y_cord = int(name.split(".")[0].split("_")[-1])

        patch_class = row['predicted_class']
        #{0: 'artifact_free', 1: 'blood', 2: 'blur', 3: 'bubble', 4: 'damage', 5: 'fold'}
        if patch_class == 0: # Artifact-free
            color = (128, 128, 128)  # Gray

        elif patch_class == 1: # Blood
            color = (255, 0, 0)  # Red

        elif patch_class == 2: # Blur
            color = (255, 165, 0)  # Orange

        elif patch_class == 3: # Airbubble
            color = (0, 255, 0)  # Green

        elif patch_class == 4: # Damage
            color = (255, 255, 0)  # Yellow

        elif patch_class == 5: # Fold
            color = (255, 65, 90)  # Pink

        segmentation_mask[int(y_cord/downsize), int(x_cord/downsize)] = color


    colors = [(0.5, 0.5, 0.5), (1, 0, 0), (1, 165/255, 0), (0,1,0), (1,1,0), (1, 65/255, 90/255)]
    labels = ['Artifact-free', 'Blood', 'Blur', 'Airbubble', 'Damage', 'Fold']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]

    plt.figure(figsize=(20, 20)) 
    plt.clf()
    plt.axis("off")
    plt.title(None)
    plt.imshow(segmentation_mask)
    # plt.colorbar(label='Class')
    plt.legend(handles=patches, loc='best', fontsize=16, framealpha=0.3, labelcolor='white', facecolor='white')
    plt.savefig(f"{sav_path}/#segmentation#mask_with_df.png", dpi=600, bbox_inches='tight', pad_inches=0)


def segmentation_color_mask_with_df_v2(df, sav_path, wsi_shape, downsize=50):
    mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize), 3)# h,w
    segmentation_mask = np.zeros(mask_shape, dtype=np.uint8)
    for i, row in df.iterrows():

        name = row['files']
        x_cord = int(name.split(".")[0].split("_")[-2])
        y_cord = int(name.split(".")[0].split("_")[-1])

        patch_class = row['predicted']
        #{0: 'artifact_free', 1: 'artifact'}
        if patch_class == 0: # Artifact-free
            color = (128, 128, 128)  # Gray

        elif patch_class == 1: # Artifact
            color = (255, 0, 0)  # Red

        segmentation_mask[int(y_cord/downsize), int(x_cord/downsize), :] = color

    colors_n = [(0.5, 0.5, 0.5), (1.0, 0, 0)]
    labels = ['Artifact-free', 'Artifact']
    patches = [mpatches.Patch(color=colors_n[i], label=labels[i]) for i in range(len(colors_n))]

    plt.figure(figsize=(20, 20))    
    plt.clf()
    plt.axis("off")
    plt.title(None)
    plt.imshow(segmentation_mask)
    # plt.colorbar(label='Class')
    plt.legend(handles=patches, loc='best', fontsize=16, framealpha=0.3, facecolor="white")
    plt.savefig(f"{sav_path}/#segmentation#mask_with_df.png", dpi=600, bbox_inches='tight', pad_inches=0)
        

def post_process_mask_v2(dataf, mask_saving_path, wsi_shape, downsize=50):
    # used in binary pipelines to produce artifact-free mask
    # dataframe can be loaded from excel sheet instead
    # dataf = pd.read_excel(path_to_excel, engine='openpyxl')
   
    print("-----Producing mask for artifacts-------")
    a_df = dataf[dataf['predicted'] == 1]
    mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize))# h,w
    mask = np.full(mask_shape, False)
    for name in a_df['files'].to_list():
        #print(name)
        x_cord = int(name.split(".")[0].split("_")[-2])
        y_cord = int(name.split(".")[0].split("_")[-1])
        mask[int(y_cord/downsize), int(x_cord/downsize)] = True
 
    artifact_mask =  mask.astype(int) * 255
    artifact_mask = Image.fromarray(artifact_mask.astype(np.uint8))
    sav_fig(mask_saving_path, artifact_mask, sav_name="#artifact#mask", cmap='gray')

    print("-----Producing mask for artifacts-free-------")
    afree_df = dataf[dataf['predicted'] == 0]
    mask_shape = (round(wsi_shape[1]/downsize), round(wsi_shape[0]/downsize))# h,w
    mask = np.full(mask_shape, False)
    # print("Length of artifactFree_df", len(afree_df))
    for name in afree_df['files'].to_list():
        # print(name)
        # for patch naming style SUShud37_30_8064_6048.png
        x_cord = int(name.split(".")[0].split("_")[-2])
        y_cord = int(name.split(".")[0].split("_")[-1])
        mask[int(y_cord/downsize), int(x_cord/downsize)] = True

    artifactfree_mask_o =  mask.astype(int) * 255
    artifactfree_mask_o = Image.fromarray(artifactfree_mask_o.astype(np.uint8))
    sav_fig(mask_saving_path, artifactfree_mask_o, sav_name="#original#artifactfree#mask", cmap='gray')

    # artifactfree_mask = Image.fromarray(mask).convert("L")   
    artifactfree_mask = dilation(mask, square(2)) # removing small holes from black regions
    artifactfree_mask = closing(artifactfree_mask, square(2)) # closing small holes
    artifactfree_mask =  artifactfree_mask.astype(int) * 255
    artifactfree_mask = Image.fromarray(artifactfree_mask.astype(np.uint8))
    sav_fig(mask_saving_path, artifactfree_mask, sav_name="#artifactfree#mask", cmap='gray')




def artifact_free_mask(path):
    binary_mask = Image.open(os.path.join(path, "#tissue#mask.png")).convert("L")
    artifact_mask = Image.open(os.path.join(path, "#merged#mask.png")).convert("L")
    shape = binary_mask.size
    binary_mask = np.asarray(binary_mask.resize(shape, resample= Image.LANCZOS), dtype=np.bool)
    artifact_mask = np.asarray(artifact_mask.resize(shape, resample= Image.LANCZOS), dtype=np.bool)
    output_mask = binary_mask.astype(int) - artifact_mask.astype(int)
    output_mask = (output_mask == 1)

    artifactfree_mask_o =  output_mask.astype(int) * 255
    artifactfree_mask_o = Image.fromarray(artifactfree_mask_o.astype(np.uint8))
    sav_fig(path, artifactfree_mask_o, sav_name="#original#artifactfree#mask", cmap='gray')

    
    # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    # output_mask = dilation(output_mask, kernel)
    output_mask = opening(output_mask, square(2)).astype(int) * 255
    final_mask = Image.fromarray(output_mask.astype(np.uint8))
    sav_fig(path, final_mask, sav_name="#artifactfree#mask", cmap='gray')



def segmentation_color_mask(path, colors= colors):
    # used in multiclass and ensemble pipelines
    listofmasks = os.listdir(path)
    listofmasks = [f for f in listofmasks if f.endswith("png") and not (f.startswith("#merged") or
                        f.startswith("#segm") or f.startswith("#original") or f.startswith("#artifactfree") or f.startswith("#thumb"))]
    shape = Image.open(os.path.join(path, listofmasks[0])).size # Take shape of One_mask
    seg_img = np.zeros(shape=(shape[1], shape[0], 3))# make a 3D array

    legend_patch = []
    sorted_list = sorted(listofmasks, key=lambda x: x.startswith("#t"), reverse=True)
    for f in sorted_list:
        try:
            mask_type = f.split("#")[1]
        except: 
            mask_type = f.split("#")[0]
        mask = Image.open(os.path.join(path, f)).convert("L")
        mask_1 = np.asarray(mask.resize(shape), dtype=np.float32)
        seg_img[:, :, 0] += (mask_1 * (colors[mask_type][0]))
        seg_img[:, :, 1] += (mask_1 * (colors[mask_type][1]))
        seg_img[:, :, 2] += (mask_1 * (colors[mask_type][2]))
        legend_patch.append(mpatches.Patch(color=colors[mask_type], label=mask_type.capitalize()))

    plt.clf()
    plt.axis("off")
    plt.title(None)
    plt.legend(handles=legend_patch, loc='best', fontsize=12, framealpha=0.3, facecolor="white")

    im = Image.fromarray((seg_img*255).astype(np.uint8), 'RGB')
    plt.imshow(im)
    plt.savefig(f"{path}/#segmentation#mask.png", dpi=600, bbox_inches='tight', pad_inches=0)


def segmentation_color_mask_v2(path, colors= colors):
    # used for binary pipeline
    listofmasks = os.listdir(path)
    listofmasks = [f for f in listofmasks if f.endswith("png") and not (f.startswith("#merged") or
                        f.startswith("#segm") or f.startswith("#original") or f.startswith("#thumb") or f.startswith("#artifactfree"))]

    shape = Image.open(os.path.join(path, listofmasks[0])).size # Take shape of One_mask
    seg_img = np.zeros(shape=(shape[1], shape[0], 3))# make a 3D array
    legend_patch = []
    # first mask color comes first on the colormap
    # sort to make binary mask come first
    sorted_list = sorted(listofmasks, key=lambda x: x.startswith("#t"), reverse=True)

    for f in sorted_list:
        mask_type = f.split("#")[1]
        mask = Image.open(os.path.join(path, f)).convert("L")
        mask_1 = np.asarray(mask.resize(shape), dtype=np.float32)
        seg_img[:, :, 0] += (mask_1 * (colors[mask_type][0]))
        seg_img[:, :, 1] += (mask_1 * (colors[mask_type][1]))
        seg_img[:, :, 2] += (mask_1 * (colors[mask_type][2]))
        legend_patch.append(mpatches.Patch(color=colors[mask_type], label=mask_type.capitalize()))

    plt.clf()
    plt.axis("off")
    plt.title(None)
    plt.legend(handles=legend_patch, loc='best', fontsize=8, framealpha=0.3, facecolor="y")
    
    im = Image.fromarray((seg_img*255).astype(np.uint8), 'RGB')
    plt.imshow(im)
    plt.savefig(f"{path}/#segmentation#mask.png", dpi=600, bbox_inches='tight', pad_inches=0)


def refine_artifacts_wsi(path_to_wsi, path, name=None):
    artifact_free_mask(path)
    artifactfree_mask = os.path.join(path, "#artifactfree#mask.png")
    mask = vips.Image.new_from_file(artifactfree_mask).flatten()
    wsi = read_vips(path_to_wsi)
    # if path_to_wsi.endswith("mrxs"): # mrxs are scanned with
    #     print("File is MRXS")
    #     wsi = vips.Image.new_from_file(path_to_wsi, level=1, autocrop=True).flatten()
    # else:
    #     wsi = vips.Image.new_from_file(path_to_wsi, level=0, autocrop=True).flatten()
    mask = mask.resize(wsi.width / mask.width, vscale=(wsi.height / mask.height), kernel="nearest")
    wsi *= mask / 255.0
    wsi.write_to_file(f"{path}/refined_{name}.tiff",  tile=True, properties=True, tile_width=512,
                      tile_height=512, compression="jpeg", pyramid=True) ### CHECK THIS

def refine_artifacts_wsi_v2(path_to_wsi, path, name=None):
    artifactfree_mask = os.path.join(path, "#artifactfree#mask.png")
    mask = vips.Image.new_from_file(artifactfree_mask).flatten()
    if path_to_wsi.endswith("mrxs"): # mrxs are scanned with
        print("File is MRXS")
        wsi = vips.Image.new_from_file(path_to_wsi, level=1, autocrop=True).flatten()
    else:
        wsi = vips.Image.new_from_file(path_to_wsi, level=0, autocrop=True).flatten()
    mask = mask.resize(wsi.width / mask.width, vscale=(wsi.height / mask.height), kernel="nearest")
    wsi *= mask / 255.0
    wsi.write_to_file(f"{path}/refined_{name}.tiff",  tile=True, properties=True, tile_width=512,
                      tile_height=512, compression="jpeg", pyramid=True) ### CHECK THIS


def calculate_quality(path_to_masks):
    # used in multiclass/ensemble pipelines
    report = dict()
    listofmasks = os.listdir(path_to_masks)
    listofmasks = [f for f in listofmasks if f.endswith("png") and not
                                            (f.startswith("#merged") or f.startswith("#original#merged") or f.startswith("original") or f.startswith("#segm")
                                            or f.startswith("#thumb") or f.startswith("#artifactfree") or f.startswith("#tissue"))]
    artifact_mask_s = Image.open(os.path.join(path_to_masks, "#original#merged#mask.png")).convert("L").size
    print("Shape of artifact_mask, ", artifact_mask_s )
    baseline_matrix = Image.open(os.path.join(path_to_masks, "#tissue#mask.png")).convert("L").resize(artifact_mask_s)
    print("Shape of tissue_mask, ", baseline_matrix.size)
    baseline_matrix = np.asarray(baseline_matrix, dtype=np.bool)
    total_pixels = np.sum(baseline_matrix == 1)
    sorted_list = sorted(listofmasks, key=lambda x: x.startswith("#or"), reverse=False)
    for img in sorted_list:
        mask_sum = np.sum(np.asarray(Image.open(os.path.join(path_to_masks, img)).convert("L"), dtype=np.bool) == 1)
        label = img.split("#")[1]
        if label == "original":
            label = "artifact-free"
        report[label] = str(round(mask_sum/total_pixels * 100, 2)) + " %"
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(report)
    with open(f"{path_to_masks}/quality_report.json", "w") as f:
        json.dump(report, f, indent=4)

def calculate_quality_v2(path_to_masks):
    # used in binary pipeline
    report = dict()
    listofmasks = os.listdir(path_to_masks)
    listofmasks = [f for f in listofmasks if f.endswith("png") and f.startswith("#original")]
    baseline_matrix = np.asarray(Image.open(os.path.join(path_to_masks, "#tissue#mask.png")).convert("L"), dtype=np.bool)
    total_pixels = np.sum(baseline_matrix == 1)
    for img in listofmasks:
        mask_sum = np.sum(np.asarray(Image.open(os.path.join(path_to_masks, img)).convert("L"), dtype=np.bool) == 1)
        label = img.split("#")[1]
        report[label] = str(round(mask_sum/total_pixels * 100, 2)) + " %"
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(report)
    with open(f"{path_to_masks}/quality_report.json", "w") as f:
        json.dump(report, f, indent=4)

def check_tissue_region(patch):
    patch = np.asarray(patch.getdata())[:, 0]
    val = np.histogram(patch, bins=[100, 235, 255])[0]
    if val[0] < val[1]:
        return False
    else:
        return True

def extract_patches_coords(location, file, path, sav_patch_folder,
                           patch_size, use_mask_to_threshold=True,
                           level=0, overlap=0,
                           threshold=80.0, sav_patches=False):
    """
    Extracts patch cords from a whole slide image using pyvips.

    Parameters:
    slide_path (str): The path to the whole slide image.
    patch_size (int): The size of the patches to extract.
    level (int): The level of the slide to extract patches from. Default is 0.
    overlap (int): The overlap between adjacent patches. Default is 0.
    threshold (int): The threshold for patch selection. Default is 0.
    return_coordinates (bool): If True, return the spatial position of each patch. Default is False.

    Returns:
        coordinates (list): A list of tuples containing the spatial position of each patch
    """
    # Load the slide with pyvips
    file_path = os.path.join(location, file)
    st = time.time()

    slide = read_vips(file_path, level=level)
    width, height = slide.width, slide.height

    mask_path = os.path.join(path, "#tissue#mask.png")
    mask = vips.Image.new_from_file(mask_path)
    resized_mask = mask.resize(width/mask.width,
                               vscale=height/mask.height, kernel="nearest")

    # Compute the number of patches in each direction
    num_patches_x = (width - patch_size) // (patch_size - overlap) + 1
    num_patches_y = (height - patch_size) // (patch_size - overlap) + 1

    # Extract patches from the slide
    coordinates = []
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            x = i * (patch_size - overlap)
            y = j * (patch_size - overlap)
            if use_mask_to_threshold:
                mask_patch = resized_mask.crop(x, y, patch_size, patch_size)
                if mask_patch.avg()/2.55 > threshold:
                    coordinates.append((x, y))
                    if sav_patches:
                        patch = slide.crop(x, y, patch_size, patch_size)
                        base_name = f"{x}_{y}.png"
                        patch.write_to_file(os.path.join(sav_patch_folder, base_name))

            else:
            # Check if the patch meets the threshold criteria
                patch = slide.crop(x, y, patch_size, patch_size)
                if patch.avg()/2.55 > threshold:
                    coordinates.append((x, y))
                    if sav_patches:
                        base_name = f"{x}_{y}.png"
                        patch.write_to_file(os.path.join(sav_patch_folder, base_name))

    time_elapsed = time.time() - st
    minutes = time_elapsed/60
    print(f"Total patches {len(coordinates)} created for {file} in {minutes:.2f} minutes.")

    return coordinates

class WSI_Patch_Dataset(Dataset):
    def __init__(self, slide_path, coords_list, patch_size=224, transform=None):
        self.slide_path = slide_path
        self.coords_list = coords_list
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        # Load the slide image using pyvips
        # slide_image = vips.Image.new_from_file(self.slide_path, access='sequential')
        slide_image = vips.Image.new_from_file(self.slide_path, level=0,
                                            autocrop=True).flatten()

        # Get the coordinates and extract the patch
        x, y = self.coords_list[idx]
        patch = slide_image.extract_area(x, y, self.patch_size, self.patch_size).write_to_memory()

        # Convert the patch to a tensor and apply the transformation
        if self.transform:
            patch = self.transform(patch)
        return patch



def create_foreground_mask_vips(wsi_dir, f, save_path=None, downsize=50):
    # Open the slide image using PyVips
    print("\n##########################################")
    print(f"Creating basic binary masks for {f}")
    st = time.time()
    slide_path = os.path.join(wsi_dir, f)
    slide = read_vips(slide_path)
    if "#tissue#mask.png" not in os.listdir(save_path):
    # Downsize the image if requested
        if downsize != 1:
            # slide = slide.reduce(downsize, downsize)
            slide = slide.resize(1/downsize)
        # Convert the image to grayscale and threshold it to create a binary mask
        sav_fig(save_path, slide, sav_name="#thumbnail")
        gray = slide.colourspace('b-w')
        _, binary_mask = cv2.threshold(np.ndarray(buffer=gray.write_to_memory(), dtype=np.uint8, shape=[gray.height, gray.width]),
                                       0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # binary_mask = gray.more(threshold=0, direction='above')

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = ~binary_mask
        # Save the binary mask if requested
        if save_path is not None:
            # sav_path = os.path.join(save_path, "#binary#mask")
            # cv2.imwrite(sav_path, binary_mask)
            sav_fig(save_path, binary_mask, sav_name="#tissue#mask", cmap='gray')

    print(f"Time taken for creating binary mask {time.time()-st:.2f} seconds")
    return slide.width, slide.height

def infer_cnn_v2(test_loader, model, samples=5):
    model.eval()

    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

    y_pred, y_true, probs, feature, lower_0c, upper_0c, lower_1c, upper_1c, mean_1 = [], [], [], [], [], [], [], [], []
    for data, target in test_loader:
        temp_p = []
        # for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        for i in range(samples):  # Number of monte carlo simulations

            output, ftr = model(data)
            un, preds = torch.max(output, 1)
            probabilities = F.softmax(output, dim=1).detach().cpu().numpy()
            temp_p.append(probabilities)
        temp_p = np.array(temp_p)
        m_0, s_0 = temp_p[:, :, 0].mean(0), temp_p[:, :, 0].std(0)
        lower_0, upper_0 = m_0 - (s_0 * 1.96) / np.sqrt(5), m_0 + (s_0 * 1.96) / np.sqrt(5)

        m_1, s_1 = temp_p[:, :, 1].mean(0), temp_p[:, :, 1].std(0)
        lower_1, upper_1 = m_1 - (s_1 * 1.96) / np.sqrt(5), m_1 + (s_1 * 1.96) / np.sqrt(5)
        #
        lower_0c.append(list(lower_0))
        upper_0c.append(list(upper_0))
        mean_1.append(list(m_1))
        lower_1c.append(list(lower_1))
        upper_1c.append(list(upper_1))
        probs.append(list(probabilities))
        y_pred.append(list(preds.cpu().numpy()))
        y_true.append(list(target.cpu().numpy()))
        feature.append(list(ftr.detach().cpu().numpy()))
    return y_pred, y_true, probs, feature, lower_0c, upper_0c, mean_1, lower_1c, upper_1c

def infer_cnn_v3(test_loader, model, samples=5):
    model.eval()

    y_pred, y_true, mean_1 = [], [], []
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output, _ = model(data)
        un, preds = torch.max(output, 1)
        probabilities = F.softmax(output, dim=1).detach().cpu().numpy()

        m_1, s_1 = temp_p[:, :, 1].mean(0), temp_p[:, :, 1].std(0)

        mean_1.append(list(m_1))
        y_pred.append(list(preds.cpu().numpy()))
        y_true.append(list(target.cpu().numpy()))

    return convert_batch_list(y_pred), convert_batch_list(y_true),  convert_batch_list(mean_1)

def infer_cnn_v4(model,test_loader, use_prob_threshold = None):
    y_pred, probs, y_true = [], [], []
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output, _ = model(data)
        probabilities = F.softmax(output, dim=1)

        if use_prob_threshold is not None:
            preds = (probabilities >= use_prob_threshold)
            _, preds = torch.max(preds, 1)
        else:
            _, preds = torch.max(output, 1)

        probs.append(list(np.around(probabilities[:,1].detach().cpu().numpy(), decimals=3)))
        y_pred.append(list(preds.cpu().numpy()))
        y_true.append(list(target.cpu().numpy()))

    return convert_batch_list(y_pred), convert_batch_list(y_true),  convert_batch_list(probs)

def infer_vit_v3(model, test_loader):
    y_preds, probs = [], []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            probabilities = F.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            y_true.append(list(target.cpu().numpy()))
            probs.append(list(np.around(probabilities[:,1].detach().cpu().numpy(), decimals=3)))
            y_preds.append(list(preds.cpu().numpy()))
    return convert_batch_list(y_preds),convert_batch_list(y_true),  convert_batch_list(probs)


def infer_vit_v4(model, test_loader, use_prob_threshold = None):
    y_preds, probs, y_true = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            probabilities = F.softmax(output, dim=1)

            if use_prob_threshold is not None:
                preds = (probabilities >= use_prob_threshold)
                _, preds = torch.max(preds, 1)
            else:
                _, preds = torch.max(output, 1)

          
            y_true.append(list(target.cpu().numpy()))
            probs.append(list(np.around(probabilities[:,1].detach().cpu().numpy(), decimals=3)))
            y_preds.append(list(preds.cpu().numpy()))

    return convert_batch_list(y_preds),convert_batch_list(y_true),  convert_batch_list(probs)


def train_simple_transformer(model, criterion, optimizer, train_loader, epoch):
    model.train()
    train_losses = []
    correct = 0
    print(f"Training epoch: {epoch}")
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        try:
            output = model(data)
        except:
            output, _ = model(data)

        _, preds = torch.max(output, 1)
        loss = criterion(output, target)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        correct += preds.eq(target.view_as(preds)).cpu().sum()
    train_accuracy = (100. * correct / float(len(train_loader.dataset))).cpu().detach().numpy()
    train_loss = np.average(train_losses)
    return train_accuracy, train_loss

def val_simple_transformer(model, early_stopping, timestamp, test_loader, epoch, path, criterion):
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        stop = False
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            try:
                output = model(data)
            except:
                output, _ = model(data)

            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            valid_losses.append(loss.item())
            correct += preds.eq(target.view_as(preds)).cpu().sum()
        val_accuracy = (100. * correct / float(len(test_loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        early_stopping(valid_loss, model, epoch, timestamp, path)
        if early_stopping.early_stop:
            # stop_flag_count += 1
            if early_stopping.counter >= early_stopping.patience:
                stop = True
        print("Validation accuracy: {0:.3f} %\n".format(val_accuracy))
        return val_accuracy, valid_loss, stop

def epoch_test_transformer(model, loader, criterion):

    y_pred, y_true, probs = [], [], []
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        for data, target in loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            _, preds = torch.max(output, 1)
            y_pred.append(list(preds.detach().cpu().numpy()))
            y_true.append(list(target.cpu().numpy()))
            probabilities = F.softmax(output, dim=1)
            probs.append(list(probabilities.detach().cpu().numpy()))

            loss = criterion(output, target)
            # print(loss)

            try:
                valid_losses.append(loss.item())
            except:
                valid_losses.append(loss)

            correct += preds.eq(target.view_as(preds)).cpu().sum()

        val_accuracy = (100. * correct / float(len(loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        return y_pred, y_true, probs, val_accuracy, valid_loss

class EarlyStopping_v2:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', timestamp=0000, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.timestamp = timestamp
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, epoch, timestamp, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, timestamp, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, timestamp, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, epoch, timestamp, path):
        path_w = f"{path}/model_checkpoints"
        if not os.path.exists(os.path.join(os.getcwd(), path_w)):
            os.mkdir(os.path.join(os.getcwd(), path_w))
            print("\nDirectory for model checkpoints created.")
        sav_path = f"{path_w}/Epoch:{epoch}_{timestamp}.dat"
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}). \nSaving model to path...{sav_path}')
        state_dict = model.state_dict()
        torch.save({'model': state_dict}, sav_path)
        self.val_loss_min = val_loss

def make_pretty_cm(cf, group_names=None, categories='auto', count=True,
                   percent=True, cbar=True, xyticks=True, xyplotlabels=True, sum_stats=True,
                   figsize=None, cmap='Blues', title=None):
    

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]
    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))
        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.5)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

def make_cm(y_true, y_pred, classes):
    # labels = [‘True Neg’,’False Pos’,’False Neg’,’True Pos’]
    cm = confusion_matrix(y_true, y_pred)
    confusion_matrix_df = pd.DataFrame(cm, columns=classes)
    fig = plt.figure(figsize=(14, 14))
    fig = sns.heatmap(confusion_matrix_df, annot=True, fmt="d", cmap="coolwarm")
    fig.set(ylabel="True", xlabel="Predicted", title='DKL predictions')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    return cm

def get_class_distribution(dataset_obj):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}
    idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
    # print("Distribution of classes: \n", get_class_distribution(natural_img_dataset))
    return count_dict

def train_cnn(model, criterion, optimizer, train_loader, epoch):
    model.train()
    train_losses = []
    correct = 0
    print(f"Training epoch: {epoch}")
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        try:
            output, _, _ = model(data)
        except:
            output, _ = model(data)

        _, preds = torch.max(output, 1)
        loss = criterion(output, target)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        correct += preds.eq(target.view_as(preds)).cpu().sum()
    train_accuracy = (100. * correct / float(len(train_loader.dataset))).cpu().detach().numpy()
    train_loss = np.average(train_losses)
    # print("Training accuracy: {0:.3f} %\n".format(train_accuracy))
    return train_accuracy, train_loss

def val_cnn(model, early_stopping, timestamp, test_loader, epoch, path, criterion):
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        stop = False
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output, _ = model(data)
            _, preds = torch.max(output, 1)
            # Convert to probabilities if output is logsoftmax
            #  ps = torch.exp(log_ps)
            loss = criterion(output, target)
            valid_losses.append(loss.item())
            # Calculate accuracy
            # equals = pred == targets
            # accuracy = torch.mean(equals)
            correct += preds.eq(target.view_as(preds)).cpu().sum()

        val_accuracy = (100. * correct / float(len(test_loader.dataset))).detach().cpu().numpy()
        valid_loss = np.average(valid_losses)
        early_stopping(valid_loss, model, epoch, timestamp, path)
        if early_stopping.early_stop:
            # stop_flag_count += 1
            if early_stopping.counter >= early_stopping.patience:
                stop = True
        print("Validation accuracy: {0:.3f} %\n".format(val_accuracy))
        return val_accuracy, valid_loss, stop

def epoch_test_cnn(model, loader, criterion):
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct = 0
        for data, target in loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            try:
                output, _ = model(data)
            except:
                output = model(data)

            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            # print(loss)
            try:
                valid_losses.append(loss.item())
            except:
                valid_losses.append(loss)
            correct += preds.eq(target.view_as(preds)).cpu().sum()

        val_accuracy = (100. * correct / float(len(loader.dataset))).detach().cpu().numpy()
        # print(valid_losses)
        valid_loss = np.average(valid_losses)
        return val_accuracy, valid_loss


def extract_features(DenseNetModel, dataloader):
    f = []
    feature = DenseNetModel.features
    # features = torch.nn.Sequential(*list(DenseNetModel.children())[:-1])
    for data, target in dataloader:
        # for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        out = feature(data)
        out = F.relu(out, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1) # only works for inputs of 32 x 32
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1)).view(len(data), -1)
        f.append(list(out.detach().cpu().numpy()))
    return f

def make_pretty_cm_v3(cm, categories, figsize=(20,20), title=None):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    # cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                # annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                annot[i, j] = '%.2f%%\n%d' % (p, c)
            elif c == 0:
                annot[i, j] = '%.1f%%\n%d' % (0.0,0)
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=categories, columns=categories)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    vmin = np.min(cm)
    vmax = np.max(cm)
    off_diag_mask = np.eye(*cm.shape, dtype=bool)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=2.0)
    # sns.heatmap(cm, annot=annot, fmt='', cmap="coolwarm", ax=ax, xticklabels=categories, yticklabels=categories, vmin=0, vmax=1)
    # sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, xticklabels=categories, yticklabels=categories, vmin=vmin, vmax=vmax)
    # sns.heatmap(cm, annot=annot,fmt = '', cmap='OrRd',mask=~off_diag_mask, cbar=False, linewidth=1, ax=ax, xticklabels=categories, yticklabels=categories)


    sns.heatmap(cm_norm, annot=annot,fmt = '', cmap='Blues', cbar=True,  linewidth=1, ax=ax, xticklabels=categories, yticklabels=categories)
    
    # ax.xaxis.tick_top()
    # sns.heatmap(cm, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))
    plt.xticks(rotation=45, fontsize=24)
    plt.yticks(rotation=45, fontsize=24)
    plt.title(title, fontsize=32)


def infer_binary_v3(model, test_loader, use_prob_threshold = None):
    y_preds, probs, y_true = [], [], []
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            # ID to classes  {0: 'artifact_free', 1: 'blood', 2: 'blur',
            # 3: 'bubble', 4: 'damage', 5: 'fold'}
        try:
            output, _ = model(data)
        except:
            output = model(data)

        probabilities = F.softmax(output, dim=1)

        if use_prob_threshold is not None:
            preds = (probabilities >= use_prob_threshold)
            _, preds = torch.max(preds, 1)
        else:
            _, preds = torch.max(output, 1)

        probs.append(list(np.around(probabilities[:,0].detach().cpu().numpy(), decimals=3)))
        y_preds.append(list(preds.cpu().numpy()))
        y_true.append(list(target.cpu().numpy()))

    y_true = convert_batch_list(y_true)
    probs = convert_batch_list(probs)
    y_preds = convert_batch_list(y_preds)

    return y_true, y_preds, probs

def plot_roc_curve_v3(df, title=None):
    # Compute ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(df['truth_label'], df['truth_prob'], pos_label = 1)
    auc_score = roc_auc_score(df['truth_label'], df['truth_prob'])

    # Plot ROC curve
    plt.plot(fpr, tpr, marker='*', markeredgecolor='yellow', markeredgewidth=3, color='blue', linewidth=6, label='ROC curve (AUC = {:.3f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=7, label='Random Classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=24)
    # 1 – specificity (= false positive fraction = FP/(FP+TN))
    plt.ylabel('True Positive Rate (Sensitivity) ', fontsize=24)
    # sensitivity (= true positive fraction = TP/(TP+FN))
    plt.title(f'{title}', fontsize=28)

    # Find the best threshold
    
    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    best_threshold = thresholds[np.argmin(distances)]
    print('\nClosest Point to Left Corner,  Best Threshold: {:.2f}'.format(best_threshold))
    #Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    
    print('\nFrom Youden`s J Static,  Best Threshold: {:.2f}'.format(best_threshold))

    #Find the threshold that maximizes sensitivity (TPR)
    
    max_sensitivity_index = np.argmax(tpr)
    # max_sensitivity_index = np.argmin(fpr)
    max_sens = thresholds[max_sensitivity_index]
    
    # y_pred_best = np.where(y_pred >= max_sens_r, 1, 0)   # use best senstitivity to predict
    
    max_sens_r = np.round(max_sens, decimals=3)
    plt.plot(fpr[max_sensitivity_index], tpr[max_sensitivity_index], linewidth=8, marker='s', markersize=30, markerfacecolor="tab:red", color='red', label=f'Max. Sensitivity @ {max_sens_r:.3f}')
    best_threshold = np.round(best_threshold, decimals=3)
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], linewidth=8, marker='8', markersize=30, markerfacecolor="tab:blue", color='yellow', label=f'Best Threshold @ {best_threshold:.3f}')
    plt.legend(loc='lower right', fontsize=28, framealpha=0.3, facecolor="y")

    plt.show()

def plot_roc_curve_v4(df, sensitivity_val = 0.90, title=None):
    # Compute ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(df['truth_label'], df['truth_prob'], pos_label = 1)
    auc_score = roc_auc_score(df['truth_label'], df['truth_prob'])

    # Plot ROC curve
    plt.plot(fpr, tpr, marker='*', markeredgecolor='yellow', markeredgewidth=3, color='blue', linewidth=6, label='ROC curve (AUC = {:.3f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=7, label='Random Classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=24)
    # 1 – specificity (= false positive fraction = FP/(FP+TN))
    plt.ylabel('True Positive Rate (Sensitivity) ', fontsize=24)
    # sensitivity (= true positive fraction = TP/(TP+FN))
    plt.title(f'{title}', fontsize=28)

    # Find the best threshold
    
    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    best_threshold = thresholds[np.argmin(distances)]
    print('\nClosest Point to Left Corner,  Best Threshold: {:.2f}'.format(best_threshold))
    #Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    
    print('\nFrom Youden`s J Static,  Best Threshold: {:.2f}'.format(best_threshold))

    #Find the threshold that maximizes sensitivity (TPR)
    
   
    max_sensitivity_index = np.argmax(tpr >= sensitivity_val)
    # print("Max Sen Index ", max_sensitivity_index)
    max_sens = thresholds[max_sensitivity_index]
    print("Max Sen Prob ", max_sens)
    
    # y_pred_best = np.where(y_pred >= max_sens_r, 1, 0)   # use best senstitivity to predict
    
    max_sens_r = np.round(max_sens, decimals=3)
    plt.plot(fpr[max_sensitivity_index], tpr[max_sensitivity_index], linewidth=8, marker='s', markersize=30, markerfacecolor="tab:red", color='red', label=f'Max. Sensitivity @ {max_sens_r:.3f}')

    best_threshold = np.round(best_threshold, decimals=3)
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], linewidth=8, marker='8', markersize=30, markerfacecolor="tab:blue", color='yellow', label=f'Best Threshold @ {best_threshold:.3f}')
    plt.legend(loc='lower right', fontsize=28, framealpha=0.3, facecolor="y")

    plt.show()

def plot_confusion_matrix(cm, classes, normalize=True, title=None, cmap='tab20b', figsize=(14,14)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        total = np.sum(cm)
        cm = cm.astype('float') / total
        fmt = '.2f'
    else:
        fmt = 'd'

    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    df_cm_numbers = pd.DataFrame(cm, index=classes, columns=classes).applymap(lambda x: '{:.0f} ({:.2f}%)'.format(x*total, x*100))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_cm_numbers, annot=True, fmt='', cmap=cmap, cbar=True, ax=ax)

    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(classes, rotation=0, fontsize=12)

    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.title(title, fontsize=16)

    plt.show()


def assign_class(row):
    if row['predicted'] == 1:
        dic = {'blood_p':1, 'blur_p':2, 'bubble_p':3, 'damage_p':4, 'fold_p':5}
        columns = ['blood_p', 'blur_p', 'bubble_p', 'damage_p', 'fold_p']
        max_col = max(columns, key=lambda col: row[col])
        return dic[max_col] 
    else:
        return 0

def assign_class_v2(row):
    if row['predicted'] != 0:
        prob = row['probs']
        max_prob_index = np.argmax(prob)
        return max_prob_index
    else:
        return 0

def best_prob(row):
    if row['predicted'] == 1:
        max_prob = row[['blood_p', 'blur_p', 'bubble_p', 'damage_p', 'fold_p']].max()
        return np.round(max_prob, decimals=3)
    else:
        return 0

def max_prob(row):
    max_prob = row[['blood_p', 'blur_p', 'bubble_p', 'damage_p', 'fold_p']].max()
    return np.round(max_prob, decimals=3)
   

def make_binary_label(row):
    if row['ground_truth'] == 0:
        return 0
    else:
        return 1

def truth_prob_ensemb(row):
    max_prob = row[['blood_p', 'blur_p', 'bubble_p', 'damage_p', 'fold_p']].max()
    return np.round(1-max_prob, decimals=3)

def infer_multiclass_v3(model, test_loader, use_prob_threshold = None):
    y_preds, probs, artifact_free, blood, blur, bubble, damage, fold, y_true = [], [], [], [], [], [], [], [], []
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            # ID to classes  {0: 'artifact_free', 1: 'blood', 2: 'blur', # 3: 'bubble', 4: 'damage', 5: 'fold'}
        try:
            output, _ = model(data)
        except:
            output = model(data)

        probabilities = F.softmax(output, dim=1)

        if use_prob_threshold is not None:
            preds = (probabilities >= use_prob_threshold)
            _, preds = torch.max(preds, 1)
        else:
            _, preds = torch.max(output, 1)
        # probabilities = F.softmax(output, dim=1).detach().cpu().numpy()

        probs.append(list(np.around(probabilities.detach().cpu().numpy(), decimals=3)))
        y_pred = preds.cpu().numpy()

        artifact_free.append(list((y_pred == 0).astype(int)))
        blood.append(list((y_pred == 1).astype(int)))
        blur.append(list((y_pred == 2).astype(int)))
        bubble.append(list((y_pred == 3).astype(int)))
        damage.append(list((y_pred == 4).astype(int)))
        fold.append(list((y_pred == 5).astype(int)))

        y_preds.append(list(y_pred))
        y_true.append(list(target.cpu().numpy()))

    y_true = convert_batch_list(y_true)
    artifact_free = convert_batch_list(artifact_free)
    blood = convert_batch_list(blood)
    blur = convert_batch_list(blur)
    bubble = convert_batch_list(bubble)
    damage = convert_batch_list(damage)
    fold = convert_batch_list(fold)
    probs = convert_batch_list(probs)
    y_preds = convert_batch_list(y_preds)

    return y_true, y_preds, artifact_free, blood,  blur, bubble, damage, fold, probs