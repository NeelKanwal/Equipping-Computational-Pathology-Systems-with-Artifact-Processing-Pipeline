""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This is the main file provides ENSEMBLE of binary CNN and ViT models for end-to-end solution for what is mentioned in the paper.
# Update paths to processed datasets


# All libraries

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
import os

vipshome = 'path_to/vips-dev-8.12/bin/'
# vipshome = 'path_to/vips-dev-8.11/bin/'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips as vips
import openslide
print("Pyips: ", vips.__version__)
print("Openslide: ", openslide.__version__)

import pandas as pd
import torch
import numpy as np
import time
from torchvision import transforms
from utils import create_binary_mask, create_patches, data_generator, load_cnn_model, load_vit_model, \
    infer_cnn, infer_vit, post_process_masks, segmentation_color_mask, calculate_quality, \
    refine_artifacts_wsi, best_prob, truth_prob_ensemb, assign_class, truth_prob_ensemb,\
    segmentation_color_mask_v2, max_prob, segmentation_color_mask_with_df, merge_masks_new

from mmcv.cnn import get_model_complexity_info
# Alternate Libraries to
# from flopth import flopth
from numerize import numerize
# from calc_flops import calc_flops
# from fvcore.nn import FlopCountAnalysis
font = {'family': 'serif',
        'weight': 'normal',
        'size': 28}
plt.rc('font', **font)

# plt.style.use('science')
test_transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

torch.cuda.empty_cache()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
cuda_device = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
torch.cuda.empty_cache()
# torch.cuda.set_device(cuda_device)
print("Current CUDA device = ", torch.cuda.current_device())

def find_directories(directory):
    direct = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isdir(path):
            direct.append(file)
    return direct
path = "path_to/full_artifact_pipeline/new_WSIs/Inference/"
for dirs in find_directories(path):
    print(f"-------Inferring on {dirs} ----------")
    # Loading directory
    # Update paths here
    path_to_dataset = f"path_to/full_artifact_pipeline/new_WSIs/Inference/{dirs}/" # Use processed datasets from zenodo link in the repository
    wsi_dir = path_to_dataset
    save_dir = f"path_to/full_artifact_pipeline/new_WSIs/Inference/{dirs}/"

    
    models_location = "path_to/full_artifact_pipeline/model_weights/"
    
    choose_model = "cnns" # "vits", "cnns"
    evaluate_with_prob = 0.326 # Use this probablity for thresholding, 
    #set to None for not using this feature
    ## Ensemble CNN = 0.326
    ## Ensemble ViTs = 0.052

    # CNN Models Weights =
    blood_cnn = "blood_cnn.dat"
    blur_cnn = "blur_cnn.dat"

    fold_cnn = "fold_cnn.dat"
    damaged_cnn = "damage_cnn.dat"
    airbubble_cnn = "airbubble_cnn.dat"

    # ViT Models Weights =
    blood_vit = "blood_vit.dat"
    blur_vit = "blur_vit.dat"
    fold_vit = "fold_vit.dat"
    damaged_vit = "damage_vit.dat"
    airbubble_vit = "airbubble_vit.dat"


    # postprocessing output masks
    segmentation_mask = True
    refined_wsi = True
    quality_report = True
    cal_throughput = True
    

    fig = plt.subplots(figsize=(12, 12))
    

    # Other params
    
    
    
    downsize = 224
    patch_extraction_size = 224
    mask_overlap = 80.0
    batch_size = 64
    cpu_workers = 40
    # use_prob_threshold = 0.98 
    # None  # whether to give final prediction {0,1} based on certain probability

    torch.manual_seed(250)

    # read the files
    wsi_files = os.listdir(wsi_dir)
    wsi_files = [f for f in wsi_files if f.endswith("scn") or f.endswith("tif") or f.endswith("ndpi") or f.endswith("mrxs")]
    # get all files except temp directory containing patches
    # wsi_files = ['SUS033']
    print(f"Total files in {wsi_dir} directory are {len(wsi_files)}")

    if choose_model == "cnns":
        d_path = os.path.join(save_dir, "cnn_ensemble")
        if not os.path.exists(d_path):
            os.mkdir(d_path)
    else:
        d_path = os.path.join(save_dir, "vit_ensemble")
        if not os.path.exists(d_path):
            os.mkdir(d_path)

    # start patching process
    for f in wsi_files:
        st = time.time()
        # find binary mask to locate tissue on WSI
        fname =  f.split(".")[0]
        # path = os.path.join(d_path, fname)
        path = d_path
        # just take the name not extension
        if not os.path.exists(path):
            os.mkdir(path)
        w, h = create_binary_mask(wsi_dir, f, path, downsize=downsize)
        # print(f"Binary tissue mask created for {f}")
        # start splitting WSI into patches
        patch_folder = os.path.join(path, "patches")
        if not os.path.exists(patch_folder):
            os.mkdir(patch_folder)
            # assuming patches directory exists and patches are already created.
            create_patches(wsi_dir, f, path, patch_folder,  workers=cpu_workers,
                        patch_size=patch_extraction_size, mask_overlap=mask_overlap)

        data_loader, total_patches = data_generator(patch_folder,  test_transform=test_transform,
                                        batch_size=batch_size, worker=cpu_workers)

        if choose_model =="cnns": 
            print("\nLoading CNN ensemble of MobileNetv3")
            # blur
            blur_model = load_cnn_model(models_location, blur_cnn)
            blood_model = load_cnn_model(models_location, blood_cnn)
            fold_model = load_cnn_model(models_location, fold_cnn)
            damaged_model = load_cnn_model(models_location, damaged_cnn)
            airbubble_model = load_cnn_model(models_location, airbubble_cnn)

        else:
            print("\nLoading ViT ensemble of MobileNetv3")
            blur_model = load_vit_model(models_location, blur_vit)
            blood_model = load_vit_model(models_location, blood_vit)
            fold_model = load_vit_model(models_location, fold_vit)
            damaged_model = load_vit_model(models_location, damaged_vit)
            airbubble_model = load_vit_model(models_location, airbubble_vit)

        flops, params = get_model_complexity_info(blur_model, ((3,224,224)),
                                                  as_strings=False, print_per_layer_stat=False)
        million_param = numerize.numerize(params*5)
        gflops = numerize.numerize(flops*5)
        print(f"\nTotal model Mparam {million_param} and GFlops {gflops} in the {choose_model} ensemble.")

        if torch.cuda.is_available():
            print("Cuda is available")
            # model should be on cuda before selection of optimizer
            blur_model = blur_model.cuda()
            blood_model = blood_model.cuda()
            damaged_model = damaged_model.cuda()
            fold_model = fold_model.cuda()
            airbubble_model = airbubble_model.cuda()

        print("\n########### Inference Starts ##############")
        st2 = time.time()
        if evaluate_with_prob is not None:
            print("Using probablity thresholding @ ", evaluate_with_prob)
        if choose_model =="cnns":     

            blur_pred, blur_prob = infer_cnn(blur_model, data_loader, use_prob_threshold=evaluate_with_prob)
            blood_pred, blood_prob = infer_cnn(blood_model, data_loader, use_prob_threshold=evaluate_with_prob)
            damaged_pred, damaged_prob = infer_cnn(damaged_model, data_loader, use_prob_threshold=evaluate_with_prob)
            fold_pred, fold_prob = infer_cnn(fold_model, data_loader, use_prob_threshold=evaluate_with_prob)
            airbubble_pred, airbubble_prob = infer_cnn(airbubble_model, data_loader, use_prob_threshold=evaluate_with_prob)

        else: 
            blur_pred,  blur_prob = infer_vit(blur_model, data_loader, use_prob_threshold=evaluate_with_prob)
            blood_pred,  blood_prob = infer_vit(blood_model, data_loader, use_prob_threshold=evaluate_with_prob)
            damaged_pred,  damaged_prob = infer_vit(damaged_model, data_loader, use_prob_threshold=evaluate_with_prob)
            fold_pred,  fold_prob = infer_vit(fold_model, data_loader, use_prob_threshold=evaluate_with_prob)
            airbubble_pred, airbubble_prob = infer_vit(airbubble_model, data_loader, use_prob_threshold=evaluate_with_prob)

        # setting them to boolean

        seconds = time.time()-st2
        minutes = seconds/60
        print(f"Time consumed in inference for {f} in {minutes:.2f} minutes.\n")

        # Calculate throughtput
        if cal_throughput:
            print("Throughput: {:.2f} patches/seconds".format(total_patches/seconds))

        blur_pred_b = np.array(blur_pred).astype(bool)
        blood_pred_b = np.array(blood_pred).astype(bool)
        damaged_pred_b = np.array(damaged_pred).astype(bool)
        fold_pred_b = np.array(fold_pred).astype(bool)
        airbubble_pred_b = np.array(airbubble_pred).astype(bool)

        # ensemble output
        artifact_list = [blur_pred_b[i] | blood_pred_b[i] | damaged_pred_b[i] | fold_pred_b[i] | airbubble_pred_b[i]
                         for i in range(len(blur_pred))]
        artifact_list = [a.astype(int) for a in artifact_list]

        file_names = [im.split("/")[-1] for im in data_loader.dataset.data_path]
        data = {"files": file_names, "predicted": artifact_list,"blur_p": blur_prob, "blood_p": blood_prob,
        "bubble_p": airbubble_prob,  "damage_p": damaged_prob, "fold_p": fold_prob, "blur": blur_pred, "blood": blood_pred,
        "bubble": airbubble_pred, "damage": damaged_pred, "fold": fold_pred}

        dframe = pd.DataFrame(data)
        
        dframe.insert(1, 'artifact_p', dframe.apply(max_prob, axis=1))
        dframe.insert(2, 'afree_p', dframe.apply(truth_prob_ensemb, axis=1))
        dframe.insert(3, 'predicted_class', dframe.apply(assign_class, axis=1))

        if evaluate_with_prob is not None:
            print(f"Probablity threshold of {evaluate_with_prob} used for determining overall prediction.\n")

        with pd.ExcelWriter(f"{path}/ensemble_{choose_model}_predictions.xlsx") as wr:
            dframe.to_excel(wr, index=False)

        print("########### Postprocessing Starts ##########")
        print(f"Using probabiliy threshold {evaluate_with_prob}.")
        # postprocess from dataframe
        st3 = time.time()
        post_process_masks(dframe, path, wsi_shape=(w, h), downsize=downsize)

        if segmentation_mask:
            st4 = time.time()
            # segmentation_color_mask(path)
            segmentation_color_mask_with_df(dframe, sav_path= path, wsi_shape= (w, h), downsize=downsize)
            minutes3 = (time.time()-st4)/60
            print(f"Created color segmentation mask, time consumed {minutes3:.2f} minutes.")
        if refined_wsi:
            st5 = time.time()
            refine_artifacts_wsi(os.path.join(wsi_dir, f), path, name=f"{fname}_ensemble_{choose_model}")
            minutes4 = (time.time()-st5)/60
            print(f"Refined {f} for artifacts, time consumed {minutes4:.2f} minutes.\n")
        if quality_report:
            st6 = time.time()
            # read artifact masks from path and save the json file with percentage of artifacts
            calculate_quality(path)
            minutes5 = (time.time()-st6)/60
            print(f"\nPrepared quality report for {f}, time consumed {minutes5:.2f} minutes.")

        minutes = (time.time()-st3)/60
        print(f"Time consumed in post-processing for {f} in {minutes:.2f} minutes.\n")

        minutes = (time.time()-st)/60
        print(f"Total for end-to-end processing for {f} in {minutes:.2f} minutes.")

        print("\n-------------------------------------------------")
        print("//////////////////////////////////////////////////")
        print("--------------------------------------------------")

