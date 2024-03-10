#""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides inference code for ensembles of binary DCNN or ViT models mentioned in the paper.
# Update paths to processed datasets
import os 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'serif','weight':'normal','size'   : 36}
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (20, 20)

import pandas as pd
import numpy as np
import seaborn as sns
import os
import torch

ensemble = "vits" # "vits", "cnns"
BATCH_SIZE = 256
use_prob_threshold = 0.052 # Use this probablity for thresholding, set to None for not using this feature
## Ensemble CNN = 0.326
## Ensemble ViTs = 0.052

cuda_device = 2
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
torch.cuda.empty_cache()
sns.set_style("white")


from torch.utils.data import DataLoader
import time
import pprint
from datetime import datetime
import json
import torch, torchvision
from torchvision import datasets, models
import torchvision.transforms as transforms

from torch import nn
from utils import make_cm, make_pretty_cm, infer_vit_v4, infer_cnn_v4, load_cnn_model, load_vit_model , plot_roc_curve_v4
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, roc_auc_score, precision_score, accuracy_score, f1_score
from scikitplot.metrics import plot_roc, plot_precision_recall 

import timm

from utils import plot_confusion_matrix, assign_class, best_prob, make_binary_label,  make_pretty_cm_v3, truth_prob_ensemb

def find_directories(directory):
    direct = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isdir(path):
            direct.append(file)
    return direct

path = "path_to/full_artifact_pipeline/new_WSIs/Inference/"
for dirs in find_directories(path):
    print(f"-------Inferring on {dirs} using {ensemble} ensemble ----------")

    torch.manual_seed(250)
    sens_thresh = 0.98 # for plot_roc curve to show probablity that gives this.


    path_to_dataset = f"path_to/new_WSIs/Inference/{dirs}/Processed/" # Use processed datasets from zenodo link in the repository
    sav_dir = f"path_to/full_artifact_pipeline/new_WSIs/Inference/{dirs}/"

    if ensemble == "cnns":
        d_path = os.path.join(sav_dir, "cnn_ensemble")
        if not os.path.exists(d_path):
            os.mkdir(d_path)
    else:
        d_path = os.path.join(sav_dir, "vit_ensemble")
        if not os.path.exists(d_path):
            os.mkdir(d_path)
    
    models_location = "path_to/model_weights/" # Use models from model_weights in repository


    # Names of DCNN weights
    blood_cnn = "blood_cnn.dat"
    blur_cnn = "blur_cnn.dat"
    fold_cnn = "fold_cnn.dat"
    damaged_cnn = "damage_cnn.dat"
    airbubble_cnn = "airbubble_cnn.dat"

    # Names of ViT weights
    blood_vit = "blood_vit.dat"
    blur_vit = "blur_vit.dat"
    fold_vit = "fold_vit.dat"
    damaged_vit = "damage_vit.dat"
    airbubble_vit = "airbubble_vit.dat"

    val_compose = transforms.Compose([transforms.Resize((224, 224)),
        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    t = time.time()
    val_images = datasets.ImageFolder(root=path_to_dataset, transform=val_compose)
    idx2class = {v: k for k, v in val_images.class_to_idx.items()}
    classes_list = list(idx2class.values())
    print("ID to classes ", idx2class)
    num_classes = len(val_images.classes)
    val_loader = DataLoader(val_images, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

    print(f"Total data loading time in minutes: {(time.time() - t) / 60:.3f}")

   
    # blur
    if ensemble =="cnns":  
        print("\nLoading CNN ensemble of MobileNetv3\n")
        blur_model = load_cnn_model(models_location, blur_cnn)
        blood_model = load_cnn_model(models_location, blood_cnn)
        fold_model = load_cnn_model(models_location, fold_cnn)
        damaged_model = load_cnn_model(models_location, damaged_cnn)
        airbubble_model = load_cnn_model(models_location, airbubble_cnn)
    else:
        print("\nLoading CNN ensemble of VITs\n")
        blur_model = load_vit_model(models_location, blur_vit)
        blood_model = load_vit_model(models_location, blood_vit)
        fold_model = load_vit_model(models_location, fold_vit)
        damaged_model = load_vit_model(models_location, damaged_vit)
        airbubble_model = load_vit_model(models_location, airbubble_vit)

    if torch.cuda.is_available():
        print("Cuda is available\n")
        # model should be on cuda before selection of optimizer
        blur_model = blur_model.cuda()
        blood_model = blood_model.cuda()
        damaged_model = damaged_model.cuda()
        fold_model = fold_model.cuda()
        airbubble_model = airbubble_model.cuda()

    print("-------------- TESTING -------------------------")

    if use_prob_threshold is not None:
        print("Using thresholding @ ", use_prob_threshold)


    if ensemble == "cnns":    
        blur_pred, y_true, blur_prob = infer_cnn_v4(blur_model, val_loader, use_prob_threshold=use_prob_threshold)
        blood_pred, y_true1, blood_prob= infer_cnn_v4(blood_model, val_loader, use_prob_threshold=use_prob_threshold)
        damaged_pred, y_true2, damaged_prob = infer_cnn_v4(damaged_model, val_loader, use_prob_threshold=use_prob_threshold)
        fold_pred, y_true, fold_prob = infer_cnn_v4(fold_model, val_loader, use_prob_threshold=use_prob_threshold)
        airbubble_pred, y_true, airbubble_prob = infer_cnn_v4(airbubble_model, val_loader, use_prob_threshold=use_prob_threshold) # use_prob_threshold
    else:
        blur_pred, y_true, blur_prob = infer_vit_v4(blur_model, val_loader, use_prob_threshold=use_prob_threshold)
        blood_pred, y_true1, blood_prob = infer_vit_v4(blood_model, val_loader, use_prob_threshold=use_prob_threshold)
        damaged_pred, y_true2, damaged_prob = infer_vit_v4(damaged_model, val_loader, use_prob_threshold=use_prob_threshold)
        fold_pred, y_true, fold_prob = infer_vit_v4(fold_model, val_loader, use_prob_threshold=use_prob_threshold)
        airbubble_pred, y_true, airbubble_prob = infer_vit_v4(airbubble_model, val_loader, use_prob_threshold=use_prob_threshold)

    assert y_true == y_true1, "The GroundTruths are not similar"
    assert y_true == y_true2, "The GroundTruths are not similar"

    blur_prob,  blood_prob, damaged_prob, fold_prob, airbubble_prob = np.round(blur_prob, decimals=5), np.round(blood_prob, decimals=5),\
     np.round(damaged_prob, decimals=5), np.round(fold_prob, decimals=5), np.round(airbubble_prob, decimals=5)

    blur_pred_b = np.array(blur_pred).astype(bool)
    blood_pred_b = np.array(blood_pred).astype(bool)
    damaged_pred_b = np.array(damaged_pred).astype(bool)
    fold_pred_b = np.array(fold_pred).astype(bool)
    airbubble_pred_b = np.array(airbubble_pred).astype(bool)

    artifact_list = [blur_pred_b[i] | blood_pred_b[i] | damaged_pred_b[i] | fold_pred_b[i] | airbubble_pred_b[i]
                         for i in range(len(blur_pred))]
    artifact_list = [a.astype(int) for a in artifact_list]

    file_names = [im[0].split("/")[-1] for im in val_loader.dataset.imgs]

    data = {"files": file_names, "ground_truth": y_true, "predicted": artifact_list, "blur": blur_pred, "blood": blood_pred,
            "damage": damaged_pred, "fold": fold_pred, "bubble": airbubble_pred,  "blur_p": blur_prob, "blood_p": blood_prob,
            "damage_p": damaged_prob, "fold_p": fold_prob, "bubble_p": airbubble_prob}

    dframe = pd.DataFrame(data)
    print("Length of dataframe ", len(dframe))

    dframe.insert(2, 'predicted_class', dframe.apply(assign_class, axis=1))

    dframe['binary_label'] = dframe.apply(make_binary_label, axis=1)

    dframe['max_prob'] =  dframe.apply(best_prob, axis=1)

    dframe['truth_label'] = dframe['binary_label'].apply(lambda x: 1 if x == 0 else 0)


    dframe['truth_prob'] =  dframe.apply(truth_prob_ensemb, axis=1)


    with pd.ExcelWriter(f"{sav_dir}/{ensemble}_predictions_ensemble.xlsx") as wr:
            dframe.to_excel(wr, index=False)

    labels = ['Artifact_free', 'Blood', 'Blur', 'Bubble', 'Damage', 'Fold']

    y_true = y_true
    y_pred = dframe['predicted_class'].tolist()
    cm = make_cm(y_true, y_pred, classes_list)

    make_pretty_cm_v3(cm, categories=labels, title=f"{ensemble}_ensemble")

    plt.savefig(f"{sav_dir}/{ensemble}_CM_ensemble.png")

    f1_mirco = f1_score(y_true, y_pred, average='micro')
    print("\nMicro F1 Score: ", np.round(f1_mirco, decimals=5))

    f1_macro = f1_score(y_true, y_pred, average='weighted')
    print("\nWeighted F1 Score: ", np.round(f1_macro, decimals=5))

    micro_acc = accuracy_score(y_true, y_pred)
    print("\nMicro Accuracy: ", np.round(micro_acc, decimals=5))

    macro_acc = accuracy_score(y_true, y_pred, normalize=True)
    print("\nMacro Accuracy: ", np.round(macro_acc, decimals=5))

    mcc = matthews_corrcoef(y_true, y_pred)
    print("\nMCC: ", np.round(mcc, decimals=5))

    class_index = 0  # Index of the artifact_free class
    tp = cm[class_index, class_index]
    fn = np.sum(cm[class_index, :]) - tp
    fp = np.sum(cm[:, class_index]) - tp
    tn = np.sum(cm) - np.sum(cm[class_index, :]) - np.sum(cm[:, class_index]) + tp

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("\nAccuracy of artifact-free class: ", np.round(accuracy, decimals=5))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score_af = 2 * (precision * recall) / (precision + recall)
    print("\nF1-Score of artifact-free class: ", np.round(f1_score_af, decimals=5))

    sens = tp/ (tp + fn)
    print("\nSensitivity of artifact-free class: ", np.round(sens, decimals=5))

    spec = tn/ (tn + fp)
    print("\nSpecificity of artifact-free class: ", np.round(spec, decimals=5))


    plt.clf()
    plot_roc_curve_v4(dframe, sensitivity_val = sens_thresh, title=f"{ensemble}_ROC_ensemble")
    plt.savefig(f"{d_path}/{ensemble}_ROC_ensemble.png")

print("\n## Finished ##")