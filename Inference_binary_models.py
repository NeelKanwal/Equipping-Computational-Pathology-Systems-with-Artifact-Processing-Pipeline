
""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides inference code for binary DCNN and ViT models mentioned in the paper.
# Update paths to processed datasets

if __name__ == "__main__":

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    import matplotlib.pyplot as plt


    import matplotlib
    font = {'family' : 'serif',
        'weight':'normal',
        'size'   : 36}
    matplotlib.rc('font', **font)
    plt.rcParams["figure.figsize"] = (20, 20)

    import pandas as pd
    import numpy as np
    import seaborn as sns
    sns.set_style("white")
    import os
    import torch

    load_model = "cnns" # "vits", "cnns"
    BATCH_SIZE = 128
    evaluate_with_prob = 0.4  # Use this probablity for thresholding, set to None for not using this feature
    ## Threshold for CNN = 0.001
    ## Threshold for ViT = 0.001

    # cuda_device = 3
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # torch.cuda.set_device(cuda_device)
    # torch.cuda.empty_cache()
    # print("Current CUDA device = ", torch.cuda.current_device())

    from torch.utils.data import DataLoader
    import time
    import pprint
    from datetime import datetime
    import json
    import torch, torchvision
    from torchvision import datasets, models
    import torchvision.transforms as transforms

    from torch import nn
    from utils import  infer_binary_v3, load_cnn_model, load_vit_model, make_pretty_cm_v3, plot_roc_curve_v4
    from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, f1_score, accuracy_score
    from scikitplot.metrics import plot_roc, plot_precision_recall 
    import timm

    torch.manual_seed(250)
    sens_thresh = 0.98 # for plot_roc curve to show probablity that gives this.

    path_to_dataset = "path_to/binary_artifact_data" # Use processed datasets from zenodo link in the repository
    models_location = "path_to/single_pipeline/model_weights/" # Use models from model_weights in repository
    sav_dir = "path_to/preprocessing_models/"

    # model weights names
    binary_cnn = "cnn_binary.dat"
    binary_vit = "vit_binary.dat"

    test_compose = val_compose = transforms.Compose([transforms.Resize((224, 224)),
        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    t = time.time()
    val_images = datasets.ImageFolder(root=path_to_dataset + "/test", transform=val_compose)
    idx2class = {v: k for k, v in val_images.class_to_idx.items()}
    classes_list = list(idx2class.values())
    print("ID to classes ", idx2class)
    num_classes = len(val_images.classes)
    val_loader = DataLoader(val_images, batch_size=BATCH_SIZE, shuffle=False, num_workers=16,pin_memory=True)

    test_images = datasets.ImageFolder(root=path_to_dataset + "/validation", transform=test_compose)
    test_loader = DataLoader(test_images, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    print(f"Total data loading time in minutes: {(time.time() - t) / 60:.3f}")

   
        # blur
    if load_model =="cnns":  
        print("\nLoading Binary MobileNetv3\n")
        multiclass_model = load_cnn_model(models_location, binary_cnn, num_classes=2)
 
    else:
        print("\nLoading Binary ViT\n")
        multiclass_model = load_vit_model(models_location, binary_vit, num_classes=2)

    if torch.cuda.is_available():
        print("Cuda is available\n")
        # model should be on cuda before selection of optimizer
        multiclass_model = multiclass_model.cuda()


    print("--------------Validation Set-------------------------")

    if evaluate_with_prob is not None:
        print("Using thresholding @ ", evaluate_with_prob)
    y_true, y_pred, prob = infer_binary_v3(multiclass_model, val_loader)
  
    file_names = [im[0].split("/")[-1] for im in val_loader.dataset.imgs]

    data = {"files": file_names, "ground_truth": y_true,  "afree_prob": prob}

    dframe = pd.DataFrame(data)

    print("Length of dataframe ", len(dframe))
   

    dframe['truth_label'] = dframe['ground_truth'].apply(lambda x: 1 if x == 0 else 0)

    dframe['truth_prob'] = np.round(dframe['afree_prob'], decimals=5)

    with pd.ExcelWriter(f"{sav_dir}/{load_model}_predictions_binary_validation.xlsx") as wr:
            dframe.to_excel(wr, index=False)

    labels = ['Artifact_free', 'Artifact']


    cm = make_cm(y_true, y_pred, classes_list)
    print(cm)

    make_pretty_cm_v3(cm, categories=labels, title=f"{load_model}_binary_validation")
    plt.savefig(f"{sav_dir}/{load_model}_CM_binary_validation.png")

    f1_mirco = f1_score(y_true, y_pred, average='micro')
    print("\nMicro F1 Score: ", np.round(f1_mirco, decimals=4))

    micro_acc = accuracy_score(y_true, y_pred)
    print("\nMicro Accuracy: ", np.round(micro_acc, decimals=5))

    macro_acc = accuracy_score(y_true, y_pred, normalize=True)
    print("\nMacro Accuracy: ", np.round(macro_acc, decimals=4))

    mcc = matthews_corrcoef(y_true, y_pred)
    print("\nMCC: ", np.round(mcc, decimals=4))

    tn, fp, fn, tp = cm.ravel()    
    recall = tp / (tp + fn)  # TPR
    print("\nSensitivity: ", np.round(recall, decimals=5))

    spec = tn/ (tn + fp)
    print("\nSpecificity of artifact-free class: ", np.round(spec, decimals=5))


    plt.clf()
    plot_roc_curve_v4(dframe, sensitivity_val = sens_thresh, title=f"{load_model}_ROC_binary_validation")
    plt.savefig(f"{sav_dir}/{load_model}_ROC_binary_validation.png")

    print("--------------Test Set-------------------------")

    y_true, y_pred, prob = infer_binary_v3(multiclass_model, test_loader, use_prob_threshold = evaluate_with_prob)

    file_names = [im[0].split("/")[-1] for im in test_loader.dataset.imgs]

    data = {"files": file_names, "ground_truth": y_true,  "afree_prob": prob}

    dframe = pd.DataFrame(data)

    print("Length of dataframe ", len(dframe))
   

    dframe['truth_label'] = dframe['ground_truth'].apply(lambda x: 1 if x == 0 else 0)

    dframe['truth_prob'] = np.round(dframe['afree_prob'], decimals=5)

    with pd.ExcelWriter(f"{sav_dir}/{load_model}_predictions_binary_test.xlsx") as wr:
            dframe.to_excel(wr, index=False)

    labels = ['Artifact_free', 'Artifact']

    cm = make_cm(y_true, y_pred, classes_list)
    print(cm)

    make_pretty_cm_v3(cm, categories=labels, title=f"{load_model}_binary_test")
    plt.savefig(f"{sav_dir}/{load_model}_CM_binary_test.png")

    f1_mirco = f1_score(y_true, y_pred, average='micro')
    print("\nMicro F1 Score: ", np.round(f1_mirco, decimals=4))

    micro_acc = accuracy_score(y_true, y_pred)
    print("\nMicro Accuracy: ", np.round(micro_acc, decimals=5))

    macro_acc = accuracy_score(y_true, y_pred, normalize=True)
    print("\nMacro Accuracy: ", np.round(macro_acc, decimals=4))

    mcc = matthews_corrcoef(y_true, y_pred)
    print("\nMCC: ", np.round(mcc, decimals=4))

    tn, fp, fn, tp = cm.ravel()    
    recall = tp / (tp + fn)  # TPR
    print("\nSensitivity: ", np.round(recall, decimals=5))

    spec = tn/ (tn + fp)
    print("\nSpecificity of artifact-free class: ", np.round(spec, decimals=5))

    plt.clf()
    plot_roc_curve_v4(dframe, sensitivity_val = sens_thresh, title=f"{load_model}_ROC_binary_test")
    plt.savefig(f"{sav_dir}/{load_model}_ROC_binary_test.png")


print("\n## Finished ##")