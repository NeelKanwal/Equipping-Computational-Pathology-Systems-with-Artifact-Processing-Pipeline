
""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides inference code for multiclass DCNN and ViT models mentioned in the paper.
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
    import os
    import torch

    load_model = "vits" # "vits", "cnns"
    BATCH_SIZE = 128
    evaluate_with_prob = 0.015 # Use this probablity for thresholding, set to None for not using this feature

    # cuda_device = 3
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # sns.set_style("white")
    # torch.cuda.set_device(cuda_device)
    # torch.cuda.empty_cache()
    # print("Current CUDA device = ", torch.cuda.current_device())

    torch.cuda.empty_cache()
    cuda_device = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)


    from torch.utils.data import DataLoader
    import time
    import pprint
    from datetime import datetime
    import json
    import torch, torchvision
    from torchvision import datasets, models
    import torchvision.transforms as transforms

    from torch import nn
    from utils import get_class_distribution, make_cm, make_pretty_cm, convert_batch_list, infer_multiclass_v3, load_cnn_model, load_vit_model
    from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, f1_score, accuracy_score
    from scikitplot.metrics import plot_roc, plot_precision_recall 
    import timm

    from utils import best_prob, make_binary_label,  make_pretty_cm_v3, plot_roc_curve_v4

    torch.manual_seed(250)
    sens_thresh = 0.98 # for plot_roc curve to show probablity that gives this.

    path_to_dataset = "path_to/multiclass_artifact_data" # Use processed datasets from zenodo link in the repository
    models_location = "path_to/single_pipeline/model_weights/" # Use models from model_weights in repository
    sav_dir = "path_to/preprocessing_models/"

    
    # model weights names
    multiclass_cnn = "multiclass_cnn.dat"
    multiclass_vit = "multiclass_vit.dat"

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
        print("\nLoading Multiclass MobileNetv3\n")
        multiclass_model = load_cnn_model(models_location, multiclass_cnn, num_classes=6)
 
    else:
        print("\nLoading Multiclass ViT\n")
        multiclass_model = load_vit_model(models_location, multiclass_vit, num_classes=6)

    if torch.cuda.is_available():
        print("Cuda is available\n")
        # model should be on cuda before selection of optimizer
        multiclass_model = multiclass_model.cuda()

    print("--------------Validation Set-------------------------")

    if evaluate_with_prob is not None:
        print("Using thresholding @ ", evaluate_with_prob)

   
    y_true, y_pred, afree_pred, blood_pred, blur_pred, airbubble_pred, \
            damaged_pred, fold_pred, prob = infer_multiclass_v3(multiclass_model, val_loader)
  
    blur_pred_b = np.array(blur_pred).astype(bool)
    blood_pred_b = np.array(blood_pred).astype(bool)
    damaged_pred_b = np.array(damaged_pred).astype(bool)
    fold_pred_b = np.array(fold_pred).astype(bool)
    airbubble_pred_b = np.array(airbubble_pred).astype(bool)


    artifact_list = [blur_pred_b[i] | blood_pred_b[i] | damaged_pred_b[i] | fold_pred_b[i] | airbubble_pred_b[i]
                         for i in range(len(blur_pred))]
    artifact_list = [a.astype(int) for a in artifact_list]       

    file_names = [im[0].split("/")[-1] for im in val_loader.dataset.imgs]

    data = {"files": file_names, "ground_truth": y_true, "predicted_class": y_pred, "predicted_artifact": artifact_list, "afree":afree_pred, 
            "blood": blood_pred,  "blur": blur_pred,  "bubble": airbubble_pred, "damage": damaged_pred, "fold": fold_pred, "probs": prob}

    dframe = pd.DataFrame(data)
    print("Length of dataframe ", len(dframe))
   
    dframe[['afree_prob', 'blood_p', 'blur_p', 'bubble_p', 'damage_p', 'fold_p']] = pd.DataFrame(dframe['probs'].tolist(), index=dframe.index)


    dframe['actual_label'] = dframe.apply(make_binary_label, axis=1)

    dframe['max_prob'] =  dframe.apply(best_prob, axis=1)

    dframe['truth_label'] = dframe['actual_label'].apply(lambda x: 1 if x == 0 else 0)

    dframe['truth_prob'] = np.round(dframe['afree_prob'], decimals=4)

    with pd.ExcelWriter(f"{sav_dir}/{load_model}_predictions_multiclass_validation.xlsx") as wr:
            dframe.to_excel(wr, index=False)

    labels = ['Artifact_free', 'Blood', 'Blur', 'Bubble', 'Damage', 'Fold']

    y_true = y_true
    y_pred = dframe['predicted_class'].tolist()
    cm = make_cm(y_true, y_pred, classes_list)

    # make_pretty_cm(cm, categories=labels, cmap="tab20b", figsize=(14,14), title=f"{load_model}_ensemble")
    make_pretty_cm_v3(cm, categories=labels, title=f"{load_model}_multiclass_validation")
    plt.savefig(f"{sav_dir}/{load_model}_CM_multiclass_validation.png")

    f1_mirco = f1_score(y_true, y_pred, average='micro')
    print("\nMicro F1 Score: ", np.round(f1_mirco, decimals=4))

    f1_macro = f1_score(y_true, y_pred, average='weighted')
    print("\nWeighted F1 Score: ", np.round(f1_macro, decimals=4))

    micro_acc = accuracy_score(y_true, y_pred)
    print("\nMicro Accuracy: ", np.round(micro_acc, decimals=4))

    macro_acc = accuracy_score(y_true, y_pred, normalize=True)
    print("\nMacro Accuracy: ", np.round(macro_acc, decimals=4))

    mcc = matthews_corrcoef(y_true, y_pred)
    print("\nMCC: ", np.round(mcc, decimals=4))

    class_index = 0  # Index of the artifact_free class
    tp = cm[class_index, class_index]
    fn = np.sum(cm[class_index, :]) - tp
    fp = np.sum(cm[:, class_index]) - tp
    tn = np.sum(cm) - np.sum(cm[class_index, :]) - np.sum(cm[:, class_index]) + tp
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("\nAccuracy of artifact-free class: ", np.round(accuracy, decimals=4))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score_af = 2 * (precision * recall) / (precision + recall)
    print("\nF1-Score of artifact-free class: ", np.round(f1_score_af, decimals=4))

    sens = tp/ (tp + fn)
    print("\nSensitivity of artifact-free class: ", np.round(sens, decimals=4))

    spec = tn/ (tn + fp)
    print("\nSpecificity of artifact-free class: ", np.round(spec, decimals=4))


    plt.clf()
    plot_roc_curve_v4(dframe, sensitivity_val = sens_thresh, title=f"{load_model}_ROC_multiclass_validation")
    plt.savefig(f"{sav_dir}/{load_model}_ROC_multiclass_validation.png")

    print("--------------Test Set-------------------------")

    y_true, y_pred, afree_pred, blood_pred, blur_pred, airbubble_pred, \
            damaged_pred, fold_pred, prob = infer_multiclass_v3(multiclass_model, test_loader, use_prob_threshold=evaluate_with_prob)
             
    blur_pred_b = np.array(blur_pred).astype(bool)
    blood_pred_b = np.array(blood_pred).astype(bool)
    damaged_pred_b = np.array(damaged_pred).astype(bool)
    fold_pred_b = np.array(fold_pred).astype(bool)
    airbubble_pred_b = np.array(airbubble_pred).astype(bool)

    artifact_list = [blur_pred_b[i] | blood_pred_b[i] | damaged_pred_b[i] | fold_pred_b[i] | airbubble_pred_b[i]
                         for i in range(len(blur_pred))]
    artifact_list = [a.astype(int) for a in artifact_list]       

    file_names = [im[0].split("/")[-1] for im in test_loader.dataset.imgs]

    data = {"files": file_names, "ground_truth": y_true, "predicted_class": y_pred, "predicted_artifact": artifact_list, "afree":afree_pred, 
            "blood": blood_pred,  "blur": blur_pred,  "bubble": airbubble_pred, "damage": damaged_pred, "fold": fold_pred, "probs": prob}


    dframe = pd.DataFrame(data)
    print("Length of dataframe ", len(dframe))

    dframe[['afree_prob', 'blood_p', 'blur_p', 'bubble_p', 'damage_p', 'fold_p']] = pd.DataFrame(dframe['probs'].tolist(), index=dframe.index)

    dframe['actual_label'] = dframe.apply(make_binary_label, axis=1)

    dframe['max_prob'] =  dframe.apply(best_prob, axis=1)

    dframe['truth_label'] = dframe['actual_label'].apply(lambda x: 1 if x == 0 else 0)

    dframe['truth_prob'] = dframe['afree_prob']


    with pd.ExcelWriter(f"{sav_dir}/{load_model}_predictions_multiclass_test.xlsx") as wr:
            dframe.to_excel(wr, index=False)

    labels = ['Artifact_free', 'Blood', 'Blur', 'Bubble', 'Damage', 'Fold']

    y_true = y_true
    y_pred = dframe['predicted_class'].tolist()
    cm = make_cm(y_true, y_pred, classes_list)
    # make_pretty_cm(cm, categories=labels, cmap="tab20b", figsize=(14,14), title=f"{load_model}_ensemble")
    make_pretty_cm_v3(cm, categories=labels, title=f"{load_model}_multiclass_test")
    plt.savefig(f"{sav_dir}/{load_model}_CM_multiclass_test.png")


    plt.savefig(f"{sav_dir}/{load_model}_CM_multiclass_test.png")

    f1_mirco = f1_score(y_true, y_pred, average='micro')
    print("\nMicro F1 Score: ", np.round(f1_mirco, decimals=4))

    f1_macro = f1_score(y_true, y_pred, average='weighted')
    print("\nWeighted F1 Score: ", np.round(f1_macro, decimals=4))

    micro_acc = accuracy_score(y_true, y_pred)
    print("\nMicro Accuracy: ", np.round(micro_acc, decimals=4))

    macro_acc = accuracy_score(y_true, y_pred, normalize=True)
    print("\nMacro Accuracy: ", np.round(macro_acc, decimals=4))

    mcc = matthews_corrcoef(y_true, y_pred)
    print("\nMCC: ", np.round(mcc, decimals=4))

    class_index = 0  # Index of the artifact_free class
    tp = cm[class_index, class_index]
    fn = np.sum(cm[class_index, :]) - tp
    fp = np.sum(cm[:, class_index]) - tp
    tn = np.sum(cm) - np.sum(cm[class_index, :]) - np.sum(cm[:, class_index]) + tp

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("\nAccuracy of artifact-free class: ", np.round(accuracy, decimals=4))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("\nPrecision of artifact-free class: ", np.round(f1_score_af, decimals=4))

    print("\nRecall of artifact-free class: ", np.round(f1_score_af, decimals=4))
    f1_score_af = 2 * (precision * recall) / (precision + recall)
    print("\nF1-Score of artifact-free class: ", np.round(f1_score_af, decimals=4))

    sens = tp/ (tp + fn)
    print("\nSensitivity of artifact-free class: ", np.round(sens, decimals=4))

    spec = tn/ (tn + fp)
    print("\nSpecificity of artifact-free class: ", np.round(spec, decimals=4))

    plt.clf()
    plot_roc_curve_v4(dframe, sensitivity_val = sens_thresh, title=f"{load_model}_ROC_multiclass_test")
    plt.savefig(f"{sav_dir}/{load_model}_ROC_multiclass_test.png")

print("\n## Finished ##")