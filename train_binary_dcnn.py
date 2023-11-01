""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides training code for binary DCNN models mentioned in the paper.
# Update paths to processed datasets

if __name__ == "__main__":

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    import matplotlib.pyplot as plt

    import matplotlib
    import matplotlib
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import os
    font = {'family' : 'serif',
        'weight':'normal',
        'size'   : 28}
    matplotlib.rc('font', **font)
    plt.rcParams["figure.figsize"] = (20, 20)
    sns.set_style("white")
   
    #Select GPU to run
    # torch.cuda.empty_cache()
    # cuda_device = 3
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    import time
    import pprint
    from datetime import datetime
    import json
    import torchvision
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, models
    import torchvision.transforms as transforms
    from torch.optim import SGD, Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch import nn
    from utils import get_class_distribution, make_cm, make_pretty_cm_v3, infer_binary_v3, plot_roc_curve_v4
    from utils import train_cnn,  val_cnn, extract_features, custom_classifier, epoch_test_cnn, EarlyStopping_v2
    from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, brier_score_loss, \
        accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score, precision_score
    from scikitplot.metrics import plot_roc, plot_precision_recall, plot_lift_curve, plot_ks_statistic, \
        plot_calibration_curve
    import copy
    from sklearn.manifold import TSNE

    ## Experiment selection
    iterations = 3
    artifact = "binary" # [ "blood","damage", "blur", "fold", "airbubble", "multiclass", "binary"]
    pretrained = True  # train from scratch if False
    architectures = ["MobileNet"] # [VGG16", "EfficientNet", "DenseNet", "DenseNetConfig"]["ResNet"# ["ResNet"]#[

    freeze = False  # True for using ImageNet weights, False for retraining entire architecture.
    data_augmentation = True  # True to apply transformations to training data.
    
    BATCH_SIZE = 128
    n_epochs = 200
    patience = 20
    learning_rate = [0.01]  # [0.1, 0.01, 0.001]
    NUM_WORKER = 32  # Number of simultaneous compute tasks == number of physical cores
    stop_flag_count = 0
    opt = ["SGD"]  # [SGD", "Adam"]
    lr_scheduler = ["ReduceLROnPlateau"]  # ["ReduceLROnPlateau", "ExponentialLR"]
    dropout = 0.2
    torch.manual_seed(250)
    sens_thresh = 0.96 # use threshold for plot_roc_vurve_v4 to find a probablity to threshold later.

    if artifact == "damage":
        path_to_dataset = "path_to/artifact_dataset/damage"
    elif artifact == "blood":
        path_to_dataset = "path_to/artifact_dataset/blood"
    elif artifact == "airbubble":
        path_to_dataset = "path_to/artifact_dataset/bubbles"
    elif artifact == "blur":
        path_to_dataset = "path_to/artifact_dataset/blur"
    elif artifact == "fold":
        path_to_dataset = "path_to/artifact_dataset/fold"
    elif artifact == "binary":
        path_to_dataset = "path_to/binary_artifact_data"
    else:
        print("Artifact dataset not available")
        raise AssertionError

    if data_augmentation:
        train_compose = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    else:
        train_compose = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_compose = val_compose = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    t = time.time()

    
    print(f"\nLoading {str(artifact)} Artifact Dataset...................")
    train_images = datasets.ImageFolder(root=path_to_dataset + "/training", transform=train_compose)
    idx2class = {v: k for k, v in train_images.class_to_idx.items()}
    classes_list = list(idx2class.values())
    print("ID to classes ", idx2class)
    classes = train_images.classes
    class_distribution = get_class_distribution(train_images)
    print("Class distribution in training: ", class_distribution)
    # Get the class weights. Class weights are the reciprocal of the number of items per class, to obtain corresponding weight for each target sample.
    target_list = torch.tensor(train_images.targets)
    class_count = [i for i in class_distribution.values()]

    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    class_weights_all = class_weights[target_list]
    train_sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all),
                                          replacement=True)
    train_loader = DataLoader(train_images, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKER,
                              pin_memory=True)
    print(f"Length of training {len(train_images)} with {len(classes_list)} classes")
   

    # val_images = datasets.ImageFolder(root=path_to_dataset + "/validation", transform=val_compose)
    val_images = datasets.ImageFolder(root=path_to_dataset + "/test", transform=val_compose)
    idx2class = {v: k for k, v in val_images.class_to_idx.items()}
    num_classes = len(val_images.classes)
    val_loader = DataLoader(val_images, batch_size=BATCH_SIZE, shuffle=True, sampler=None, num_workers=NUM_WORKER,
                            pin_memory=True)
    print(f"Length of validation {len(val_images)} with {num_classes} classes")

   
    test_images = datasets.ImageFolder(root=path_to_dataset + "/validation", transform=test_compose)
    idx2class = {v: k for k, v in test_images.class_to_idx.items()}
    num_classes_ts = len(test_images.classes)
    test_loader = DataLoader(test_images, batch_size=BATCH_SIZE, shuffle=False, sampler=None,
                             num_workers=NUM_WORKER, pin_memory=True)
    print(f"Length of test {len(test_images)} with {num_classes_ts} classes")
    print(f"Total data loading time in minutes: {(time.time() - t) / 60:.3f}")

    
    for architecture in architectures:
        print("\n#############################################################################################################################")
        print(f"Artifact: {artifact}   Model: {architecture}  Data Augmentation:{data_augmentation} ")
        print("#############################################################################################################################\n")
        # Initialize model
        for op in opt:
            for sch in lr_scheduler:
                for lr in learning_rate:
                    for i in range(iterations):
                        print(f"\n//////////////  Iteration {i}  /////////////////\n")    
                        print("#########################################################")
                        print(f"Optimizer: {op}   Scheduler: {sch}  Learning rate: {lr} ")
                        print("#########################################################\n")

                        loss_tr, loss_val, acc_tr, acc_val = [], [], [], []
                        t = time.time()

                        if architecture   == "DenseNet":
                            print("Initializing DenseNet161 Model...............")
                            model = models.densenet161(weights=pretrained)  # growth_rate = 48, num_init_features= 96, config = (6,12,36,24)
                            if freeze:
                                for param in model.parameters():
                                    param.requires_grad = False
                            num_features = model.classifier.in_features  # 2208 --> less than 256
                            # model.classifier = nn.Linear(num_features, num_classes)
                            model.classifier = custom_classifier(num_features, num_classes, dropout=dropout)
                            print("Number of out features for patch is ", num_features)
                            pytorch_total_params = sum(p.numel() for p in model.parameters())
                            print("Total model parameters: ", pytorch_total_params)
                            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            print("Trainable parameters: ", trainable_params)

                        elif architecture == "GoogleNet":
                            print("Initializing GoogleNet Model...............")
                            if pretrained:
                                model = models.googlenet(pretrained=pretrained)
                            else:
                                model = models.googlenet(pretrained=pretrained, init_weights=True)
                                model.aux_logits = False
                            if freeze:
                                for param in model.parameters():
                                    param.requires_grad = False
                            num_features = model.fc.in_features
                            model.fc = custom_classifier(num_features, num_classes, dropout=dropout)
                            # model.fc = nn.Linear(num_features, num_classes)
                            print("Number of input features for patch is ", num_features)
                            pytorch_total_params = sum(p.numel() for p in model.parameters())
                            print("Total model parameters: ", pytorch_total_params)
                            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            print("Trainable parameters: ", trainable_params)

                        elif architecture == "ResNet":
                            # print("Initializing ResNet152 Model...............")
                            # model = models.resnet152(pretrained=pretrained)
                            print("Initializing ResNet18 Model...............")
                            model = models.resnet18(pretrained=pretrained)
                            if freeze:
                                for param in model.parameters():
                                    param.requires_grad = False
                            num_features = model.fc.in_features
                            model.fc = custom_classifier(num_features, num_classes, dropout=dropout)
                            # model.fc = nn.Linear(num_features, num_classes)
                            print("Number of input features for patch is ", num_features)
                            pytorch_total_params = sum(p.numel() for p in model.parameters())
                            print("Total model parameters: ", pytorch_total_params)
                            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            print("Trainable parameters: ", trainable_params)

                        elif architecture == "MobileNet":
                            print("Initializing MobileNet Model...............")
                            model = models.mobilenet_v3_large(pretrained=pretrained)
                            if freeze:
                                for param in model.parameters():
                                    param.requires_grad = False
                            model.classifier = custom_classifier(960, num_classes, dropout=dropout)
                            # model.classifier[-1] = nn.Linear(1280, num_classes)
                            print("Number of input features for patch is ", 960)
                            pytorch_total_params = sum(p.numel() for p in model.parameters())
                            print("Total model parameters: ", pytorch_total_params)
                            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            print("Trainable parameters: ", trainable_params)

                        elif architecture == "EfficientNet":
                            print("Initializing EfficientNetv2_s Model...............")
                            # model = models.efficientnet_v2_s(weights=torchvision.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT)
                            model = models.efficientnet_v2_s(weights=None)
                            if freeze:
                                for param in model.parameters():
                                    param.requires_grad = False
                            # num_features = model.classifier.in_features
                            model.classifier = custom_classifier(1280, num_classes, dropout=dropout)
                            # model.classifier[-1] = nn.Linear(1280, num_classes)
                            print("Number of input features for patch is ", 1280)
                            pytorch_total_params = sum(p.numel() for p in model.parameters())
                            print("Total model parameters: ", pytorch_total_params)
                            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            print("Trainable parameters: ", trainable_params)

                        elif architecture == "VGG16":
                            print("Initializing VGG16 Model...............")
                            model = models.vgg16(pretrained=pretrained)
                            if freeze:
                                for param in model.parameters():
                                    param.requires_grad = False
                            num_features = model.classifier[0].in_features
                            # model.classifier[6] = nn.Linear(num_features, num_classes)
                            model.classifier[6] = custom_classifier(4096, num_classes, dropout=dropout)
                            print("Number of input features for patch is ", num_features)
                            pytorch_total_params = sum(p.numel() for p in model.parameters())
                            print("Total model parameters: ", pytorch_total_params)
                            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            print("Trainable parameters: ", trainable_params)

                        else:
                            print("\nModel Does not exist")
                            raise AssertionError

                        print("Loss function is CrossEntropy")
                        criterion = nn.CrossEntropyLoss()

                        if torch.cuda.is_available():
                            print("Cuda is available")  # model should be on uda before selection of optimizer
                            model = model.cuda()

                        
                        if op == "SGD":
                            optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
                        elif op == "Adam":
                            optimizer = Adam(model.parameters(), lr=lr, betas=(0., 0.9), eps=1e-6, weight_decay=0.01)
                        else:
                            print("Optimizer does not exists in settings.\n")
                            raise AssertionError

                
                        if sch == "ReduceLROnPlateau":
                            # Reduce learning rate when a metric has stopped improving.
                            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
                        elif sch == "ExponentialLR":
                            # Decays the learning rate of each parameter group by gamma every epoch.
                            scheduler = ExponentialLR(optimizer, gamma=0.8)
                        else:
                            print("Scheduler does not exists in settings.\n")
                            raise AssertionError

                        print("\nTraining Starts....................")
                        now = datetime.now()
                        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
                        print(f"\nFiles for will be saved with {date_time} timestamp.")

                        if not os.path.exists(os.path.join(os.getcwd(), "experiments", str(architecture), date_time)):
                            if not os.path.exists(os.path.join(os.getcwd(), "experiments", str(architecture))):
                                os.mkdir(os.path.join(os.getcwd(), "experiments", str(architecture)))
                            path = os.path.join(os.getcwd(), "experiments", str(architecture), date_time)
                            os.mkdir(path)
                            print(f"\nDirectory Created {path}.")

                        param_dict = {"BATCH_SIZE": BATCH_SIZE,
                                      "EPOCHS": n_epochs,
                                      "PATIENCE": patience,
                                      "Learning Rate": lr,
                                      "Optimizer": op,
                                      "LR Scheduler": sch,
                                      "Artifact": artifact,
                                      "Model": architecture,
                                      "Weight Freezing": freeze,
                                      "Data Augmentation": data_augmentation,
                                      "Pretrained": pretrained}

                        # pprint.pprint(param_dict)
                        with open(f"{path}/Parameters.json", "a+") as f:
                            json.dump(param_dict, f, indent=4)
                        early_stopping = EarlyStopping_v2(patience=patience, verbose=False, timestamp=date_time, path=path)

                        # training loop
                        epoch_finished = 0
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_acc = 0.0
                        # to test model before running first epoch
                        tr_acc, tr_loss = epoch_test_cnn(model, train_loader, criterion)
                        val_acc, val_loss = epoch_test_cnn(model, val_loader, criterion)

                        print("\nEpoch 0")
                        print("\nValidation accuracy : {0:.3f} %\n".format(val_acc))
                        loss_val.append(val_loss)
                        loss_tr.append(tr_loss)
                        acc_val.append(val_acc)
                        acc_tr.append(tr_acc)

                        for epoch in range(1, n_epochs + 1):
                            tr_acc, tr_loss = train_cnn(model, criterion, optimizer, train_loader, epoch)
                            val_acc, val_loss, stop = val_cnn(model, early_stopping, date_time, val_loader, epoch, path, criterion)
                            loss_val.append(val_loss)
                            loss_tr.append(tr_loss)
                            acc_val.append(val_acc)
                            acc_tr.append(tr_acc)
                            epoch_finished += 1

                            if val_acc > best_acc:
                                best_acc = val_acc
                                best_model_wts = copy.deepcopy(model.state_dict())
                            if stop:
                                print(f"Early stopping at epoch {epoch}...\n")
                                break
                            if sch == "ReduceLROnPlateau":
                                scheduler.step(val_loss)
                            else:
                                scheduler.step()

                        print(f"(run {i}) training time for epochs in minutes: ",
                              (time.time() - t) / 60)
                        print(f"(run {i}) Best accuracy for {str(architecture)} is {best_acc:.3f} % .")
                        torch.save({'model': best_model_wts}, f"{path}/best_weights.dat")

                        plt.clf()
                        plt.figure(1)
                        plt.plot(loss_tr, "goldenrod",linewidth=3, label="Training loss")
                        plt.plot(loss_val, "slategray",linewidth=3, label="Validation loss")
                        plt.title(f"{str(architecture)} Loss Curve")
                        plt.legend(loc="best")
                        plt.savefig(f"{path}/Loss Curve.png")
                        # https://rstudio-conf-2020.github.io/dl-keras-tf/notebooks/learning-curve-diagnostics.nb.html

                        plt.clf()
                        plt.figure(2)
                        plt.plot(acc_tr, "indianred",linewidth=3, label="Training accuracy")
                        plt.plot(acc_val, "goldenrod",linewidth=3, label="Validation accuracy")
                        plt.title(f"{str(architecture)} Accuracy Curve")
                        plt.legend(loc="best")
                        plt.savefig(f"{path}/Accuracy Curve.png")
                        plt.clf()

                        with open(f"{path}/Experimental Values.txt", "a+", encoding='utf-8') as f:
                            acc_list_tr = [a.tolist() for a in acc_tr]
                            acc_list_val = [a.tolist() for a in acc_val]
                            dict = {"training_loss": loss_tr, "validation_loss": loss_val, "training_accuracy": acc_list_tr, \
                                    "validation_accuracy": acc_list_val}
                            f.write(str(dict))

                        # loading best model weights to find metrices
                        print(f"\nBest model weights with accuracy {best_acc:.3f} % loaded to compute metrices.....\n")
                        model.load_state_dict(best_model_wts)

                        print("######################################################")
                        print("--------------Validation Set-------------------------")

   
                        y_true, y_pred, prob = infer_binary_v3(model, val_loader)
                      
                        file_names = [im[0].split("/")[-1] for im in val_loader.dataset.imgs]

                        data = {"files": file_names, "ground_truth": y_true, "predicted_class": y_pred, "afree_prob": prob}

                        dframe = pd.DataFrame(data)

                        print("Length of dataframe ", len(dframe))


                        dframe['truth_label'] = dframe['ground_truth'].apply(lambda x: 1 if x == 0 else 0) # Flip artifact-free as one

                        # dframe['truth_prob'] = dframe['afree_prob']
                        dframe['truth_prob'] = dframe['afree_prob'].apply(lambda x: x[0])

                        with pd.ExcelWriter(f"{path}/CNN_predictions_{artifact}_validation.xlsx") as wr:
                                dframe.to_excel(wr, index=False)

                        labels = ['Artifact_free', 'Artifacts']

                        y_true = y_true
                        y_pred = y_pred
                        cm = make_cm(y_true, y_pred, classes_list)
                        print(cm)

                        # make_pretty_cm(cm, categories=labels, cmap="tab20b", figsize=(14,14), title=f"{load_model}_ensemble")
                        make_pretty_cm_v3(cm, categories=labels, title=f"CNN_{artifact}_validation")
                        plt.savefig(f"{path}/CNN_CM_multiclass_validation.png")

                        micro_acc = accuracy_score(y_true, y_pred)
                        print("\nMicro Accuracy= ", np.round(micro_acc, decimals=4))

                        f1_mirco = f1_score(y_true, y_pred, average='micro')
                        print("\nMicro F1 Score= ", np.round(f1_mirco, decimals=4))

                        tn, fp, fn, tp = cm.ravel()    
               

                        f1_macro = f1_score(y_true, y_pred, average='weighted')
                        print("\nWeighted F1 Score= ", np.round(f1_macro, decimals=4))

                        macro_acc = accuracy_score(y_true, y_pred, normalize=True)
                        print("\nMacro Accuracy= ", np.round(macro_acc, decimals=4))

                        mcc = matthews_corrcoef(y_true, y_pred)
                        print("\nMCC= ", np.round(mcc, decimals=4))

                        recall = tp / (tp + fn)  # TPR
                        print("\nSensitivity= ", np.round(recall, decimals=4))

                        spec = tn/ (tn + fp)
                        print("\nSpecificity= ", np.round(spec, decimals=4))



                        plt.clf()
                        # plot_roc_curve_v3(dframe, title=f"CNN_ROC_{artifact}_validation")
                        plot_roc_curve_v4(dframe, sensitivity_val = sens_thresh, title=f"cnns_ROC_binary_validation")
                        plt.savefig(f"{path}/CNN_ROC_{artifact}_validation.png")

                        print("--------------Test Set-------------------------")

                        y_true, y_pred, prob = infer_binary_v3(model, test_loader)
                      
                        file_names = [im[0].split("/")[-1] for im in test_loader.dataset.imgs]

                        data = {"files": file_names, "ground_truth": y_true, "predicted_class": y_pred, "afree_prob": prob}

                        dframe = pd.DataFrame(data)

                        print("Length of dataframe ", len(dframe))

                        dframe['truth_label'] = dframe['ground_truth'].apply(lambda x: 1 if x == 0 else 0) # Flip artifact-free as one

                        # dframe['truth_prob'] = dframe['afree_prob']
                        dframe['truth_prob'] = dframe['afree_prob'].apply(lambda x: x[0])

                        with pd.ExcelWriter(f"{path}/CNN_predictions_{artifact}_test.xlsx") as wr:
                                dframe.to_excel(wr, index=False)

                        labels = ['Artifact_free', 'Artifacts']

                        y_true = y_true
                        y_pred = y_pred
                        cm = make_cm(y_true, y_pred, classes_list)
                        print(cm)

                        make_pretty_cm_v3(cm, categories=labels, title=f"cnns_{artifact}_test")
                        plt.savefig(f"{path}/CNN_CM_{artifact}_test.png")

                        macro_acc = accuracy_score(y_true, y_pred, normalize=True)
                        print("\nMacro Accuracy= ", np.round(macro_acc, decimals=4))

                        f1_mirco = f1_score(y_true, y_pred, average='micro')
                        print("\nMicro F1 Score= ", np.round(f1_mirco, decimals=4))

                        tn, fp, fn, tp = cm.ravel()    
                        recall = tp / (tp + fn)  # TPR
                        print("\nSensitivity= ", np.round(recall, decimals=4))

                        f1_macro = f1_score(y_true, y_pred, average='weighted')
                        print("\nWeighted F1 Score= ", np.round(f1_macro, decimals=4))

                        micro_acc = accuracy_score(y_true, y_pred)
                        print("\nMicro Accuracy= ", np.round(micro_acc, decimals=4))

                        recall = tp / (tp + fn)  # TPR
                        print("\nSensitivity= ", np.round(recall, decimals=4))

                        spec = tn/ (tn + fp)
                        print("\nSpecificity= ", np.round(spec, decimals=4))


                        

                        mcc = matthews_corrcoef(y_true, y_pred)
                        print("\nMCC: ", np.round(mcc, decimals=4))

                        plt.clf()
                        # plot_roc_curve_v3(dframe, title=f"CNN_ROC_{artifact}_test")
                        plot_roc_curve_v4(dframe, sensitivity_val = sens_thresh, title=f"cnns_ROC_binary_test")
                        plt.savefig(f"{path}/CNN_ROC_{artifact}_test.png")

                        plt.close('all')
                       

    print("--------------------------------------------")
    print(f"Program finished for {architecture}.......")
    print("--------------------------------------------")


