""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides training code for SOTA DCNNs models mentioned in the paper.
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
        'size'   : 28}
    matplotlib.rc('font', **font)

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import os

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
    from torch.optim import SGD, Adadelta, Rprop, RMSprop, Adam, Adagrad, ASGD
    from torch.optim.lr_scheduler import MultiStepLR, LinearLR, ReduceLROnPlateau, ExponentialLR
    from torch import nn
    from utils import get_class_distribution, make_cm,  make_pretty_cm, convert_batch_list
    from utils import train_cnn, infer_cnn_v2, val_cnn,  custom_classifier, EarlyStopping_v2, epoch_test_cnn

    from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, brier_score_loss, \
        accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score, precision_score
    from scikitplot.metrics import plot_roc, plot_precision_recall, plot_lift_curve, plot_ks_statistic, \
        plot_calibration_curve
    from scikitplot.helpers import binary_ks_curve
    import copy
    from sklearn.manifold import TSNE

    ## Experiment selection
    iterations = 5
    artifact = "multiclass"  # [ "blood", "damage",  "blur", "fold", "airbubble", "multiclass"]
    pretrained = True  # train from scratch if False
    architectures = ["MobileNet"] # [VGG16", "EfficientNet", "DenseNet","ResNet"]

    freeze = False  # True for using ImageNet weights, False for retraining entire architecture.
    data_augmentation = True  # True to apply transformations to training data.
    test = True
    
    BATCH_SIZE = 128
    n_epochs = 200
    patience = 20
    learning_rate = [0.01]  # [0.1, 0.01, 0.001]
    NUM_WORKER = 32  # Number of simultaneous compute tasks == number of physical cores
    stop_flag_count = 0
    opt = ["SGD"]  # ["Adadelta", "SGD", "Adam", "Rprop", "RMSprop"]
    lr_scheduler = ["ReduceLROnPlateau"]  # ["ReduceLROnPlateau", "MultiStepLR", "LinearLR", "ExponentialLR"]
    dropout = 0.2
    torch.manual_seed(250)

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
    elif artifact == "multiclass":
        path_to_dataset = "path_to/multiclass_artifact_data"
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
    print("Class count in training ", class_count)

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

    if test:
        # test_images = datasets.ImageFolder(root=path_to_dataset + "/test", transform=test_compose)
        test_images = datasets.ImageFolder(root=path_to_dataset + "/validation", transform=test_compose)
        idx2class = {v: k for k, v in test_images.class_to_idx.items()}
        num_classes_ts = len(test_images.classes)
        test_loader = DataLoader(test_images, batch_size=BATCH_SIZE, shuffle=False, sampler=None,
                                 num_workers=NUM_WORKER, pin_memory=True)
        print(f"Length of test {len(test_images)} with {num_classes_ts} classes")
    print(f"Total data loading time in minutes: {(time.time() - t) / 60:.3f}")

    
    for architecture in architectures:
        print("\n#######################################################################################################")
        print(f"Artifact: {artifact}   Model: {architecture}  Data Augmentation:{data_augmentation} ")
        print("##########################################################################################################\n")
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

                        if op == "Adadelta":
                            optimizer = Adadelta(model.parameters(), lr=lr, rho=0.8)
                        elif op == "SGD":
                            optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
                        elif op == "ASGD":
                            optimizer = ASGD(model.parameters(), lr=lr)
                        elif op == "Adagrad":
                            optimizer = Adagrad(model.parameters(), lr=lr)
                        elif op == "Adam":
                            optimizer = Adam(model.parameters(), lr=lr, betas=(0., 0.9), eps=1e-6, weight_decay=0.01)
                        elif op == "Rprop":
                            optimizer = Rprop(model.parameters(), lr=lr)
                        elif op == "RMSprop":
                            optimizer = RMSprop(model.parameters(), lr=lr, alpha=0.9, weight_decay=0)
                        else:
                            print("Optimizer does not exists in settings.\n")
                            raise AssertionError

                        if sch == "MultiStepLR":
                            # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
                            scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
                        elif sch == "LinearLR":
                            # Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
                            scheduler = LinearLR(optimizer)
                        elif sch == "ReduceLROnPlateau":
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

                        param_dict = {
                                      "BATCH_SIZE": BATCH_SIZE,
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
}

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
                            val_acc, val_loss, stop = val_cnn(model, early_stopping, date_time, val_loader, epoch, path,
                                                              criterion)
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
                        plt.title(f"{str(architecture)} Loss Curve for {artifact} classification")
                        plt.legend(loc="best")
                        plt.savefig(f"{path}/Loss Curve for {str(artifact)}.png")
                        # https://rstudio-conf-2020.github.io/dl-keras-tf/notebooks/learning-curve-diagnostics.nb.html

                        plt.clf()
                        plt.figure(2)
                        plt.plot(acc_tr, "indianred",linewidth=3, label="Training accuracy")
                        plt.plot(acc_val, "goldenrod",linewidth=3, label="Validation accuracy")
                        plt.title(f"{str(architecture)} Accuracy Curve for {artifact} classification")
                        plt.legend(loc="best")
                        plt.savefig(f"{path}/Accuracy Curve for {str(artifact)}.png")
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

                        print("--------------------------------------------------------------")
                        print("--------------------------------------------------------------")
                        print(f"\nMETRICS for validation dataset {artifact} classification")
                        # y_pred, y_true, probs, feature = predict_cnn(val_loader, model)
                        y_pred, y_true, probs, feature, _, _, mean_1, lower_1c, upper_1c = infer_cnn_v2(val_loader, model)
                        y_pred = convert_batch_list(y_pred)
                        y_true = convert_batch_list(y_true)
                        probs = convert_batch_list(probs)
                        feature = convert_batch_list(feature)
                        mean_1 = convert_batch_list(mean_1)
                        lower_1c = convert_batch_list(lower_1c)
                        upper_1c = convert_batch_list(upper_1c)
                        print(classification_report(y_true, y_pred))


                        file_names = [im[0].split("/")[-1] for im in val_loader.dataset.imgs]
                        data = {"files": file_names, "ground_truth": y_true, "prediction": y_pred,
                                "probabilities": probs,"cnn_mean1": mean_1, "cnn_lower1": lower_1c,
                                "cnn_upper1": upper_1c}
                        dframe = pd.DataFrame(data)

                        with pd.ExcelWriter(f"{path}/predictions_for_validation_{str(artifact)}.xlsx") as wr:
                            dframe.to_excel(wr, index=False)


                        with open(f"{path}/Classification Report (val).txt", "a+") as f:
                            f.write(str(classification_report(y_true, y_pred)))

                        cm = make_cm(y_true, y_pred, classes_list)
                        print(cm)
        
                        accuracy = accuracy_score(y_true, y_pred)
                        print("Accuracy: ", accuracy)

                        # added here for multiclass
                        f1_mirco = f1_score(y_true, y_pred, average='micro')
                        print("Micro F1 Score: ", f1_mirco)

                        f1_macro = f1_score(y_true, y_pred, average='macro')
                        print("Macro F1 Score: ", f1_macro)

                        f1_weighted = f1_score(y_true, y_pred, average='weighted')
                        print("Weight F1 Score: ", f1_weighted)

                        # f1 = f1_score(y_true, y_pred)
                        # print("F1 Score: ", f1)
                        # # recall = tp / (tp + fn)  # TPR
                        # roc = roc_auc_score(y_true, y_pred)
                        # print("ROC AUC Score: ", roc)
                        mathew_corr = matthews_corrcoef(y_true, y_pred)
                        # It’s a correlation between predicted classes and ground truth
                        # When working on imbalanced problems
                        print("Mathew Correlation Coefficient: ", mathew_corr)
                        # precision = tp / (tp + fp)  # precision = hit rate = PPV
                        # # print("\nPositive Predictive Value/Precision: ",precision)  # how many observations predicted as positive are in fact positive

                        # # print("Sensitivity/Recall: ",recall)  # how many observations out of all positive observations have we classified as positive
                        # tnr = tn / (tn + fp)
                        # # print("Specificity: ", tnr)
                        # # print("\nFalse Positive Rate: ", fpr)  # fraction of false alerts
                        # fnr = fn / (tp + fn)
                        # # print("False Negative Rate: ", fnr)  # fraction of missed alerts

                        # # print("True Negative Rate: ",tnr)  # out of all negative observations have we classified as negative
                        # npv = tn / (tn + fn)
                        # # print("Negative Predictive Value: ",npv)  # how many predictions out of all negative predictions were correct
                        # fdr = fp / (tp + fp)
                        # print("False Discovery Rate: ",fdr)  # how many predictions out of all positive predictions were incorrect

                        # avg_precision = average_precision_score(y_true, y_pred)
                        # # Area Under the Precision-Recall Curve to get one number that describes model performance
                        # # you care more about positive than negative class
                        # # print("Average Precision: ", avg_precision)
                        # m_precision = precision_score(y_true, y_pred, average='micro')
                        # # print("Micro Precision: ", m_precision)

                        cohen_kappa = cohen_kappa_score(y_true, y_pred)
                        # how much better is your model over the random classifier that predicts based on class frequencies
                        # print("Cohen Kappa: ", cohen_kappa)

                        # measure of how far your predictions lie from the true values
                        # brier_loss = brier_score_loss(y_true, y_pred)
                        # # print("Brier Score: ", brier_loss)

                        # res = binary_ks_curve(y_true, y_pred)
                        # # get one number that we can use as a metric we can look at all thresholds (dataset cutoffs)
                        # ks_stat = res[3]
                        # # print("KS-Statistic: ", ks_stat)

                        # metrics = {"Accuracy": accuracy,
                        #            "F1-score": f1,
                        #            "ROC AUC": roc,
                        #            "Mathew Correlation": mathew_corr,
                        #            "FPR": fpr,
                        #            "TPR": recall,  # recall
                        #            "FNR": fnr,
                        #            "TNR": tnr,
                        #            "Specificity": tnr,
                        #            # Specificity (True Negative Rate) refers to the probability of a negative test
                        #            "NPV": npv,
                        #            "PPV": precision,  #
                        #            "FDR": fdr,
                        #            "Sensitivity": recall,
                        #            # Sensitivity (True Positive Rate) refers to the probability of a positive test
                        #            "Recall": recall,
                        #            "Precision": precision,
                        #            "Hit-rate": precision,
                        #            "Micro Precision": m_precision,
                        #            "Average Precision": avg_precision,
                        #            "Cohen Kappa": cohen_kappa,
                        #            "Brier Score": brier_loss,
                        #            "KS-Statistic": ks_stat}

                        # with open(f"{path}/Metrics (val).json", "a+") as f:
                        #     json.dump(metrics, f, indent=4)
                        plt.clf()
                        plt.figure(3)

                        # It is a curve that combines precision (PPV) and Recall (TPR) in a single visualization
                        # The higher on y-axis your curve is the better your model performance.
                        plot_precision_recall(y_true, probs, title=f'Precision-Recall Curve for {artifact}',
                                              title_fontsize='large')
                        plt.savefig(f"{path}/Precision-Recall Curve (val) for {artifact}.png")

                        plt.clf()
                        plt.figure(4)
                        #  visualizes the tradeoff between true positive rate (TPR) and false positive rate (FPR)
                        plot_roc(y_true, probs, classes_to_plot=1, title=f'ROC Curve for {artifact}',
                                 title_fontsize='large')
                        plt.savefig(f"{path}/ROC Curve for (val) {artifact}.png")


                        plt.clf()
                        plt.figure(5)
                        # group_names = ["TN", "FP", "FN", "TP"] group_names=group_names,
                        make_pretty_cm(cm, categories=classes_list, figsize=(12, 12), sum_stats=True,
                            cmap="tab20b", title=f"{architecture} Prediction for {str(artifact)}")
                        # cmap Accent, jet, brg, Blues, 'tab20b'
                        plt.savefig(f"{path}/Pretty Confusion Matrix (val) for {str(artifact)}.png")

                        # plt.clf()
                        # plt.figure(6)
                        # print(f"Making T-SNE for {artifact}")
                        # cnn_tsne = TSNE(n_components=2, perplexity=40.0, n_iter=400).fit_transform(feature)
                        # sns.set_style("white")
                        # sns.scatterplot(cnn_tsne[:, 0], cnn_tsne[:, 1], hue=y_true,palette=sns.color_palette("hls", num_classes),legend="full")
                        # L = plt.legend()
                        # # L.get_texts()[0].set_text('Artifact_free')
                        # # L.get_texts()[1].set_text(f'{artifact}')
                        # plt.title(f't-SNE from {architecture} CNN')
                        # plt.savefig(f"{path}/cnn_TSNE Plot (val) for {artifact}.png")

                        if test:
                            print("\n---------------------------------------------------------")
                            print(f"\nMETRICS for test dataset {artifact} classification")
                            # y_pred, y_true, probs, feature = predict_cnn(test_loader, model)
                            y_pred, y_true, probs, feature, _, _, mean_1, lower_1c, upper_1c = infer_cnn(
                                test_loader, model, samples=50)
                            y_pred = convert_batch_list(y_pred)
                            y_true = convert_batch_list(y_true)
                            probs = convert_batch_list(probs)
                            feature = convert_batch_list(feature)
                            mean_1 = convert_batch_list(mean_1)
                            lower_1c = convert_batch_list(lower_1c)
                            upper_1c = convert_batch_list(upper_1c)
                            print(classification_report(y_true, y_pred))

                            file_names = [im[0].split("/")[-1] for im in test_loader.dataset.imgs]
                            data = {"files": file_names, "ground_truth": y_true, "prediction": y_pred,
                                    "probabilities": probs, "cnn_mean1": mean_1,
                                    "cnn_lower1": lower_1c, "cnn_upper1": upper_1c}
                            dframe = pd.DataFrame(data)

                            with pd.ExcelWriter(f"{path}/predictions_for_test_{str(artifact)}.xlsx") as wr:
                                dframe.to_excel(wr, index=False)

                            with open(f"{path}/Classification Report (test).txt", "a+") as f:
                                f.write(str(classification_report(y_true, y_pred)))

                            cm = make_cm(y_true, y_pred, classes_list)
                            print(cm)
                            # tn, fp, fn, tp = cm.ravel()
                            # fpr = fp / (fp + tn)
                            accuracy = accuracy_score(y_true, y_pred)
                            print("Accuracy: ", accuracy)

                             # added here for multiclass
                            f1_mirco = f1_score(y_true, y_pred, average='micro')
                            print("Micro F1 Score: ", f1_mirco)

                            f1_macro = f1_score(y_true, y_pred, average='macro')
                            print("Macro F1 Score: ", f1_macro)

                            f1_weighted = f1_score(y_true, y_pred, average='weighted')
                            print("Weight F1 Score: ", f1_weighted)
                            # f1 = f1_score(y_true, y_pred)
                            # print("F1 Score: ", f1)
                            # roc = roc_auc_score(y_true, y_pred)
                            # print("ROC AUC Score: ", roc)
                            mathew_corr = matthews_corrcoef(y_true, y_pred)
                            # It’s a correlation between predicted classes and ground truth
                            # When working on imbalanced problems
                            print("Mathew Coefficient: ", mathew_corr)

                            # precision = tp / (tp + fp)  # precision = hit rate = PPV
                            # # print("\nPPV: ", precision)

                            # npv = tn / (tn + fn) 
                            # print("NPV: ", npv)  # how many predictions out of all negative predictions were correct

                            # brier_loss = brier_score_loss(y_true, y_pred)
                            # # print("Brier Score: ", brier_loss)

                            # print("\nFalse Positive Rate: ", fpr)  # fraction of false alerts, known as Type-I error
                            # fnr = fn / (tp + fn)
                            # # print("False Negative Rate: ", fnr)  # fraction of missed alerts, known as Type-II error
                            # tnr = tn / (tn + fp)
                            # # print("True Negative Rate: ",tnr)  # out of all negative observations have we classified as negative

                            # fdr = fp / (tp + fp)
                            # # print("False Discovery Rate: ",fdr)  # how many predictions out of all positive predictions were incorrect
                            # recall = tp / (tp + fn)  # also called True Positive Rate = sensitivity
                            # print("Sensitivity/Recall/TPR: ",recall)  # how many observations out of all positive observations have we classified as positive

                            # print("Precision/Hit Rate: ",precision)  # how many observations predicted as positive are in fact positive
                            # avg_precision = average_precision_score(y_true, y_pred)

                            # # you care more about positive than negative class
                            # # print("Average Precision: ", avg_precision)
                            # m_precision = precision_score(y_true, y_pred, average='micro')
                            # print("Micro Precision: ", m_precision)

                            cohen_kappa = cohen_kappa_score(y_true, y_pred)
                            # how much better is your model over the random classifier that predicts based on class frequencies
                            # print("Cohen Kappa: ", cohen_kappa)
                            #  should use ROC AUC when you care equally about positive and negative classes
                            # # measure of how far your predictions lie from the true values
                            # res = binary_ks_curve(y_true, y_pred)
                            # # get one number that we can use as a metric we can look at all thresholds (dataset cutoffs)
                            # ks_stat = res[3]
                            # print("KS-Statistic: ", ks_stat)

                            # metrics = {"Accuracy": accuracy,
                            #            "F1-score": f1,
                            #            "ROC AUC": roc,
                            #            "Mathew Correlation": mathew_corr,
                            #            "FPR": fpr,
                            #            "TPR": recall,  # recall
                            #            "FNR": fnr,
                            #            "TNR": tnr,
                            #            "Specificity": tnr,
                            #            # Specificity (True Negative Rate) refers to the probability of a negative test
                            #            "NPV": npv,
                            #            "PPV": precision,  #
                            #            "FDR": fdr,
                            #            "Sensitivity": recall,
                            #            # Sensitivity (True Positive Rate) refers to the probability of a positive test
                            #            "Recall": recall,
                            #            "Precision": precision,
                            #            "Hit-rate": precision,
                            #            "Micro Precision": m_precision,
                            #            "Average Precision": avg_precision,
                            #            "Cohen Kappa": cohen_kappa,
                            #            "Brier Score": brier_loss,
                            #            "KS-Statistic": ks_stat}

                            # with open(f"{path}/Metrics (test).json", "a+") as f:
                            #     json.dump(metrics, f, indent=4)
                            plt.clf()
                            plt.figure(7)
                            # It is a curve that combines precision (PPV) and Recall (TPR) in a single visualization
                            # The higher on y-axis your curve is the better your model performance.
                            # Area Under the Precision-Recall Curve to get one number that describes model performance
                            plot_precision_recall(y_true, probs, title=f'Precision-Recall Curve for {artifact}',
                                                  title_fontsize='large')
                            plt.savefig(f"{path}/Precision-Recall Curve (test) for {artifact}.png")

                            plt.clf()
                            plt.figure(8)
                            #  visualizes the tradeoff between true positive rate (TPR) and false positive rate (FPR)
                            plot_roc(y_true, probs, classes_to_plot=1, title=f'ROC Curve for {artifact}',
                                     title_fontsize='large')
                            plt.savefig(f"{path}/ROC Curve for (test) {artifact}.png")


                            plt.clf()
                            plt.figure(9)
                            # group_names = ["TN", "FP", "FN", "TP"] group_names=group_names,
                            make_pretty_cm(cm, categories=classes_list, figsize=(12, 12), sum_stats=True,
                                        cmap="tab20b", title=f"{architecture} Prediction for {str(artifact)}")
                            # cmap Accent, jet, brg, Blues, 'tab20b'
                            plt.savefig(f"{path}/Pretty Confusion Matrix (test) for {str(artifact)}.png")

                            # plt.clf()
                            # plt.figure(10)
                            # print(f"Making T-SNE for {artifact}")
                            # cnn_tsne = TSNE(n_components=2, perplexity=40.0, n_iter=400).fit_transform(feature)
                            # sns.set_style("white")
                            # sns.scatterplot(cnn_tsne[:, 0], cnn_tsne[:, 1], hue=y_true,palette=sns.color_palette("hls", num_classes), legend="full")
                            # L = plt.legend()
                            # # L.get_texts()[0].set_text('Artifact_free')
                            # # L.get_texts()[1].set_text(f'{artifact}')
                            # plt.title(f't-SNE from CNN ')
                            # plt.savefig(f"{path}/cnn_TSNE (test) Plot for {artifact}.png")
                            plt.close('all')
                        plt.close('all')
                

    print("--------------------------------------------")
    print(f"Program finished for {architecture}.......")
    print("--------------------------------------------")


