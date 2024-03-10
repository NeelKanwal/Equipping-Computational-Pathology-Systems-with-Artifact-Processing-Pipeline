import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore all FutureWarning warnings that might flood the console log
warnings.simplefilter(action='ignore', category=DeprecationWarning) # Ignore all DeprecationWarning warnings that might flood the console log

import numpy as np
import skimage.metrics as sm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from skimage import metrics
import cv2

def iou(pred, gt):
    """Calculate Intersection over Union (IoU)"""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union

def dice_score(pred, gt):
    """Calculate Dice score"""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (2 * intersection) / union

def jaccard_index(pred, gt):
    """Calculate Jaccard Index"""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union - intersection + 1e-7)

def precision(pred, gt):
    """Calculate Precision"""
    true_positives = np.logical_and(pred, gt).sum()
    predicted_positives = pred.sum()
    return true_positives / predicted_positives


def recall(pred, gt):
    """Calculate Recall"""
    true_positives = np.logical_and(pred, gt).sum()
    true_negatives = np.logical_and(~pred, ~gt).sum()
    return true_positives / (true_positives + true_negatives)


def f1_score(pred, gt):
    """Calculate F1 score"""
    precision_score = precision(pred, gt)
    recall_score = recall(pred, gt)
    return 2 * (precision_score * recall_score) / (precision_score + recall_score)


def confusion_matrix_iou(pred, gt):
    """Calculate Confusion Matrix and IoU"""
    cm = confusion_matrix(gt.flatten(), pred.flatten())
    iou_score = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    return iou_score

fname  = "s6"

ensembles = ["cnn_ensemble", "vit_ensemble"]
# fnames = ["s1", "s2", "s3", "s4", "s5", "s6"]
fnames = ["s6"]

for ensem in ensembles:
    for fname in fnames :
    # Your predicted segmentation mask
        pred_mask_path = f"/path_to/Inference/{fname}/{ensem}/#original#merged#mask.png"

        # Your ground truth segmentation mask
        gt_mask_path = f"/path_to/full_artifact_pipeline/new_WSIs/Inference/{fname}/{fname}_merged.png" 


        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)


        height, width = gt_mask.shape
        pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        iou_score = iou(pred_mask, gt_mask)
        precision_score = precision(pred_mask, gt_mask)
        recall_score = recall(pred_mask, gt_mask)
        f1_score_value = f1_score(pred_mask, gt_mask)
        confusion_matrix_iou_score = confusion_matrix_iou(pred_mask, gt_mask)
        dice_score_value = dice_score(pred_mask, gt_mask)
        jaccard_index_value = jaccard_index(pred_mask, gt_mask)
        distance = metrics.hausdorff_distance(pred_mask, gt_mask)

        print(f"\n######## Results for {fname} with {ensem} #########\n")

        print(f"Jaccard Index: {jaccard_index_value:.3f}")
        print(f"Dice score: {dice_score_value:.3f}")
        print(f"\nIoU: {iou_score:.3f}")
        print(f"Hausdorff Distance: {distance:.3f}")
        print(f"Precision: {precision_score:.3f}")
        print(f"Recall: {recall_score:.3f}")
        print(f"F1 score: {f1_score_value:.3f}")
        print("IoU for positive and negative class: \n", confusion_matrix_iou_score)

print("### FINISHED ######\n")