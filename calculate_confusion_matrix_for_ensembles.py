""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file provides confusion matrix for ensemble of binary models for better comparisons.

import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from utils import make_confusion_matrix

font = {'family': 'serif',
        'weight': 'normal',
        'size': 28}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = (14, 14)

make_for = "cnn" # "cnn", "vit"
giva_a_name = "validation"

# To load all excel sheets produced from train_binary_dccn.py and train_binary_vit.py
base_location = "path/excel_sheets/"

# Individual paths to artifact models.
# You can also load the model_weights and run them over processed dataset to obtain these excel sheets.
blood_predictions = f"{make_for}_predictions_for_{giva_a_name}_blood.xlsx"
blur_predictions = f"{make_for}_predictions_for_{giva_a_name}_blur.xlsx"
damage_predictions = f"{make_for}_predictions_for_{giva_a_name}_damage.xlsx"
fold_predictions = f"{make_for}_predictions_for_{giva_a_name}_fold.xlsx"
bubble_predictions = f"{make_for}_predictions_for_{giva_a_name}_airbubble.xlsx"

df_blood = pd.read_excel(os.path.join(base_location, blood_predictions), engine='openpyxl')
df_blur = pd.read_excel(os.path.join(base_location, blur_predictions), engine='openpyxl')
df_damage = pd.read_excel(os.path.join(base_location, damage_predictions), engine='openpyxl')
df_fold = pd.read_excel(os.path.join(base_location, fold_predictions), engine='openpyxl')
df_bubble = pd.read_excel(os.path.join(base_location, bubble_predictions), engine='openpyxl')

# using artifact free predictions from since blur was the one with highest MCC on validation in ViTs and MobileNet

# Get artifact free labels from all dataframes
af_1 = df_blur[df_blur['ground_truth'] == 0][['files', 'ground_truth', 'prediction']]
af_2 = df_blood[df_blood['ground_truth'] == 0][['files', 'ground_truth', 'prediction']]
af_3 = df_bubble[df_bubble['ground_truth'] == 0][['files', 'ground_truth', 'prediction']]
af_4 = df_damage[df_damage['ground_truth'] == 0][['files', 'ground_truth', 'prediction']]
af_5 = df_fold[df_fold['ground_truth'] == 0][['files', 'ground_truth', 'prediction']]

# result = pd.merge(af_1, af_2, on='files', how='inner')
# result = pd.merge(result, af_3, on='files', how='inner')
# result = pd.merge(result, af_4, on='files', how='inner')
# af = pd.merge(result, af_5, on='files', how='inner')
af = pd.concat([af_1, af_2, af_3, af_4, af_5])
af = af.drop_duplicates(subset=['files'])
before = len(af_1) + len(af_2) + len(af_3) + len(af_4) + len(af_5)
print("Total length of artifact free images before merging  ", before, "   and after, ", len(af))
af['ground_truth'] = 0

blood = df_blood[df_blood['ground_truth'] == 1][['files', 'ground_truth', 'prediction']]
blood['ground_truth'] = 1
blur = df_blur[df_blur['ground_truth'] == 1][['files', 'ground_truth', 'prediction']]
blur['ground_truth'] = 2
blur.loc[blur['prediction'] == 1] = 2
bubble = df_bubble[df_bubble['ground_truth'] == 1][['files', 'ground_truth', 'prediction']]
bubble['ground_truth'] = 3
bubble.loc[bubble['prediction'] == 1] = 3
damage = df_damage[df_damage['ground_truth'] == 1][['files', 'ground_truth', 'prediction']]
damage['ground_truth'] = 4
damage.loc[damage['prediction'] == 1] = 4
fold = df_fold[df_fold['ground_truth'] == 1][['files', 'ground_truth', 'prediction']]
fold['ground_truth'] = 5
fold.loc[fold['prediction'] == 1] = 5

final = pd.concat([af, blood, blur, bubble, damage, fold])
before = len(final) 
final = final.drop_duplicates(subset=['files'])
print("Total length of artifact images before merging  ", before, "   and after, ", len(final))

y_true = list(final['ground_truth'])
y_pred = list(final['prediction'])

with pd.ExcelWriter(f"{base_location}/{make_for}_predictions_for_{giva_a_name}_ensemble.xlsx") as wr:
            final.to_excel(wr, index=False)

cm = confusion_matrix(y_true, y_pred)
labels = ['Artifact_free', 'Blood', 'Blur', 'Bubble', 'Damage', 'Fold']

make_confusion_matrix(cm, categories=labels, cmap="tab20b", figsize=(12, 13))
plt.savefig(f"{base_location}/{make_for}_Pretty_Confusion_Matrix_{giva_a_name}_ensemble.png")

print("Completed")

multiclass_predictions = f"{make_for}_predictions_for_{giva_a_name}_multiclass.xlsx"
df_multiclass = pd.read_excel(os.path.join(base_location, multiclass_predictions), engine='openpyxl')
before = len(df_multiclass)
df_multiclass = df_multiclass.drop_duplicates(subset=['files'])
print("Total length of artifact images before merging  ", before, "   and after, ", len(df_multiclass))



