
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

# Scores assigned by pathologists for three tasks (Artifact Detection, Artifact-free Preservation, Overall Usability)

## P1 MoE DCNNs
pathologist_1 = {
    "Artifact Detection": [8, 9, 5, 4, 4, 3],
    "Artifact-free Preservation": [10, 10, 4, 6, 6,3],
    "Overall Usability": [9, 10, 5, 5, 5, 3]}

# P2 MoE DCNNs
pathologist_2 = {
    "Artifact Detection": [9, 9, 8, 6, 5, 6],
    "Artifact-free Preservation": [8, 7, 3, 4, 4, 4],
    "Overall Usability": [9, 8, 6, 5, 4, 4]}


# P3 MoE DCNNs
pathologist_3 = {
    "Artifact Detection": [9, 9, 9, 8, 3, 3],
    "Artifact-free Preservation": [9, 9, 9, 8, 3, 3],
    "Overall Usability": [10, 10, 10, 9, 4, 3]}

# # P1 MoE ViTs
# pathologist_1 = {
#     "Artifact Detection": [9, 9, 5, 4, 8, 5],
#     "Artifact-free Preservation": [9, 10, 4, 7, 9, 5],
#     "Overall Usability": [10, 8, 5, 5, 9, 4]}

# # P2 MoE ViTs
# pathologist_2 = {
#     "Artifact Detection": [10, 10, 6, 6, 7, 6],
#     "Artifact-free Preservation": [9, 7, 3, 3, 3, 3],
#     "Overall Usability": [10, 9, 6, 5, 5, 4]}

# # P3 MoE ViTs
# pathologist_3 = {
#     "Artifact Detection": [7, 7, 8, 7, 4, 3],
#     "Artifact-free Preservation": [7, 7, 8, 7, 5, 3],
#     "Overall Usability": [6, 6, 8, 7, 4, 3]}


data = {
    "Pathologist 1": {task: pathologist_1[task] for task in pathologist_1},
    "Pathologist 2": {task: pathologist_2[task] for task in pathologist_2},
    "Pathologist 3": {task: pathologist_3[task] for task in pathologist_3},
}



columns = ['WSI', 'Task', 'Pathologist', 'Scores']
df_list = []

colors = {'Artifact Detection': 'red',
          'Artifact-free Preservation': 'green',
          'Overall Usability': 'blue'}

for pathologist, ratings in data.items():
    for task, scores in ratings.items():
        for i, score in enumerate(scores):
            df_list.append(pd.DataFrame({'WSI': f's{i+1}', 'Task': task, 'Pathologist': pathologist, 'Scores': score}, index=[0]))

df = pd.concat(df_list, ignore_index=True)

# Create box plots
fig = px.box(df, x="WSI", y="Scores", color="Task", title="Ratings by Pathologists for ViT-based MoE", color_discrete_map=colors)

# Set figure size
fig.update_layout(width=600, height=600)

# Set legend as one row
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1))

# Set xticks
fig.update_xaxes(tickvals=df['WSI'].unique(), ticktext=df['WSI'].unique())
# fig.update_xaxes(showgrid=True, row=1, col=1)
# Save figure as high-resolution PNG
fig.write_image("boxplot_dcnns.png", scale=5)