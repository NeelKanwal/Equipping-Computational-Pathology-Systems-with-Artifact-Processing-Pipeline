from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

# Scores assigned by pathologists for three tasks (Artifact Detection, Artifact-free Preservation, Overall Usability)
## P1 MoE DCNNs
# pathologist_1 = {
#     "Artifact Detection": [8, 9, 5, 4, 4, 3],
#     "Artifact-free Preservation": [10, 10, 4, 6, 6,3],
#     "Overall Usability": [9, 10, 5, 5, 5, 3]}

## P2 MoE DCNNs
# pathologist_2 = {
#     "Artifact Detection": [9, 9, 8, 6, 5, 6],
#     "Artifact-free Preservation": [8, 7, 2, 4, 4, 4],
#     "Overall Usability": [9, 8, 6, 5, 4, 4]}


## P3 MoE DCNNs
# pathologist_3 = {
#     "Artifact Detection": [9, 9, 9, 8, 3, 3],
#     "Artifact-free Preservation": [9, 9, 9, 8, 3, 3],
#     "Overall Usability": [10, 10, 10, 9, 4, 3]}

# # P1 MoE ViTs
pathologist_1 = {
    "Artifact Detection": [9, 9, 5, 4, 8, 5],
    "Artifact-free Preservation": [9, 10, 4, 7, 9, 5],
    "Overall Usability": [10, 8, 5, 5, 9, 4]}

# # P2 MoE ViTs
pathologist_2 = {
    "Artifact Detection": [10, 10, 6, 6, 7, 6],
    "Artifact-free Preservation": [9, 7, 3, 3, 3, 3],
    "Overall Usability": [10, 9, 6, 5, 5, 4]}

# # P3 MoE ViTs
pathologist_3 = {
    "Artifact Detection": [7, 7, 8, 7, 4, 3],
    "Artifact-free Preservation": [7, 7, 8, 7, 5, 3],
    "Overall Usability": [6, 6, 8, 7, 4, 3]}



def calculate_weighted_kappa_score(pathologist1_scores, pathologist2_scores, weights = 'quadratic'):
    kappa = cohen_kappa_score(pathologist1_scores, pathologist2_scores, weights=weights)
    return kappa

def calculate_agreement(tasks, pathologist1, pathologist2):
    agreements = {}
    for task in tasks:
        kappa = calculate_weighted_kappa_score(pathologist1[task], pathologist2[task])
        agreements[task] = kappa
    return agreements

tasks = ["Artifact Detection", "Artifact-free Preservation", "Overall Usability"]

print("Agreement between Pathologist 1 and Pathologist 2:")
agreements_1_2 = calculate_agreement(tasks, pathologist_1, pathologist_2)
for task, kappa in agreements_1_2.items():
    print("Task: {}, Cohen's kappa: {:.2f}".format(task, kappa))

print("\nAgreement between Pathologist 1 and Pathologist 3:")
agreements_1_3 = calculate_agreement(tasks, pathologist_1, pathologist_3)
for task, kappa in agreements_1_3.items():
    print("Task: {}, Cohen's kappa: {:.2f}".format(task, kappa))

print("\nAgreement between Pathologist 2 and Pathologist 3:")
agreements_2_3 = calculate_agreement(tasks, pathologist_2, pathologist_3)
for task, kappa in agreements_2_3.items():
    print("Task: {}, Cohen's kappa: {:.2f}".format(task, kappa))



all_agreements = [agreements_1_2, agreements_1_3, agreements_2_3]
pairs = ["P1 - P2", "P1 - P3", "P2 - P3"]


# Create a DataFrame for plotting
plot_data = []
for pair, agreement in zip(pairs, all_agreements):
    for task, kappa in agreement.items():
        plot_data.append({"Pair": pair, "Task": task, "Agreement": kappa})

plot_df = pd.DataFrame(plot_data)

# Set seaborn style
sns.set(style="white")

# Set custom color palette
pair_palette = sns.color_palette("husl", len(pairs))
task_palette = sns.color_palette("husl", len(tasks))

# Create line plot
plt.figure(figsize=(12, 4))

# Draw lines for each pair
for i, pair in enumerate(pairs):
    pair_data = plot_df[plot_df["Pair"] == pair]
    y_vals = [i + 1] * len(pair_data)
    plt.scatter(pair_data["Agreement"], y_vals, color=task_palette, label=None, s=100, zorder=2)
    avg_agreement = np.mean(pair_data["Agreement"])
    plt.axvline(x=avg_agreement, linestyle='--', color=pair_palette[i], linewidth=2, zorder=2, label=None)

# Draw horizontal lines for each pair
for i, pair in enumerate(pairs):
    plt.axhline(y=i + 1, color=pair_palette[i], linewidth=2, zorder=1, label=pair)

# Set plot title and labels
plt.title("Agreement Between Pathologists for ViTs-based Mixture of Experts")
plt.xlabel("Cohen's Kappa")
# plt.ylabel("Pair of Pathologists")

# Set y ticks and labels
plt.yticks(range(1, len(pairs) + 1), pairs)
plt.xlim(0, 1.1)
plt.xticks(np.arange(0, 1.1, 0.2), rotation=0)
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))

# Add legend outside the plot
legend_labels = [f"{task}: " for task in tasks]
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=task_palette[i], markersize=10, label=legend_labels[i]) for i in range(len(tasks))]
# plt.legend(handles=legend_handles, labels=tasks, loc='center left', bbox_to_anchor=(1, 0.5), title='Tasks')
# plt.legend(handles=legend_handles, labels=tasks, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=len(tasks))

# Remove background grid
plt.grid(False)

# Show plot
plt.tight_layout()
plt.savefig("cohen_kappa_plot_moe_vits.png", dpi=300)
# plt.show()
