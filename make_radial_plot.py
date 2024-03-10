import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict
import matplotlib.colors as colors
import cv2
import random

def scatter_plot(dir_paths, num_files=500, alpha=0.6):
    fig, ax = plt.subplots()
    
    # Dictionary mapping index -> label
    labels = {i: f"s{i+1}" for i, _ in enumerate(dir_paths)}

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    markers = ['o', 's', 'D', 'v', '^', 'p', 'h']
    for i, (dir_path, label) in enumerate(zip(dir_paths, labels.values()), start=1):
        files = os.listdir(dir_path)[:num_files]
        hues = []
        saturations = []
        for file in files:
            img = cv2.imread(os.path.join(dir_path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue = img[:, :, 0].mean()
            saturation = img[:, :, 1].mean()
            hues.append(hue)
            saturations.append(saturation)
        
        color = colors[i - 1]
        marker = markers[(i - 1) % len(markers)]
        # ax.scatter(hues, saturations, c=color, marker=marker, alpha=alpha, label=f"{label}")
        ax.scatter(hues, saturations, c=color,  alpha=alpha, label=f"{label}", s=12)

    ax.set_ylabel('Saturation')
    ax.set_xlabel('Hue')
    ax.legend()
    plt.savefig("hue_vs_saturdation.pdf", format='pdf', dpi=600)
    plt.show()


# Replace these with the actual directory paths.
dir1 = ""
dir2 = "/"
dir3= "/nfstud/"
dir4= "/nfs/s/n/"
dir5= "/nfs/studches/"
dir6 = "/nfs/stus/"


listofdirs = [dir1, dir2, dir3, dir4, dir5, dir6]
# scatter_plot(listofdirs)


def random_grid_plot(dir_paths, num_images=3, row_cols=(6, 3)):
    fig, axes = plt.subplots(6, 3)
    fig.suptitle("Random Images HSV Values")

    for i, (dir_path, ax_group) in enumerate(zip(dir_paths, axes.flat)):
        files = sorted(os.listdir(dir_path))
        selected_indices = random.sample(range(len(files)), min(num_images, len(files)))
        hues = []
        saturations = []
        for idx in selected_indices:
            img = cv2.imread(os.path.join(dir_path, files[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue = img[:, :, 0].mean()
            saturation = img[:, :, 1].mean()
            hues.append(hue)
            saturations.append(saturation)
            
        for j, (hue_, sat_) in enumerate(zip(hues, saturations)):
            # ax = ax_group[j // row_cols[1]]
            ax.imshow(img, aspect="auto")
            ax.tick_params(left=False, bottom=False)
            ax.set_title(f"Hue:{int(hue_)}, Saturation:{int(sat_)}")
            ax.axis("off")

    plt.savefig("palletete.pdf", format='pdf', dpi=600)

    plt.show()

random_grid_plot(listofdirs)