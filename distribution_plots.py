""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file creates histograms from prediction to better understand ROC plots.
# Update paths to processed datasets

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils import plot_histogram


# Give path to excel sheets after training/inference scripts.
fname = "vits_predictions_multiclass_validation.xlsx"
path = f"/nfs/student/neel/preprocessing_models/{fname}"
df = pd.read_excel(path, engine='openpyxl')

plot_histogram(df, fname)
