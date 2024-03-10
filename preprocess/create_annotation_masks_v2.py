import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore all FutureWarning warnings that might flood the console log
warnings.simplefilter(action='ignore', category=DeprecationWarning) # Ignore all DeprecationWarning warnings that might flood the console log

import os
import json
os.environ["PATH"] = "path_to\\openslide-win64-20171122\\bin" + ";" + os.environ["PATH"]
os.environ["PATH"] = "path_to\\vips-dev-8.11\\bin" + ";" + os.environ["PATH"]

import os # Useful when running on windows.
os.environ["PATH"] = "/path_to/openslide-win64-20171122/bin/" + ";" + os.environ["PATH"]
# os.environ["PATH"] = "path_to\vips-dev-8.11\\bin" + ";" + os.environ["PATH"]
os.environ["PATH"] = "/path_to/vips-dev-8.11/bin/" + ";" + os.environ["PATH"]


import pyvips as vips
import openslide
print("Pyips: ",vips.__version__)
print("Openslide: ",openslide.__version__)
from PIL import Image
# from histolab.slide import Slide
import matplotlib.pyplot as plt
from skimage.draw import polygon
import numpy as np
# import pickle


def read_vips(file_path, level=0):
    if file_path.endswith("mrxs"): # mrxs are scanned with
        #  flatten() to force RGBA to RGB, to set a white background
        # print("MRXS file, loading file at 40x")
        try:
            img_400x = vips.Image.new_from_file(file_path, level=level+1,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_path, page=level+1,
                                                autocrop=True).flatten()
    else:
        try:
            img_400x = vips.Image.new_from_file(file_path, level=level,
                                                autocrop=True).flatten()
        except:
            img_400x = vips.Image.new_from_file(file_path, page=level,
                                                autocrop=True).flatten()
    return img_400x

# directory = "D:\\mask_from_xml\\qunatitative_test" # "train/"  , "validation/"  #os.getcwd()
directory = "/path_to/full_artifact_pipeline/new_WSIs/"
sav_dir = "/path_to/full_artifact_pipeline/new_WSIs/masks/"

t_files = os.listdir(directory)
total_wsi = [f for f in t_files if f.endswith("mrxs")]
total_xml = [f for f in t_files if f.endswith("xml")]

for ann in ['s3.xml']:

    with open(os.path.join(directory, ann), "r") as f:
        reducing_factor = 50
        annotation = json.loads(f.read())

        fname = annotation['filename'].split('/')[2][:-4] + ".ndpi"

        slide = read_vips(os.path.join(directory,fname))
        w, h = slide.width, slide.height

        print("The original shape of file {} is {} * {} but will reduce by {} to save the mask.".format(fname, w, h, reducing_factor))
    
        # thumbnail = slide.get_thumbnail((w/100, h/100))
        thumbnail = slide.resize(1/100)
        # sav_fig(sav_path, thumbnail, sav_name="#thumbnail")
        plt.imshow(thumbnail)
        plt.axis('off')
        plt.title(None)
        plt.savefig(os.path.join(sav_dir, "%s_thumbnail.png"%fname), bbox_inches='tight',pad_inches = 0, dpi=100)

        
        shape = (int(w/reducing_factor), int(h/reducing_factor))
   
        regionsDict = annotation['Regions']
        masks = dict()
        for region in regionsDict:
            region_label = region['name']
            segments = region['path'][1]['segments']

            points = np.array(segments)
            points = np.transpose(points)

            imgp = np.full(shape, False)
            rr, cc = polygon(*points,shape=shape)
            imgp[rr, cc] = True

            if region_label not in masks:
                masks[region_label] = np.full(shape, False)
                masks[region_label] = masks[region_label] | imgp
            else:
                masks[region_label] = masks[region_label] | imgp

        for mask in masks:
            curr_mask = masks[mask].T
            plt.axis("off")
            plt.title(None)
            plt.imshow(Image.fromarray(curr_mask) , cmap="gray")    
            mask_img = Image.fromarray(curr_mask)
            
            mask_img.save(os.path.join(sav_dir,"%s_%s.png"%(fname, mask)))
        print("WSI {} contains {} labels".format(fname,list(masks.keys())))
