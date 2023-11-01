# This script create masks from annotations.

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore all FutureWarning warnings that might flood the console log
warnings.simplefilter(action='ignore', category=DeprecationWarning) # Ignore all DeprecationWarning warnings that might flood the console log

import os
import json
# Use this while working on windows as it has issues with libvips
os.environ["PATH"] = "E:\\Histology\\WSIs\\openslide-win64-20171122\\bin" + ";" + os.environ["PATH"]
os.environ["PATH"] = "E:\\Histology\\WSIs\\vips-dev-8.11\\bin" + ";" + os.environ["PATH"]

import pyvips as vips
import openslide
print("Pyips: ",vips.__version__)
print("Openslide: ",openslide.__version__)
from PIL import Image
from histolab.slide import Slide
import matplotlib.pyplot as plt
from skimage.draw import polygon
import numpy as np
import pickle

directory = "path_ti_annotations_and_wsis" # "train/"  , "validation/"  #os.getcwd()
t_files = os.listdir(directory)
total_wsi = [f for f in t_files if f.endswith("scn")]
total_xml = [f for f in t_files if f.endswith("xml")]
print("Total WSI {} and total annotations {}".format(len(total_wsi), len(total_xml)))

# turtle = py_wsi.Turtle(file_dir, db_location, db_name, xml_dir=xml_dir, label_map=label_map, storage_type='disk')

# # print("Total WSI images in test:    " + str(turtle.num_files))
# print("LMDB name:           " + str(turtle.db_name))
# print("File names:          " + str(turtle.files))
# print("XML files found:     " + str(turtle.get_xml_files()))
#
# current = turtle.files[0]
# print("Starting patching process for ", current)
# level_count, level_tiles, level_dims = turtle.retrieve_tile_dimensions(current, patch_size=patch_size)
# print("Level count:         " + str(level_count))
# print("Level tiles:         " + str(level_tiles))
# print("Level dimensions:    " + str(level_dims))
#
# turtle.sample_and_store_patches(patch_size, level, overlap, load_xml=True, limit_bounds=True)
#
#
# import py_wsi.dataset as ds
#
# dataset = ds.read_datasets(turtle,
#                            set_id=1,
#                            valid_id=0,
#                            total_sets=2,
#                            shuffle_all=True,
#                            augment=True)
#
# print("Total training set patches:     " + str(len(dataset.train.images)))
# print("Total validation set patches:   " + str(len(dataset.valid.images)))
#
# tk.show_labeled_patches(dataset.train.images, dataset.train.image_cls)
# assert len(total_wsi)==len(total_xml)
# print("Annotaion missing")
# for c_file in total_wsi:
#     current_img_400x = vips.Image.new_from_file(os.path.join(directory,c_file), level=0)
#     height , width = current_img_400x.height, current_img_400x.width
#     print("The Original shape of file {} is {} * {} but will reduce by 100 to save the mask.".format(c_file,height,width))
for ann in total_xml:
# for ann in ["CZ604TP.xml"]:
    with open(os.path.join(directory, ann), "r") as f:
        annotation = json.loads(f.read())
        fname = annotation['filename'].split('/')[2][:-4] + ".scn"
        # current_img_400x = vips.Image.new_from_file(os.path.join(directory,fname), level=0)
        # w, h = current_img_400x.width, current_img_400x.height
        slide = openslide.OpenSlide(os.path.join(directory,fname))
        (w,h) = slide.dimensions
        curr_slide = Slide(os.path.join(directory, fname), os.path.join(directory, fname))
        # current_slide.level_magnification_factor(level=1)
        # slide = openslide(os.path.join(directory,fname))
        # image = slide.read_region((0,0),1,slide.dimensions)
        # resized_image = image.resize(size=shape)
        # resized_image = resized_image.convert('RGB')
        # width, height = curr_slide.level_dimensions(level=0)
        # width, height = curr_slide.dimensions
        print("The original shape of file {} is {} * {} but will reduce by 100 to save the mask.".format(fname, w, h))
        # thumbnail = curr_slide.scaled_image(10)
        # thumbnail = vips.Image.thumbnail(os.path.join(directory,fname), width=int(w/100), height=int(h/100), linear=bool)
        # thumbnail = current_img_400x.thumbnail_image(int(w/100), height=int(h/100))
        thumbnail = slide.get_thumbnail((w/100, h/100))
        plt.imshow(thumbnail)
        plt.axis('off')
        plt.title(None)
        plt.savefig(os.path.join(directory, "%s_thumbnail.png"%fname), bbox_inches='tight',pad_inches = 0, dpi=100)
        # plt.show()

        reducing_factor = 100
        shape = (int((w/reducing_factor)), int((h/reducing_factor)))
        # current_slide.__getattribute__("levels")
        # shape = curr_slide._thumbnail_size
        # thumbnail = current_slide.thumbnail # 200 for new slides scanned at 80x, 100 to older one scanned on 40x


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


        # # my_functions.pickle_save(masks, '%s.obj'%fname)
        # with open(os.path.join(directory,'%s.obj'%fname), 'wb') as handle:
        #     pickle.dump(masks,handle,protocol= pickle.HIGHEST_PROTOCOL)


        for mask in masks:
            # if mask== list(masks.keys())[0]:
            plt.axis("off")
            plt.title(None)
            # plt.title("%s_%s"%(fname,mask))
            plt.imshow(Image.fromarray(masks[mask].T) , cmap="gray")    #
            # plt.savefig(os.path.join(directory,"%s_%s.png"%(fname,mask)),bbox_inches='tight',pad_inches = 0)
            # plt.show() # to see the mask in pycharm.
            curr_mask = masks[mask]
            mask_img = Image.fromarray(masks[mask].T)
            # new_img = mask_img.resize(shape, resample=Image.BILINEAR)
            mask_img.save(os.path.join(directory,"%s_%s.png"%(fname,mask)))
        print("WSI {} contains {} labels".format(fname,list(masks.keys())))
