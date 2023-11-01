""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This is the main file provides MULTICLASS (CNN and ViT) models for end-to-end solution for what is mentioned in the paper.
# Update paths to processed datasets

if __name__ == '__main__':
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Ignore all FutureWarning warnings that might flood the console log
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import torch
    import numpy as np
    import time
    from torchvision import transforms
    from torch.utils.data import DataLoader
    # functions for preprocessing (foreground-background segementation, patching), running DL models and post-processing.
    from utils import create_binary_mask, create_patches, data_generator, load_vit_model, \
        infer_multiclass, post_process_masks, segmentation_color_mask,\
        calculate_quality, refine_artifacts_wsi, load_cnn_model, \
        extract_patches_coords, WSI_Patch_Dataset, create_foreground_mask_vips

    from mmcv.cnn import get_model_complexity_info
    # Alternate Libraries to
    # from flopth import flopth
    from numerize import numerize
    # from calc_flops import calc_flops
    # from fvcore.nn import FlopCountAnalysis

    font = {'family': 'serif',
            'weight': 'normal',
            'size': 24}
    plt.rc('font', **font)
    test_transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Loading directory
    wsi_dir ="path_to/wsi_data/"

    # Saving directory
    save_dir = wsi_dir

    models_location = "path_to/model_weights/"

    # CNN Models Weights =
    multiclass_vit = "multiclass_vit.dat"
    multiclass_cnn = "multiclass_cnn.dat"

    # postprocessing output masks
    segmentation_mask = True
    refined_wsi = True
    quality_report = True

    fig = plt.subplots(figsize=(12, 8))

    # Other params
    cal_throughput = True
    choose_model = "CNN" # "CNN", "ViT"

    cuda_gpu = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_gpu)
    torch.cuda.set_device(cuda_gpu)
    torch.cuda.empty_cache()

    downsize = 224
    patch_extraction_size = 224
    mask_overlap = 80.0
    batch_size = 128
    cpu_workers = 40
    use_prob_threshold = 0.95 # None  # whether to give final prediction {0,1} based on certain probability

    torch.manual_seed(250)

    # read the files
    wsi_files = os.listdir(wsi_dir)
    wsi_files = [f for f in wsi_files if f.endswith("scn") or f.endswith("mrxs")]
    # get all files except temp directory containing patches
    print(f"Total files in {wsi_dir} directory are {len(wsi_files)}")

    if choose_model == "CNN":
        d_path = os.path.join(wsi_dir, "cnn_multiclass")
        if not os.path.exists(d_path):
            os.mkdir(d_path)
    else:
        d_path = os.path.join(wsi_dir, "vit_multiclass")
        if not os.path.exists(d_path):
            os.mkdir(d_path)

    # start patching process
    for f in wsi_files:
    # for f in ["CZ464.ndpi"]:
        st = time.time()
        # find binary mask to locate tissue on WSI
        path = os.path.join(d_path, f.split(".")[0])
        # just take the name not extension
        if not os.path.exists(path):
            os.mkdir(path)
        # w, h = create_foreground_mask_vips(wsi_dir, f, save_path=path, downsize=downsize)
        w, h = create_binary_mask(wsi_dir, f, path, downsize=downsize)
        # print(f"Binary tissue mask created for {f}")
        # start splitting WSI into patches
        patch_folder = os.path.join(path, "patches")
        if not os.path.exists(patch_folder):
            os.mkdir(patch_folder)
            # assuming patches directory exists and patches are already created.
            # Old program that saves patches
            total_patches = create_patches(wsi_dir, f, path, patch_folder,workers=cpu_workers,
                                           patch_size=patch_extraction_size,
                                           mask_overlap=mask_overlap)

        data_loader, total_patches = data_generator(patch_folder,  test_transform=test_transform,
                                                       batch_size=batch_size, worker=cpu_workers)

        # total_patches = len(data_generator)
        if choose_model == "CNN":
            print("\nLoading multiclass CNN Model")
            model = load_cnn_model(models_location, multiclass_cnn, num_classes=6)

        else:
            print("\nLoading multiclass ViT Model")
            model = load_vit_model(models_location, multiclass_vit, num_classes=6)

        flops, params = get_model_complexity_info(model, ((3,224,224)),
                                                  as_strings=False, print_per_layer_stat=False)
        million_param = numerize.numerize(params)
        gflops = numerize.numerize(flops)
        print(f"\nTotal model Mparam {million_param} and GFlops {gflops} in the multiclass {choose_model}.")

        if torch.cuda.is_available():
            print("Cuda is available")
            # model should be on cuda before selection of optimizer
            model = model.cuda()
        print("\n########### Inference Starts ##############")
        st2 = time.time()

        y_pred, afree_pred, blood_pred, blur_pred, airbubble_pred, \
            damaged_pred, fold_pred, prob = infer_multiclass(model, data_loader, use_prob_threshold)

        seconds = time.time()-st2
        minutes = seconds/60
        print(f"Time consumed in inference for {f} in {minutes:.2f} minutes.\n")

        # Calculate throughtput
        if cal_throughput:
            print("Throughput: {:.2f}  patches/seconds".format(total_patches/seconds))

        file_names = [im.split("/")[-1] for im in data_loader.dataset.data_path]
        data = {"files": file_names, "predicted": y_pred, "probs": prob, "blur": blur_pred,
                "blood": blood_pred, "damage": damaged_pred, "fold": fold_pred, "airbubble": airbubble_pred}

        dframe = pd.DataFrame(data)

        with pd.ExcelWriter(f"{path}/multiclass_{choose_model}_predictions.xlsx") as wr:
            dframe.to_excel(wr, index=False)

        if use_prob_threshold is not None:
            print(f"Probablity threshold of {use_prob_threshold} used for determining overall prediction.\n")

        # minutes = (time.time()-st2)/60
        # print(f"Time consumed in inference for {f} in {minutes:.2f} minutes.\n")

        print("########### Postprocessing Starts ##########")
        print(f"Using probabiliy threshold {use_prob_threshold}.")
        # postprocess from dataframe
        st3 = time.time()
        post_process_masks(dframe, path, wsi_shape=(w, h), downsize=downsize)
        
        if segmentation_mask:
            st4 = time.time()
            segmentation_color_mask(path, scale=10)
            minutes3 = (time.time()-st4)/60
            print(f"Created color segmentation mask, time consumed {minutes3:.2f} minutes.")
        if refined_wsi:
            st5 = time.time()
            refine_artifacts_wsi(os.path.join(wsi_dir, f), path)
            minutes4 = (time.time()-st5)/60
            print(f"Refined {f} for artifacts, time consumed {minutes4:.2f} minutes.\n")
        if quality_report:
            st6 = time.time()
            # read artifact masks from path and save the json file with percentage of artifacts
            calculate_quality(path)
            minutes5 = (time.time()-st6)/60
            print(f"\nPrepared quality report for {f}, time consumed {minutes5:.2f} minutes.")

        minutes = (time.time()-st3)/60
        print(f"Time consumed in post-processing for {f} in {minutes:.2f} minutes.\n")

        minutes = (time.time()-st)/60
        print(f"Total for end-to-end processing {f} in {minutes:.2f} minutes.")
