""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This is the main file provides ENSEMBLE of binary CNN and ViT models for end-to-end solution for what is mentioned in the paper.
# Update paths to processed datasets



# All libraries
if __name__ == '__main__':
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
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
    from utils import create_binary_mask, create_patches, data_generator, load_cnn_model, load_vit_model\
        infer_cnn, infer_vit, post_process_masks, segmentation_color_mask, calculate_quality, \
        refine_artifacts_wsi

    from mmcv.cnn import get_model_complexity_info
    # Alternate Libraries to
    # from flopth import flopth
    from numerize import numerize
    # from calc_flops import calc_flops
    # from fvcore.nn import FlopCountAnalysis
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 28}
    plt.rc('font', **font)

    # plt.style.use('science')
    test_transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Loading directory
    # Update paths here
    wsi_dir = "path_to/wsi_data/"
    # Saving directory
    save_dir = wsi_dir
    models_location = "path_to/model_weights/"
    ensemble =="cnns" # "vits"

    # CNN Models Weights =
    blood_cnn = "blood_cnn.dat"
    blur_cnn = "blur_cnn.dat"
    fold_cnn = "fold_cnn.dat"
    damaged_cnn = "damage_cnn.dat"
    airbubble_cnn = "airbubble_cnn.dat"

    # ViT Models Weights =
    blood_vit = "blood_vit.dat"
    blur_vit = "blur_vit.dat"
    fold_vit = "fold_vit.dat"
    damaged_vit = "damage_vit.dat"
    airbubble_vit = "airbubble_vit.dat"


    # postprocessing output masks
    segmentation_mask = True
    refined_wsi = True
    quality_report = True
    evaluate_with_prob = 0.326 # Use this probablity for thresholding, set to None for not using this feature

    fig = plt.subplots(figsize=(12, 8))
    cal_throughput = True

    # Other params
    cuda_gpu = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_gpu)
    torch.cuda.set_device(cuda_gpu)
    torch.cuda.empty_cache()

    downsize = 224
    patch_extraction_size = 224
    mask_overlap = 80.0
    batch_size = 128
    cpu_workers = 40
    use_prob_threshold = 0.95 
    # None  # whether to give final prediction {0,1} based on certain probability

    torch.manual_seed(250)

    # read the files
    wsi_files = os.listdir(wsi_dir)
    wsi_files = [f for f in wsi_files if f.endswith("scn") or f.endswith("mrxs")]
    # get all files except temp directory containing patches

    print(f"Total files in {wsi_dir} directory are {len(wsi_files)}")

    path = os.path.join(wsi_dir, "cnn_ensemble")
    if not os.path.exists(path):
        os.mkdir(path)

    # start patching process
    for f in wsi_files:
        st = time.time()
        # find binary mask to locate tissue on WSI
        path = os.path.join(path, f.split(".")[0])
        # just take the name not extension
        if not os.path.exists(path):
            os.mkdir(path)
        w, h = create_binary_mask(wsi_dir, f, path, downsize=downsize)
        # print(f"Binary tissue mask created for {f}")
        # start splitting WSI into patches
        patch_folder = os.path.join(path, "patches")
        if not os.path.exists(patch_folder):
            os.mkdir(patch_folder)
            # assuming patches directory exists and patches are already created.
            create_patches(wsi_dir, f, path, patch_folder,  workers=cpu_workers,
                        patch_size=patch_extraction_size, mask_overlap=mask_overlap)

        data_loader, total_patches = data_generator(patch_folder,  test_transform=test_transform,
                                        batch_size=batch_size, worker=cpu_workers)

        if ensemble =="cnns": 
            print("\nLoading CNN ensemble of MobileNetv3")
            # blur
            blur_model = load_cnn_model(models_location, blur_cnn)
            blood_model = load_cnn_model(models_location, blood_cnn)
            fold_model = load_cnn_model(models_location, fold_cnn)
            damaged_model = load_cnn_model(models_location, damaged_cnn)
            airbubble_model = load_cnn_model(models_location, airbubble_cnn)

        else:
            blur_model = load_vit_model(models_location, blur_vit)
            blood_model = load_vit_model(models_location, blood_vit)
            fold_model = load_vit_model(models_location, fold_vit)
            damaged_model = load_vit_model(models_location, damaged_vit)
            airbubble_model = load_vit_model(models_location, airbubble_vit)

        flops, params = get_model_complexity_info(blur_model, ((3,224,224)),
                                                  as_strings=False, print_per_layer_stat=False)
        million_param = numerize.numerize(params*5)
        gflops = numerize.numerize(flops*5)
        print(f"\nTotal model Mparam {million_param} and GFlops {gflops} in the ensemble.")

        if torch.cuda.is_available():
            print("Cuda is available")
            # model should be on cuda before selection of optimizer
            blur_model = blur_model.cuda()
            blood_model = blood_model.cuda()
            damaged_model = damaged_model.cuda()
            fold_model = fold_model.cuda()
            airbubble_model = airbubble_model.cuda()

        print("\n########### Inference Starts ##############")
        st2 = time.time()
        if evaluate_with_prob is not None:
            print("Using thresholding @ ", evaluate_with_prob)
        if ensemble =="cnns":     

            blur_pred, blur_prob = infer_cnn(blur_model, data_loader, use_prob_threshold)
            blood_pred, blood_prob = infer_cnn(blood_model, data_loader, use_prob_threshold)
            damaged_pred, damaged_prob = infer_cnn(damaged_model, data_loader, use_prob_threshold)
            fold_pred, fold_prob = infer_cnn(fold_model, data_loader, use_prob_threshold)
            airbubble_pred, airbubble_prob = infer_cnn(airbubble_model, data_loader, use_prob_threshold)

            # blur_pred, y_true, blur_prob = infer_cnn_v3(blur_model, test_loader, use_prob_threshold=evaluate_with_prob)
            # blood_pred, y_true1, blood_prob= infer_cnn_v3(blood_model, test_loader, use_prob_threshold=evaluate_with_prob)
            # damaged_pred, y_true2, damaged_prob = infer_cnn_v3(damaged_model, test_loader, use_prob_threshold=evaluate_with_prob)
            # fold_pred, y_true, fold_prob = infer_cnn_v3(fold_model, test_loader, use_prob_threshold=evaluate_with_prob)
            # airbubble_pred, y_true, airbubble_prob = infer_cnn_v3(airbubble_model, test_loader, use_prob_threshold=evaluate_with_prob) 
        else: 
            blur_pred,  blur_prob = infer_vit(blur_model, test_loader, use_prob_threshold=evaluate_with_prob)
            blood_pred,  blood_prob = infer_vit(blood_model, test_loader, use_prob_threshold=evaluate_with_prob)
            damaged_pred,  damaged_prob = infer_vit(damaged_model, test_loader, use_prob_threshold=evaluate_with_prob)
            fold_pred,  fold_prob = infer_vit_v3(fold_model, test_loader, use_prob_threshold=evaluate_with_prob)
            airbubble_pred, airbubble_prob = infer_vit(airbubble_model, test_loader, use_prob_threshold=evaluate_with_prob)

        # setting them to boolean

        seconds = time.time()-st2
        minutes = seconds/60
        print(f"Time consumed in inference for {f} in {minutes:.2f} minutes.\n")

        # Calculate throughtput
        if cal_throughput:
            print("Throughput: {:.2f} patches/seconds".format(total_patches/seconds))


        blur_pred_b = np.array(blur_pred).astype(bool)
        blood_pred_b = np.array(blood_pred).astype(bool)
        damaged_pred_b = np.array(damaged_pred).astype(bool)
        fold_pred_b = np.array(fold_pred).astype(bool)
        airbubble_pred_b = np.array(airbubble_pred).astype(bool)

        # ensemble output
        artifact_list = [blur_pred_b[i] | blood_pred_b[i] | damaged_pred_b[i] | fold_pred_b[i] | airbubble_pred_b[i]
                         for i in range(len(blur_pred))]
        artifact_list = [a.astype(int) for a in artifact_list]

        file_names = [im.split("/")[-1] for im in data_loader.dataset.data_path]
        data = {"files": file_names, "predicted": artifact_list, "blur": blur_pred, "blood": blood_pred,
                "damage": damaged_pred, "fold": fold_pred, "airbubble": airbubble_pred,  "blur_p": blur_prob,
                "blood_p": blood_prob,  "damage_p": damaged_prob, "fold_p": fold_prob, "airbubble_p": airbubble_prob}

        dframe = pd.DataFrame(data)

        if use_prob_threshold is not None:
            print(f"Probablity threshold of {use_prob_threshold} used for determining overall prediction.\n")

        with pd.ExcelWriter(f"{path}/cnn_ensemble_predictions.xlsx") as wr:
            dframe.to_excel(wr, index=False)

        print("########### Postprocessing Starts ##########")
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
        print(f"Total for end-to-end processing for {f} in {minutes:.2f} minutes.")
