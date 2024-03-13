# Equipping-Computational-Pathology-Systems-with-Artifact-Processing-Pipeline

This repository contains source-code for paper "Equipping Computational Pathology Systems with Artifact Processing Pipelines: A Showcase for Computation and Performance Trade-offs"

#Abstract
Histopathology is a gold standard for cancer diagnosis under a microscopic examination. However, histological tissue processing procedures result in artifacts, which are ultimately transferred to the digitized version of glass slides, known as whole slide images (WSIs). Artifacts are diagnostically irrelevant areas and may result in wrong deep learning (DL) algorithms predictions. Therefore, detecting and excluding artifacts in the computational pathology (CPATH) system is essential for reliable automated diagnosis. In this paper, we propose a mixture of experts (MoE) scheme for detecting five notable artifacts, including damaged tissue, blur, folded tissue, air bubbles, and histologically irrelevant blood from WSIs. First, we train independent binary DL models as experts to capture particular artifact morphology. Then, we ensemble their predictions using a fusion mechanism. We apply probabilistic thresholding over the final probability distribution to improve the sensitivity of the MoE. We developed DL pipelines using two MoEs and two multiclass models of state-of-the-art deep convolutional neural networks (DCNNs) and vision transformers (ViTs). DCNNs-based MoE and ViTs-based MoE schemes outperformed simpler multiclass models and were tested on datasets from different hospitals and cancer types, where MoE using DCNNs yielded the best results. The proposed MoE yields 86.15% F1 and 97.93% sensitivity scores on unseen data, retaining less computational cost for inference than MoE using ViTs. This best performance of MoEs comes with relatively higher computational trade-offs than multiclass models. The proposed artifact detection pipeline will not only ensure reliable CPATH predictions but may also provide quality control.

Link to preprint: [TBA](https://arxiv.org/abs/2403.07743)

An overview figure from the paper:

<img width="331" alt="image" src="https://github.com/NeelKanwal/Equipping-Computational-Pathology-Systems-with-Artifact-Processing-Pipeline/assets/52494244/5669799b-2a5b-40f0-9bac-8dbbe9b4c87f">



# Requirements
- Histolab
- Pytorch
- Pandas
- Numpy
- Scikit-learn
- Timm
- Pyvips
- OpenSlide python
- MMCV

Use requirement.txt for complete packages.

# Dataset 

The dataset is publicaly available at Zenodo. https://zenodo.org/records/10809442.

This work only uses D40x for training and development. 


<img width="354" alt="image" src="https://github.com/NeelKanwal/Equipping-Computational-Pathology-Systems-with-Artifact-Processing-Pipeline/assets/52494244/40bd42ee-8aff-4990-8160-a63e68580fac">

<img width="379" alt="image" src="https://github.com/NeelKanwal/Equipping-Computational-Pathology-Systems-with-Artifact-Processing-Pipeline/assets/52494244/e6b621a3-a7bc-4cae-9aa8-a0a7b563a4a4">


- To process your own WSIs for preparing a dataset, use files in the preprocess directory.
- Link to the publicly available processed dataset: TBA

# Results

<img width="341" alt="image" src="https://github.com/NeelKanwal/Equipping-Computational-Pathology-Systems-with-Artifact-Processing-Pipeline/assets/52494244/0eabba34-4c5d-4b55-a907-356fc4ab3f46">


<img width="411" alt="image" src="https://github.com/NeelKanwal/Equipping-Computational-Pathology-Systems-with-Artifact-Processing-Pipeline/assets/52494244/a694aa8f-7183-46a9-8388-38dd55d9b617">

# Cite
"
@misc{kanwal2024equipping,
    title={Equipping Computational Pathology Systems with Artifact Processing Pipelines: A Showcase for Computation and Performance Trade-offs},
    author={Neel Kanwal and Farbod Khoraminia and Umay Kiraz and Andres Mosquera-Zamudio and Carlos Monteagudo and Emiel A. M. Janssen and Tahlita C. M. Zuiverloon and Chunmig Rong and Kjersti Engan},
    year={2024},
    eprint={2403.07743},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
"
