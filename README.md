# Robust Distributed Cooperative Classification with Learned Compressed-Feature Diffusion

This repository contains the code for the paper "Robust Distributed Cooperative Classification with Learned Compressed-Feature Diffusion".

This paper introduces a novel framework for distributed inference over sensor networks. It is designed to address the key challenges of limited communication bandwidth and the risk of node failures.

In our approach, each node performs local inference using its own features and learned compressed feature representations received from other nodes. The framework is built on two key components:

1.  A **trainable feature compressor** at each node to reduce data transmission while preserving critical information.
2.  An **adaptive node weighting mechanism** that dynamically adjusts the influence of local and remote features, providing robustness to unreliable or failed nodes.

## Core Code Structure

  * **`models.py`**:

      * Contains the core PyTorch implementation of the `CFD_DC` model.
      * `CFD_DC`: The main model class, which creates node-specific `compressors`, `classifiers`, and `se_layers` (the weighting module from the paper).
      * `CFD_DC_NoWeighting`: An model that removes the adaptive weighting module.
      * `CFD_DC_50Compress`: The model used for the `d_g = d_f / 2` setting in the multi-view experiments.
      * `SELayer`: Implements the Squeeze-and-Excitation (SE) block as the adaptive weighting mechanism.

  * **`data_loader.py`**: Handles loading and preprocessing for the multi-view and underwater acoustic datasets.

## Data Preparation

The underwater acoustic experiments require pre-processed features. The files `underwater_embedding_20.pt` (for the scenario with a **20% node failure rate**) and `underwater_y.pt` are hosted on Google Drive: https://drive.google.com/drive/folders/1dy79RGW_Iia7wsIWs1jKhbSP5-y777Ec?usp=sharing

## Example Run Command

You can use the following command to reproduce the results in Table 4 on a multi-view dataset (e.g., Handwritten):

```bash
python train_multiview.py --dataset Handwritten --dg 8
```

To reproduce the results in Table 8 (Underwater acoustic classification with a 20% node failure rate):

```bash
python train_underwater.py --dg 32 --failure_rate 0.2 --weighting
```
