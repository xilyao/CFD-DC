Here is the concise README, translated into English as requested:

-----

# CFD-DC: Cooperative Classification via Adaptive Compressed-Feature Diffusion over Distributed Networks

This repository contains the code for the paper "CFD-DC: Cooperative Classification via Adaptive Compressed-Feature Diffusion over Distributed Networks".

This paper introduces CFD-DC, a novel framework for decentralized classification in distributed sensor networks. It is designed to address the key challenges of limited communication bandwidth and the risk of node failures.

In our approach, nodes collaborate by broadcasting highly compressed feature representations. The framework is built on two key components:

1.  A **trainable feature compressor** at each node to reduce data transmission while preserving critical information.
2.  An **adaptive node weighting mechanism** that dynamically adjusts the influence of features from other nodes, providing robustness against failures.

## 🔧 Core Code Structure

  * **`models.py`**:

      * Contains the core PyTorch implementation of the `CFD_DC` model.
      * `CFD_DC`: The main model class, which creates node-specific `compressors`, `classifiers`, and `se_layers` (the weighting module from the paper).
      * `CFD_DC_NoWeighting`: An ablation model used for comparison, which removes the adaptive weighting module.
      * `CFD_DC_50Compress`: The model used for the `d_g = d_f / 2` setting in the multi-view experiments.
      * `SELayer`: Implements the Squeeze-and-Excitation (SE) block as the adaptive weighting mechanism.

  * **`data_loader.py`**: Handles loading and preprocessing for the multi-view and underwater acoustic datasets.

  * **`train_utils.py`**: Contains training utilities, such as `EarlyStopping` and an LR scheduler.

  * **`utils.py`**: Contains helper functions, such as `custom_global_avg_pool` (which prepares inputs for the SE module).

## Example Run Command

You can run the training for a multi-view dataset (e.g., Handwritten) using the following command:

```bash
python train_multiview.py --dataset Handwritten --dg 8
``