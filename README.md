# Representation Learning for Hyperspectral Image Classification under Data Scarcity

[![Paper](https://img.shields.io/badge/Paper-IEEE_JSTARS-blue.svg)](URL_TO_YOUR_PAPER)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the methodology proposed in
**"Assessing the influence of using a reduced number of samples for creating different representation models for hyperspectral image classification"**

---

## 📌 Overview

This repository provides the complete experimental pipeline to evaluate the impact of training sample size ($k$) on various Representation Learning (RL) models for Hyperspectral Image (HSI) classification. The study rigorously compares **pixel-wise baselines** against advanced **spatial-spectral models** (including Vision Transformers) under extreme data scarcity constraints.

### Evaluated Architectures:
* **Pixel-wise Baselines:** Principal Component Analysis (PCA), Autoencoder (AE).
* **Spatial-spectral Baselines:** Joint Spatial Autoencoder (AE-Patch).
* **Transformer-based Models:** Vision Transformer (VIT), Masked Autoencoder VIT (MAE-VIT), Self-Distillation VIT (DINO-VIT).

---

## 🧠 Proposed Methodology

<p align="center">
  <img src="Results/Fig_Metodologia.png" width="800" alt="Methodology Flowchart">
</p>

The flowchart above illustrates the overarching Representation Learning (RL) pipeline evaluated in this study. The fundamental goal is to map the high-dimensional hyperspectral input $\mathcal{X}$ into a robust, lower-dimensional latent space $Z \in \mathbb{R}^d$ while severely restricting the training data pool ($k$). Once these compact features are extracted, a downstream Classifier (CL) evaluates the topological quality of the latent space $Z$ by predicting the final land-cover labels $\mathcal{Y}$.

---

## 📁 Repository Structure

* `01_hsi_representation_learning.ipynb`: The core experimental notebook bridging the theoretical framework with the execution code. Its internal structure is highly modular:
  * **LIBRARIES DEFINITION & FUNCTIONS DEFINITIONS:** Core setup, including the definition of the `RL STAGE` architectures (e.g., PCA, AE, ViT, MAE-ViT) and the `CLASIFICATION STAGE` downstream models (SVM, RF, XGBoost).
  * **EXPERIMENTAL CONFIGURATIONS ⭐:** The control center where critical hyperparameters are set:
    * `DATASET_ID`: The target dataset (e.g., Indian Pines, Pavia, Salinas, KSC).
    * `K`: The array of training sample proportions ($k$, e.g., 0.25 to 1.0).
    * `D`: The target latent dimensions ($d$, e.g., 2 to 64).
    * `LEARNERS` & `CLASSIFIERS`: The specific sets of RL architectures and downstream models to evaluate.
  * **REPRESENTATION LEARNINGS:**
    * **STAGE I: DATASET PREPARATION FOR REPRESENTATION LEARNING (DPRL):** The hyperspectral input $\mathcal{X}$ is strictly partitioned based on the $k$-subset constraint to prevent spatial-spectral leakage.
    * **STAGE II: REPRESENTATION LEARNING (RL):** Automated execution of training loops to learn the mapping function $f_\theta: \mathcal{X} \rightarrow Z$ across the combinatorial grid of models and $d$ dimensions.
  * **CLASIFICATION:**
    * **STAGE I: DATASET PREPARATION FOR CLASSIFICATION (DPC):** The data is loaded and spatial/spectral patches are extracted depending on the model requirement.
    * **STAGE II: CLASSIFICATION LEARNING (CL):** This main block executes three concurrent tasks exactly as depicted in the methodology flowchart:
      * **Feature Extraction (FE):** The previously trained mapping function $f_\theta$ is applied to project the raw test data into the latent space $Z$.
      * **Classification Learning (CL):** Traditional classifiers are trained on the generated embeddings $Z$ to predict the land-cover classes $\mathcal{Y}$.
      * **Classification Evaluation (CE):** Cross-validation metrics (F1-Macro, OA, Class-wise F1) are rigorously computed and exported to `.csv`.
* `02_generate_visualizations.py`: An automated, publication-ready visualization script. It parses the raw output CSVs to generate:
  * Classifier Stability Boxplots (F1-macro across datasets).
  * Latent Convergence Analysis Grids ($d$ vs F1-macro).
  * Class-wise Delta ($\Delta$) Heatmaps ($\Delta = F1_{k=1.0} - F1_{k=0.25}$).
* `environment.yml`: The Conda configuration file containing the exact versions of the libraries used, guaranteeing full reproducibility.
* `Raw_Reports/`: Contains the raw CSV output (`*_Combined_Performance.csv`) from the trained models.
* `Results/` & `Results_Delta/`: Directories generated automatically by the plotting script containing the essential high-resolution vector figures (`.pdf`) used in the paper.

---

## 🚀 Quick Start

### 1. Environment & Requirements
We provide an `environment.yml` file to easily recreate the exact conda environment used for these experiments. The core dependencies include:
* **Python 3.11**
* **TensorFlow 2.21 & Keras 3.14** (Used for AE, VIT, MAE-VIT, DINO-VIT)
* **Scikit-Learn 1.8** (Used for PCA, SVM, RF, DT, NB)
* **XGBoost 3.2**
* **Pandas 3.0, Numpy 1.26, Matplotlib 3.10**

To create and activate the environment:
```bash
conda env create -f environment.yml
conda activate hsi_deep
```

### 2. Running the Experiments
To replicate the core experiments and generate the raw performance metrics (`*_Combined_Performance.csv`):
1. Open and execute `01_hsi_representation_learning.ipynb`.
2. The notebook will automatically partition the datasets, train the RL models across different latent dimensions ($d$) and data proportions ($k$), and evaluate the downstream classifiers.

### 3. Generating Publication Figures
To recreate the exact figures and tables featured in the paper from the raw results, run:
```bash
python 02_generate_visualizations.py Raw_Reports Results
```
*Note: The script automatically handles imbalance-aware evaluation, prioritizing F1-Macro over Overall Accuracy (OA) to mitigate majority-class bias.*

---

## 📊 Evaluation & Metric Divergence

This framework incorporates a strict **Metric Divergence Check**. As demonstrated in our experiments, optimizing exclusively for Overall Accuracy (OA) often leads to catastrophic failure in minority classes. This repository automatically generates divergence tables that empirically justify the selection of F1-Macro as the primary optimization target for imbalanced HSI datasets.

---

## 📝 Citation
If you find this code or our methodology useful in your research, please consider citing our published work:

```bibtex
@inproceedings{AymaQuirita2026,
  author    = {Ayma Quirita, Victor Andres and Palacios, Aramis and Ayma Quirita, Victor Hugo and Aliaga, Walter and Costa, Gilson A. O. P.},
  title     = {Impact of Training Set Size on Representation Learning for Hyperspectral Image Classification},
  booktitle = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume    = {X-3/W4-2025},
  pages     = {21--26},
  year      = {2026},
  doi       = {10.5194/isprs-annals-X-3-W4-2025-21-2026}
}
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
