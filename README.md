
# 🫁 Bachelor Thesis – Detecting Pleural Effusion in Chest X-rays with Clinical Features

**Author:** Gabriela Zhelyazkova  
**Program:** BSc in Data Science  
**Institution:** IT University of Copenhagen  
**Supervisors:** Dr. Amelia Jiménez-Sánchez, Dr. Veronika Cheplygina  
**Date:** 2025  

---

## 📘 Project Overview

This thesis investigates how incorporating clinical features improves the diagnostic performance and calibration of machine learning models detecting pleural effusion from chest X-rays. The study evaluates two types of models—not with the goal of improving machine learning performance in general, but to understand the impact of **clinical feature inclusion** in classification models for pleural effusion. Similar methods could be tested for other chest X-ray-based disease detection tasks.

Models:
- **XGBoost**: Trained on tabular/clinical features only
- **ResNet-based deep learning**: Integrating image and tabular inputs

**Datasets:**
- **CheXpert** – Chest X-ray images with metadata  
- **CheXmask** – Segmentation masks of anatomical structures  

---

## 🗂 Repository Structure

```
Bachelor-Project/
├── EDA.ipynb                  # Initial exploratory analysis of the dataset
├── features_extraction/       # Scripts for generating features like lung asymmetry, fluid levels, histogram, corners
├── model_training.ipynb       # XGBoost training on structured tabular features
├── NN_train_exp1.py           # ResNet-based model trained on non-clinical features
├── NN_train_exp2.py           # ResNet-based model trained on clinical features
├── NN_train_exp3.py           # ResNet-based model trained on the full dataset
├── model_results/             # Saved performance metrics per model and experiment
├── charts and diagrams/       # Charts used in the report + LaTeX code for model diagrams
├── report_images/             # Figures embedded in the report
├── meeting_notes.md           # Timeline & meeting logs
├── Gabriela_proposal.pdf      # Initial research proposal
└── requirements.txt           # All dependencies used
```

---

## 🧪 How to Run the Code

> ⚠️ The project was conducted at ITU using high-performance computing (HPC) resources provided by supervisors. Data is not included in the repository.

### Environment Setup

```bash
git clone https://github.com/gazhds/Bachelor-Project.git
cd Bachelor-Project

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

💡 **GPU is strongly recommended** for training deep learning models. Make sure CUDA and a GPU-enabled PyTorch or TensorFlow is installed.

---

### XGBoost Model

- Open `model_training.ipynb` in Jupyter Notebook
- This trains and evaluates XGBoost models using:
  - Non-clinical features
  - Clinical-only features
  - Full combined feature set
- Works with tabular data only (no images)

---

### Deep Learning Experiments (GPU)

These scripts run dual-branch models (CNN + MLP):

- `NN_train_exp1.py` – Non-clinical features + image
- `NN_train_exp2.py` – Clinical features + image
- `NN_train_exp3.py` – Full feature set + image

Launch any of them with:

```bash
python NN_train_exp1.py
```

Architecture:
- CNN (ResNet) branch processes images  
- MLP processes tabular features  
- Late fusion merges both for classification  

---

## 🧹 Data Preprocessing

### Steps before training:

- **Label filtering**: Only "positive", "negative", and "uncertain" pleural effusion labels were used
- **Mask decoding**: CheXmask RLE encoded masks decoded into binary segmentation arrays
- **Feature Engineering**:
  - Greyscale histogram
  - Harris corners
  - Lung asymmetry
  - Fluid intensity

### Handling Missing Values:

- **XGBoost**: Uses native missing-value support  
- **Deep Learning**: k-NN imputation with `k=5`

---

## 📊 Results Summary

| Model        | Configuration    | AUC   | Comments                     |
|--------------|------------------|-------|------------------------------|
| XGBoost      | Clinical Only     | 0.873 | Best performance overall     |
| ResNet-based | Combined features | 0.860 | Best among deep models       |
| PCA variants | Any config        | ↓     | Decreased performance noted  |

*PCA-based dimensionality reduction is not recommended.*

---

## 🔁 For Future Work

- Apply models to **external datasets** (e.g., MIMIC-CXR, VinDr)
- Extend to **multi-label classification** (e.g., comorbidities)
- Explore **inferring clinical features from images** using multi-task learning

---

## 📬 Contact

For questions or further collaboration:

**Gabriela Zhelyazkova**  
📧 [gabigrigorieva016@gmail.com](mailto:gabigrigorieva016@gmail.com)

---
