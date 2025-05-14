📘 Project Overview
This thesis investigates how incorporating clinical features improves the diagnostic performance and calibration of machine learning models detecting pleural effusion from chest X-rays. The study evaluates two types of models and is not used for improving the performance of Machine Learning models, but to capture the impact of including clinical features in models classifying pleural effusion (similar methods could be tested for other diseases capture on chest X-rays):

XGBoost model trained on tabular/clinical features

Deep learning model (ResNet-based) integrating image and tabular data

Two public datasets are used:

CheXpert: Chest X-ray images with metadata

CheXmask: Segmentation masks of anatomical structures

🗂 Repository Structure
bash
Copy
Edit
Bachelor-Project/
├── EDA.ipynb                  # Initial exploratory analysis of the dataset
├── features_extraction/       # Scripts for generating features like lung asymmetry and fluid levels, nd also a script used for extracting grey-scale histogram as well as corners detection
├── model_training.ipynb       # XGBoost training on structured tabular features
├── NN_train_exp1.py           # ResNet-based model trained on non-clinical features
├── NN_train_exp2.py           # ResNet-based model trained on clinical features
├── NN_train_exp3.py           # ResNet-based model trained on the whole data available
├── model_results/             # Saved performance metrices for each model and experimenr
├── charts and diagrams/       # Charts used in the report and the code used for generating the model architecture in LaTeX
├── report_images/             # Images embedded in the report
├── meeting_notes.md           # Timeline & meeting logs
├── Gabriela_proposal.pdf      # Initial research proposal
└── requirements.txt           # All dependencies used

🧪 How to Run the Code

As the project serves as my Bachelor Project at IT University of Copenhagen, the data was provided by my supervisors and thus the HPC owned by ITU was used for storing the data and running the models. 

Environment Setup
Clone the repository:

bash
Copy
Edit
git clone https://github.com/gazhds/Bachelor-Project.git
cd Bachelor-Project
Create a virtual environment (optional):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt

💡 GPU is strongly recommended for training deep learning models. Make sure CUDA and a GPU-enabled PyTorch/TensorFlow are properly installed.

XGBoost Model
Open and run model_training.ipynb in Jupyter Notebook.

This notebook trains and evaluates XGBoost models on different feature configurations (non-clinical, clinical-only, full).

Uses tabular data only.

Deep Learning Experiments (GPU)
The following scripts run models that combine image + tabular data and are optimized for GPU:

NN_train_exp1.py           ResNet-based model trained on non-clinical features
NN_train_exp2.py           ResNet-based model trained on clinical features
NN_train_exp3.py           ResNet-based model trained on the whole data available

Each script can be launched as:

bash
Copy
Edit
python exp1_resnet.py           # or replace with the corresponding file
All use a dual-branch architecture:

One CNN for image features

One MLP for tabular inputs (clinical + engineered features)

Late fusion of both representations for final classification

🧹 Data Preprocessing
Before model training, several cleaning and preprocessing steps were applied:

Label Filtering: Only samples explicitly labeled "positive", "negative", or "uncertain" for pleural effusion were included

Mask Decoding: CheXmask RLE masks decoded to binary arrays for lungs and fluid regions

Feature Engineering:

Greyscale histogram (global brightness)

Harris corners (structure-rich areas)

Lung asymmetry (shape and alignment difference between lungs)

Fluid intensity (brightness in masked pleural effusion region)

Tabular features were normalized and missing values were handled using:

XGBoost: internal missing-value handling

Deep Learning: kNN imputation (k=5)

📊 Results Summary
XGBoost with clinical-only features achieved the best AUC (0.873) and calibration

ResNet with combined features achieved AUC of 0.860

PCA-based dimensionality reduction reduced performance and is not recommended

🔁 For Future Work
Apply the models to external datasets (e.g., MIMIC-CXR, VinDr)

📬 Contact
For any questions, feel free to contact:
Gabriela Zhelyazkova
📧 gabigrigorieva016@gmail.com
