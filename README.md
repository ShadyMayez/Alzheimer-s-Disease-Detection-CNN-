# Alzheimer's Disease Detection using CNN (OASIS Dataset)

## Project Overview and Purpose
This project aims to automate the detection of Alzheimer's Disease stages from Brain MRI scans. Utilizing the OASIS (Open Access Series of Imaging Studies) dataset, the model classifies images into four clinical stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. This tool is designed to assist clinicians in the early diagnosis and monitoring of neurodegenerative progression.

## Key Technologies and Libraries
- **Framework**: TensorFlow / Keras
- **Image Processing**: OpenCV, ImageDataGenerator
- **Analysis**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## Methodology and Analysis Workflow
### 1. Data Pipeline
- **Data Loading**: Systematically scanned the directory to create a DataFrame of file paths and labels.
- **Data Splitting**: Partitioned the data into Training, Validation, and Test sets (80/10/10 split) to ensure robust evaluation.
- **Preprocessing**: Images were resized to 224x224 and normalized. Used `ImageDataGenerator` for efficient batch loading.

### 2. Model Architecture
Built a deep Convolutional Neural Network (CNN) featuring:
- **Feature Extraction**: Multiple `Conv2D` layers with increasing complexity (filters ranging from 32 to 128) to identify structural brain anomalies.
- **Normalization**: `BatchNormalization` after each convolution to accelerate training and reduce internal covariate shift.
- **Regularization**: `Dropout` layers to prevent the model from memorizing specific training images (overfitting).
- **Classification**: Dense layers leading to a 4-unit Softmax output.



### 3. Optimization
- **Optimizer**: `Adamax` for superior performance on sparse data.
- **Callbacks**: Implemented `EarlyStopping` and `ReduceLROnPlateau` to monitor validation loss and adjust the learning rate dynamically.

## Results and Insights
- **Accuracy & Loss**: The training history shows a consistent convergence of loss and a high accuracy rate on the validation set.
- **Evaluation**: The project includes a full classification report and a confusion matrix to identify which stages of dementia are most similar in visual features.
- **Visual Predictions**: A prediction script displays test images along with the model's confidence percentage and the predicted class.

## How to Run
1. **Dataset**: Ensure the OASIS brain MRI dataset is available in your directory path.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
