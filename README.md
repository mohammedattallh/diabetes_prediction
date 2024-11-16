# Diabetes Prediction Model

## Introduction
This project aims to develop a predictive model for assessing the risk of diabetes based on various health indicators. Using health data such as glucose levels, blood pressure, BMI, and more, we can create a machine learning model to aid in the early detection of diabetes. Early diagnosis enables timely intervention, which is crucial for preventing complications associated with diabetes and improving patient outcomes.

## Dataset Overview
The dataset used for this project includes several features that serve as indicators of diabetes risk. Below is a breakdown of each feature:
- **Id**: Unique identifier for each record in the dataset.
- **Pregnancies**: Number of times the patient has been pregnant.
- **Glucose**: Plasma glucose concentration after a 2-hour oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm), a measure of body fat.
- **Insulin**: Serum insulin level after 2 hours of an oral glucose tolerance test (mu U/ml).
- **BMI**: Body Mass Index, calculated as weight (kg) divided by height (m^2).
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
- **Age**: Age in years.
- **Outcome**: Binary variable indicating diabetes status; 1 for diabetic and 0 for non-diabetic.

This dataset provides a diverse range of health metrics to inform a predictive model and support insights into key risk factors for diabetes.

## Project Workflow
The project follows a structured workflow, detailed as follows:

### 1. Data Loading and Initial Inspection
The project begins with loading the dataset and performing an initial inspection. This includes:
   - **Verification of Data Structure**: Checking the number of records, column names, and data types for each feature.
   - **Handling Missing Values**: Initial examination of missing values across the dataset.
   - **Descriptive Statistics**: Summary statistics are computed to understand feature distributions and detect anomalies.

### 2. Data Cleaning and Missing Value Handling
Several health-related fields, such as glucose level, blood pressure, skin thickness, and BMI, contain zero values, which are likely incorrect and are treated as missing. The following steps were taken to handle missing data:
   - **Replacing Zeroes with NaNs**: Zero values in columns where zero is biologically implausible (like glucose and BMI) are replaced with NaNs.
   - **Median Imputation by Age Group**: Missing values in these fields are filled with the median value within the same age group, ensuring that data distribution is maintained.
   
   These steps are essential for reducing bias introduced by missing data, particularly in predictive modeling.

### 3. Feature Engineering
Feature engineering was conducted to improve model performance and reduce noise:
   - **Dropping the Id Column**: Since `Id` serves only as a unique identifier, it was removed as it doesn’t contribute to the predictive capabilities of the model.
   - **Predicting Missing Insulin Values**: To fill missing values in the `Insulin` column, a `DecisionTreeRegressor` model was trained using records with non-missing insulin data. This model predicted insulin levels based on other features (e.g., glucose, BMI, age), resulting in a more complete dataset.

### 4. Exploratory Data Analysis (EDA)
EDA provides insights into the data structure and feature relationships, including:
   - **Distribution Analysis**: Histograms and boxplots are used to visualize feature distributions and detect outliers. This helps identify potential data normalization or scaling requirements.
   - **Correlation Analysis**: A correlation heatmap is generated to examine the relationships between features. Features with higher correlation to the `Outcome` variable are of particular interest as they may significantly impact prediction.

### 5. Model Building and Selection
Several machine learning models were evaluated to determine the most suitable for predicting diabetes. The following models were built and compared:
   - **Logistic Regression**: A simple, interpretable model for binary classification, particularly effective when relationships between features and the outcome are linear.
   - **Decision Tree Classifier**: A non-linear model that provides interpretable decision rules for each prediction. Useful for capturing interactions between features.
   - **Random Forest Classifier**: An ensemble model that builds multiple decision trees and averages their predictions. This model is effective in improving accuracy and reducing overfitting, especially when there are complex feature interactions.

Each model was trained using a training dataset and tuned to optimize performance metrics relevant to diabetes prediction.

### 6. Model Evaluation and Performance Metrics
Model performance was assessed using multiple metrics to understand accuracy, precision, and reliability:
   - **Accuracy**: Measures the proportion of correct predictions over the total predictions. While accuracy is helpful, it may be misleading if classes are imbalanced.
   - **Precision**: Reflects the accuracy of positive predictions, i.e., the model’s ability to correctly predict cases of diabetes without many false positives.
   - **Recall**: Indicates the model’s ability to capture actual cases of diabetes. High recall is crucial in healthcare settings to minimize false negatives.
   - **Confusion Matrix**: Shows true positives, true negatives, false positives, and false negatives, providing a detailed performance breakdown.

These metrics were used to identify the best-performing model, and hyperparameter tuning was applied to further enhance model accuracy and reliability.
### 7. Deploying the Model with FastAPI
We use FastAPI to build an API endpoint that allows users to send data and receive a diabetes prediction. This enables real-time access to the model for prediction purposes.

### 1. Setting Up FastAPI
To deploy the model with FastAPI, ensure that `fastapi` and `uvicorn` libraries are installed. You can install them using the following command:
```bash
pip install fastapi uvicorn
```
# **Diabetes Prediction API**

This project provides a real-time diabetes prediction service using a pre-trained machine learning model deployed with **FastAPI**. The model is containerized using **Docker** for easy deployment and scalability.

---

## **Features**
- Predict diabetes status based on health metrics using an API endpoint.
- Containerized with Docker for platform-independent execution.
- Built with FastAPI for high performance and ease of use.

---

## **Setup Instructions**

### **1. Clone the Repository**
Clone this repository to your local machine:
```bash
git clone https://github.com/mohammedatallah20/diabetes-prediction-api.git
cd diabetes-prediction-api
```

### Docker Hub
The prebuilt Docker image for this project is available on Docker Hub:

**Image URL:** `mohammedatallah20/diabetes-prediction-api`

Pull the image using:

```bash
docker pull mohammedatallah20/diabetes-prediction-api:latest
```
### **Build and Run the Docker Container**

#### **1.Build the Docker Image**
Run the following command to build the Docker image:
```bash
docker build -t diabetes-prediction-api .
```
#### **2.Run the Container**
Start the container with:

```bash
docker run -p 8030:8030 diabetes-prediction-api
```
The API will be accessible at http://localhost:8030.
## Requirements
To run this notebook, you will need the following libraries:
- **Core Libraries**: `numpy`, `pandas`, `matplotlib`, and `seaborn` for data manipulation and visualization.
- **scikit-learn**: For building and evaluating machine learning models.

### Installing Requirements
Install the required libraries by running:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
