# 💳 Machine Learning-Based Fraud Detection Using PaySim1 Dataset

## 📌 Project Overview

Financial transaction fraud is a major challenge in digital banking and mobile payment systems. Detecting fraudulent transactions accurately is essential to protect customers and financial institutions from financial losses.

This project develops a **Machine Learning-Based Fraud Detection System** using the PaySim1 simulated mobile-money transaction dataset. It applies machine learning classification algorithms to identify suspicious financial transactions and classify them as fraudulent or legitimate.

Two machine learning algorithms are implemented and compared:

* Logistic Regression
* Random Forest Classifier

The models are evaluated using multiple performance metrics to identify the most suitable model for fraud detection.

**Note:** This project uses simulated PaySim transaction data and does not contain genuine banking transaction records.

---

## 🎯 Project Objectives

* Analyze and understand financial transaction data.
* Perform data exploration and preprocessing.
* Identify fraudulent and non-fraudulent transaction patterns.
* Visualize transaction distributions and fraud patterns.
* Implement Logistic Regression and Random Forest algorithms.
* Evaluate model performance using classification metrics.
* Compare both models and select the best-performing model.
* Predict whether a new transaction is potentially fraudulent.
* Save trained models, evaluation reports, visualizations, and predictions.

---

## 📂 Dataset Description

**Dataset:** PaySim1 – Simulated Mobile Money Transactions

The dataset is obtained from Kaggle using KaggleHub.

Dataset source:

[PaySim1 Dataset – Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

### Dataset Features

| Feature        | Description                                     |
| -------------- | ----------------------------------------------- |
| step           | Represents the time step of the transaction     |
| type           | Type of financial transaction                   |
| amount         | Transaction amount                              |
| nameOrig       | Customer initiating the transaction             |
| oldbalanceOrg  | Origin account balance before transaction       |
| newbalanceOrig | Origin account balance after transaction        |
| nameDest       | Destination customer                            |
| oldbalanceDest | Destination account balance before transaction  |
| newbalanceDest | Destination account balance after transaction   |
| isFraud        | Indicates whether the transaction is fraudulent |
| isFlaggedFraud | Indicates whether the transaction was flagged   |

### Target Variable

* `0` – Non-Fraudulent Transaction
* `1` – Fraudulent Transaction

The dataset contains highly imbalanced classes because fraudulent transactions represent a very small proportion of all transactions.

---

## 🛠️ Technologies Used

| Technology   | Purpose                                    |
| ------------ | ------------------------------------------ |
| Python       | Programming language                       |
| Google Colab | Development environment                    |
| Pandas       | Data manipulation and analysis             |
| NumPy        | Numerical computations                     |
| Matplotlib   | Data visualization                         |
| Seaborn      | Statistical visualization                  |
| Scikit-learn | Machine learning algorithms and evaluation |
| KaggleHub    | Dataset downloading                        |
| Joblib       | Saving trained ML models                   |
| ZIP          | Packaging project outputs                  |

---

## ⚙️ Project Workflow

The project follows these major stages:

### 1. Dataset Download and Loading

* Download the PaySim1 dataset using KaggleHub.
* Load the CSV file into a Pandas DataFrame.
* Display the dataset structure and initial records.

### 2. Data Exploration

* Examine dataset dimensions.
* Identify column names and data types.
* Generate descriptive statistical summaries.
* Check missing values and duplicate records.
* Examine transaction types and target labels.

### 3. Data Cleaning and Preprocessing

* Remove duplicate records.
* Convert numerical columns into appropriate data types.
* Remove invalid target labels.
* Handle missing values in required columns.
* Preserve valid zero balances and unusual transaction amounts.

### 4. Exploratory Data Analysis (EDA)

The following visualizations are generated:

* Fraud vs Non-Fraud Transaction Counts
* Transaction Type Distribution
* Transaction Amount Distribution
* Fraud Counts by Transaction Type
* Numerical Feature Correlation Heatmap

These visualizations help identify patterns and understand the characteristics of financial transactions.

### 5. Feature Selection

The following seven features are selected for model training:

1. Step
2. Type
3. Amount
4. Old Balance Origin
5. New Balance Origin
6. Old Balance Destination
7. New Balance Destination

The target variable is `isFraud`.

Raw customer IDs and the existing fraud flag are excluded from the baseline model.

### 6. Data Preprocessing

Different preprocessing techniques are applied:

**Numerical Features:**

* Median imputation
* Standardization using StandardScaler

**Categorical Features:**

* Most frequent value imputation
* One-Hot Encoding

A ColumnTransformer combines these preprocessing operations.

### 7. Model Training

Two machine learning algorithms are implemented.

**Logistic Regression**

* A supervised classification algorithm.
* Estimates the probability of a transaction being fraudulent.
* Uses balanced class weights to address class imbalance.

**Random Forest Classifier**

* An ensemble learning algorithm.
* Combines multiple decision trees.
* Uses majority voting for classification.
* Uses balanced class weights to improve fraud-class learning.

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

Stratified splitting maintains approximately the same fraud distribution in both sets.

### 8. Model Evaluation

Both algorithms are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Average Precision

Confusion matrices and classification reports are also generated.

### 9. Model Comparison

The performance of Logistic Regression and Random Forest is compared.

The model with the highest Average Precision is selected as the final model.

A decision threshold of 0.50 is used for classification.

### 10. Transaction Prediction

The selected model predicts whether an individual transaction is fraudulent or legitimate.

The prediction function returns:

* Predicted class
* Predicted fraud probability
* Decision threshold

### 11. Output Generation

The trained model, evaluation results, feature importance, charts, and prediction results are saved.

All generated outputs are compressed into a ZIP file for downloading.

---

## 📊 Machine Learning Algorithms

### Logistic Regression

Logistic Regression estimates the probability that a transaction belongs to the fraudulent class.

It uses the sigmoid function:

P(y=1) = 1 / (1 + e^(-z))

Where:

* P(y=1) represents the probability of fraud.
* z represents the weighted sum of input features.

### Random Forest

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction performance.

It reduces overfitting compared with relying on a single decision tree and can capture complex relationships between transaction features.

---

## 📈 Model Evaluation Metrics

The following metrics are used to evaluate the classification models.

| Metric            | Description                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| Accuracy          | Measures the overall proportion of correct predictions                 |
| Precision         | Measures how many predicted fraud transactions are actually fraudulent |
| Recall            | Measures how many actual fraud transactions are correctly detected     |
| F1-Score          | Harmonic mean of precision and recall                                  |
| ROC-AUC           | Measures the model's ability to distinguish between classes            |
| Average Precision | Summarizes the precision-recall performance                            |

### Important Formulas

**Accuracy**

Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Precision**

Precision = TP / (TP + FP)

**Recall**

Recall = TP / (TP + FN)

**F1-Score**

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Where:

* TP – True Positive
* TN – True Negative
* FP – False Positive
* FN – False Negative

Since fraud transactions are rare, Precision, Recall, F1-Score, and Average Precision are particularly important for evaluating this project.

---

## 📁 Project Output Files

The project automatically generates the following files inside the `fraud_detection_outputs` folder.

| File Name                            | Description                          |
| ------------------------------------ | ------------------------------------ |
| final_fraud_detection_model.joblib   | Saved trained machine learning model |
| model_comparison_metrics.csv         | Comparison of both algorithms        |
| final_model_metrics.json             | Final model evaluation metrics       |
| final_classification_report.txt      | Classification performance report    |
| final_confusion_matrix.csv           | Final confusion matrix               |
| random_forest_feature_importance.csv | Feature importance values            |
| cleaning_summary.json                | Data cleaning information            |
| model_configuration.json             | Model configuration and settings     |
| test_set_predictions.csv             | Actual labels and predicted results  |
| project_results_report.txt           | Complete project results summary     |

### Generated Visualizations

* fraud_counts.png
* transaction_types.png
* amount_distribution.png
* fraud_by_type.png
* correlation_heatmap.png
* logistic_regression_confusion_matrix.png
* random_forest_confusion_matrix.png
* random_forest_feature_importance.png

All these files are packaged into:

`fraud_detection_outputs.zip`

---

## 🚀 How to Run the Project

### Step 1: Open Google Colab

Open [Google Colab](https://colab.research.google.com/).

### Step 2: Create a New Notebook

Create a new Python notebook.

### Step 3: Install Required Library

```python
%pip install kagglehub
```

### Step 4: Run the Project Code

Copy the complete project code into a notebook cell.

### Step 5: Download the Dataset

The dataset is automatically downloaded using KaggleHub.

```python
dataset_path = kagglehub.dataset_download("ealaxi/paysim1")
```

### Step 6: Execute the Code

Run the complete cell.

The program performs data preprocessing, visualization, model training, evaluation, and prediction.

### Step 7: Download the Results

After execution, the generated ZIP file will automatically download to your computer.

**Note:** Kaggle access and internet connectivity are required for the initial dataset download. Full-dataset training may require substantial RAM and execution time.

---

## 💡 Key Highlights

* Uses the PaySim1 financial transaction dataset.
* Implements two supervised machine learning algorithms.
* Handles class imbalance using balanced class weights.
* Uses numerical scaling and categorical encoding.
* Performs exploratory data analysis through multiple visualizations.
* Compares models using multiple evaluation metrics.
* Selects the final model based on Average Precision.
* Supports individual transaction predictions.
* Saves reusable trained models and evaluation reports.

---

## ⚠️ Limitations

* PaySim is a simulated dataset and does not represent genuine banking transactions.
* The current implementation uses tabular machine learning techniques.
* Graph traversal algorithms such as BFS and DFS are not implemented in this version.
* A web-based application or real-time banking integration is not included.
* Model performance on simulated data may differ from performance on real-world banking data.
* The default decision threshold has not been optimized.

---

## 🔮 Future Enhancements

* Integrate graph-based transaction analysis using NetworkX.
* Implement BFS and DFS to explore suspicious transaction relationships.
* Combine graph traversal with machine learning predictions.
* Develop a web-based fraud detection dashboard.
* Integrate real-time transaction monitoring.
* Experiment with advanced machine learning and anomaly detection techniques.
* Optimize the classification threshold using validation data.
* Evaluate the system using real-world financial datasets where available.

---

## 👩‍💻 Project Information

**Project Title:** Machine Learning-Based Fraud Detection Using PaySim1 Dataset

**Domain:** Artificial Intelligence and Machine Learning

**Category:** Financial Data Analytics and Fraud Detection

**Development Environment:** Python / Google Colab

**Dataset:** PaySim1 Simulated Mobile Money Transactions

---

## 📜 Disclaimer

This project is developed for academic and educational purposes. The PaySim1 dataset contains simulated mobile-money transactions. The results should not be considered evidence of performance in actual banking environments.

---

## ⭐ Conclusion

This project demonstrates how machine learning can be applied to financial transaction analysis and fraud detection.

By implementing and comparing Logistic Regression and Random Forest, the system identifies potentially fraudulent transactions and evaluates model performance using multiple classification metrics.

The project provides a foundation for future enhancements involving graph traversal, advanced fraud analysis, and real-time fraud detection applications.

---

**If you find this project useful, consider giving this repository a ⭐!**
