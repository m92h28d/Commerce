A deep learning architecture for analyzing and predicting customer churn data in e-commerce

__Description__
This work aims to predict customer churn in an e-commerce dataset using a deep learning approach. The analysis involves data preprocessing, exploratory data analysis, model training with K-fold cross-validation, and evaluation using various metrics. The dataset is imbalanced, and SMOTE (Synthetic Minority Over-sampling Technique) is applied to address this imbalance. SHAP (SHapley Additive exPlanations) is used for model interpretability and feature importance analysis.

__Dataset Information__ 
he provided e-commerce dataset contains a comprehensive set of variables designed to capture various aspects of customer behavior and characteristics, ultimately aiming to predict customer churn. Key identifiers include CustomerID, a unique identifier for each customer, and Churn, a binary flag indicating whether a customer has churned. Customer engagement is measured by Tenure (customer's time with the organization), HourSpendOnApp (hours spent on the app/website), OrderCount (total orders placed), DaySinceLastOrder (days since last order), CouponUsed (coupons used), and CashbackAmount (average cashback received). Demographic and preference data include PreferredLoginDevice, CityTier, WarehouseToHome (distance from warehouse to home), PreferredPaymentMode, Gender,  PreferedOrderCat (preferred order category), and  MaritalStatus. Additional customer service and satisfaction indicators are SatisfactionScore, NumberOfAddress (total addresses added), NumberOfDeviceRegistered (total devices registered), Complain (any complaint raised), and OrderAmountHikeFromlastYear (percentage increase in order amount from last year). These variables collectively provide a rich dataset for analyzing and predicting customer churn.

__Code Information__
This script is a Python implementation that:
Loads and preprocesses the e-commerce dataset.
Performs exploratory data analysis (EDA) including visualizations of churn distribution, preferred login device, tenure, and cashback amount.
Transforms categorical and numerical features.
Splits the data into training, validation, and testing sets.
Builds and trains a sequential deep learning model using Keras.
Applies SMOTE to the training data to handle class imbalance.
Evaluates the model using K-fold cross-validation.
Calculates and visualizes training and validation accuracy and loss.
Generates a confusion matrix and reports accuracy, precision, recall, and F1-score.
Utilizes SHAP for feature importance analysis and model interpretability.

__Usage Instructions__
To run this code:
Environment Setup: Ensure you have Python installed. The script uses PySpark, TensorFlow, Keras, scikit-learn, imblearn, pandas, numpy, matplotlib, and seaborn.
Dataset Placement: Place ECommerceDataset2.csv in the /content/drive/MyDrive/ML/ directory or update the spark.read.csv path accordingly.
Execution: Run the commerceforpeerj.py script. It can be executed in a Jupyter notebook or Google Colab environment.

__Requirements__
* Python 3.x
* pyspark==3.1.2
* tensorflow
* keras
* scikit-learn
* imblearn
* pandas
* numpy
* matplotlib
* seaborn
* shap

__Methodology__
The methodology involves:
Data Loading: Reading the ECommerceDataset2.csv into a Spark DataFrame and converting it to a Pandas DataFrame.
Exploratory Data Analysis (EDA): Visualizing the distribution of 'Churn', 'PreferredLoginDevice', 'Tenure', and 'CashbackAmount' to understand their relationship with churn.
Feature Engineering and Scaling: In this stage, Categorical features (PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus) are converted to numerical representations using their index within a predefined list and then scaled. Numerical features (CustomerID, CityTier, NumberOfDeviceRegistered, SatisfactionScore, CashbackAmount, Complain, NumberOfAddress, Tenure, HourSpendOnApp, DaySinceLastOrder, CouponUsed, OrderCount, WarehouseToHome, OrderAmountHikeFromlastYear) are scaled by dividing by their respective maximum observed values or a suitable constant. Missing values are handled by assigning -1 and then scaling.
Data Splitting: The dataset is split into training (75%) and testing (25%) sets, with the training set further divided into training (80%) and validation (20%) sets. Stratified splitting is used to maintain the proportion of churned and non-churned customers.
Model Architecture: A sequential deep learning model is built with an input layer, several dense hidden layers with ReLU activation and dropout layers for regularization, and a final dense output layer with sigmoid activation for binary classification.
Addressing Imbalance: SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to create synthetic samples of the minority class (churned customers), thereby balancing the dataset.
Model Training: The model is trained using the Adam optimizer and binary cross-entropy loss. K-fold cross-validation (with k=10) is used to robustly evaluate model performance.
Model Evaluation: The trained model is evaluated on the test set using accuracy, precision, recall, and F1-score. A confusion matrix is also generated.
Model Interpretability: SHAP (SHapley Additive exPlanations) is employed to explain the output of the model. SHAP values are used to visualize global feature importance (bar plot, beeswarm plot) and individual predictions (waterfall plot, force plot).

__Citations__
This code is part of a research study on customer churn prediction in e-commerce.
lshamsi, A. (2022). Customer Churn prediction in ECommerce Sector. PhD thesis, Rochester Institute  of Technology.

__Materials & Methods__
Computing Infrastructure: The code was developed and executed in a Google Colab environment, which typically runs on a Linux operating system. The specific hardware (CPU, GPU, RAM) depends on the Colab instance allocated at runtime. The environment was set up with OpenJDK 8 for PySpark functionality.
Evaluation Method: The proposed technique is primarily evaluated using K-fold cross-validation (specifically, 10-fold cross-validation) on the training data. This method helps to estimate the model's performance more robustly by training and evaluating the model on different subsets of the data. Additionally, a final evaluation is performed on a held-out test set to assess the generalization capability of the model on unseen data. The SMOTE oversampling technique is applied only to the training folds within the cross-validation loop to prevent data leakage.
Assessment Metrics: The following assessment metrics are used to evaluate the model's performance, particularly relevant for imbalanced datasets:
Accuracy: It measures the overall correctness of the model's predictions. It is a good general metric, but can be misleading in imbalanced datasets if the majority class dominates the predictions.
Precision: It measures the proportion of correct identifications. In the context of churn, it indicates how many of the predicted churned customers actually churned.
Recall: It measures the proportion of actual positives that were correctly identified. In the context of churn, it indicates how many of the actual churned customers the model successfully identified.
F1-score: It is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall, which is particularly useful for imbalanced datasets where a high score indicates that the model has good performance on both identifying positive instances and not making too many false positive predictions.
Loss: Binary Cross-Entropy is used as the loss function, which quantifies the error between the predicted probabilities and the true labels. Monitoring training and validation loss helps in detecting overfitting or underfitting.

__Conclusions__
The dataset used in this work was for a leading e-commerce company, which was taken from Kaggle. This article started with examining and analyzing data to expand our understanding of customer churn. Where the highest correlation is between the CashbackAmount feature and the PreferedOrderCat feature. The dataset was imbalanced. The dataset was processed and balanced dataset from it to train the proposed model. It was found that the proposed model has the best accuracy when compared with the baseline. There are some limitations to this work, which are:
* __Dataset Size and Scope:__ The analysis is based on a specific e-commerce dataset, and the generalizability of the findings and the model's performance to other e-commerce platforms or different customer demographics might be limited.
* __Static Analysis:__ The current analysis is based on a static dataset.
* __SMOTE Application:__ While SMOTE effectively addresses class imbalance, it generates synthetic samples, which might not perfectly represent real-world data points and could potentially introduce noise or overfitting to the synthetic data. 
* __Interpretability of Deep Learning Models:__ Although SHAP values provide insights into feature importance, deep learning models are inherently more "black-box" compared to traditional machine learning models. A complete causal understanding of customer churn drivers based solely on model interpretability might still be challenging.
