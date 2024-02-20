Credit Card Fraud Detection

This repository contains code and resources for credit card fraud detection. It includes various stages of data processing, analysis, and machine learning model implementation aimed at detecting fraudulent transactions in credit card data.

1. Introduction
Credit card fraud poses a significant threat to financial institutions and individuals alike. This project aims to address this issue by implementing machine learning algorithms to identify fraudulent transactions based on historical credit card data.

2. Dataset Overview
   
2.1 Data Loading and Exploration
In this section, we load and examine the dataset used for the project. The dataset, named "Banksim.csv," is loaded using the pandas library in Python. We first display the first 5 rows of the dataset to get a quick overview of its contents, followed by using the info() function to obtain general information about the dataset, such as column types, null values, and memory usage.

3. Fraud and Non-Fraud Data Comparison
   
3.1 Creation of Fraud and Non-Fraud Dataframes
This stage involves creating separate dataframes for fraudulent and non-fraudulent transactions. The purpose is to analyze the differences between these two datasets and identify factors influencing fraudulent activities.

3.2 Comparison of Fraudulent and Non-Fraudulent Transactions
Using visualization techniques from the Seaborn library, we compare fraudulent and non-fraudulent transactions to identify patterns and differences between the two categories. This includes visualizing the frequency of fraudulent transactions, distribution across different categories, and amounts involved.

4. Data Transformation and Feature Engineering
   
In this step, we preprocess the data to facilitate machine learning model implementation. Categorical columns are transformed, and independent (X) and dependent/target (y) variables are defined.

5. Balancing the Dataset
   
Given the significant class imbalance between fraudulent and non-fraudulent transactions, techniques such as SMOTE (Synthetic Minority Over-sampling Technique) are applied to balance the dataset and mitigate the impact of imbalanced classes.

6. Applying Machine Learning Algorithms
   
Various machine learning algorithms, including Artificial Neural Network, Gaussian Naive Bayes, K-Nearest Neighbors (KNN), and XGBoost, are applied to the balanced dataset. Performance metrics such as accuracy, confusion matrix, and classification report are evaluated using both training and test datasets.

7. Conclusion
   
This project demonstrates the application of machine learning techniques for credit card fraud detection and provides insights into the factors influencing fraudulent transactions.
