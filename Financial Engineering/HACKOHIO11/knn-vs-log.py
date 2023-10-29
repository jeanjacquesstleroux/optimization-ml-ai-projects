#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 20:51:51 2023

@author: stleroux
"""

import pandas as pd
import functools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, auc
)
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Define a function for loading and processing the dataset
def load_and_process_data(data_file):
    # Load the dataset
    fin_data = pd.read_csv(data_file)

    # Remove rows containing -1 values
    fin_data = fin_data[~fin_data.isin([-1]).any(axis=1)]

    return fin_data

# Define a function to split the data and train a logistic regression model
def train_logistic_regression(data):
    # Define features and target
    X = data[['Open', 'Adj Close', 'High', 'Low', 'Body Size', 'Upper Shadow',
              'O/C Low Area', 'O/C High Area', 'Ratio', 'O/C > Last Year',
              'O/C < Next Year', 'Higher > Last Year?', 'High > Next Year']]
    y = data['Likely Gravestone']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a logistic regression model with specified hyperparameters
    model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000)

    # Train the model on the training data
    model.fit(X_train, y_train)

    return X_test, y_test, model

# Define a function to evaluate the logistic regression model
def evaluate_logistic_regression(X_test, y_test, model):
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Generate a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', conf_matrix)

    # Display a classification report
    class_report = classification_report(y_test, y_pred)
    print('Classification Report:\n', class_report)

    # Plot the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    # Calculate the AUC (Area Under the Curve)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# Define a function to train a K-nearest neighbors classifier with SMOTE
def train_knn_classifier_with_smote(X, y):  # Pass X and y as parameters
    # Create a KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    # Create a SMOTE object and resample the data
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the KNN classifier on the resampled data
    knn_classifier.fit(X_train, y_train)

    return X_test, y_test, knn_classifier

# Define a function to evaluate the K-nearest neighbors classifier with SMOTE
def evaluate_knn_classifier(X_test, y_test, knn_classifier):
    # Predict on the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate the precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    # Calculate the AUC (Area Under the Curve)
    pr_auc = auc(recall, precision)
    
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

# Main function to execute the code
def main(data_file):
    data = load_and_process_data(data_file)
    X_test_logistic, y_test_logistic, logistic_model = train_logistic_regression(data)
    evaluate_logistic_regression(X_test_logistic, y_test_logistic, logistic_model)
    
    X_test_knn, y_test_knn, knn_model = train_knn_classifier_with_smote(data[[
        'Open', 'Adj Close', 'High', 'Low', 'Body Size', 'Upper Shadow',
              'O/C Low Area', 'O/C High Area', 'Ratio', 'O/C > Last Year',
              'O/C < Next Year', 'Higher > Last Year?', 'High > Next Year']], 
        data['Likely Gravestone'])  # Pass X and y
    evaluate_knn_classifier(X_test_knn, y_test_knn, knn_model)

# Call the main function with the data file path
if __name__ == "__main__":
    main('/Users/stleroux/price_data_hackohio.csv')
