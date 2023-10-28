import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    # Load the dataset
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Dropping all rows containing a -1 value
    return data[~data.isin([-1]).any(axis=1)]

def split_data(data):
    # Define features and target
    X = data[['Open', 'Adj Close', 'High', 'Low', 'Body Size', 'Upper Shadow',
              'O/C Low Area', 'O/C High Area', 'Ratio', 'O/C > Last Year',
              'O/C < Next Year', 'Higher > Last Year?', 'High > Next Year']]
    y = data['Likely Gravestone']

    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_logistic_regression(X_train, y_train):
    # C: Smaller values indicate stronger regularization
    # penalty: L2 regularization helps prevent overfitting
    # 'lbfgs': quasi-Newton method works well with L2 / small/med dataset 
    model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
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
    plot_roc_curve(fpr, tpr, roc_auc)

    # Plot the precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])

    # Calculate the area under the precision-recall curve
    pr_auc = auc(recall, precision)
    plot_precision_recall_curve(recall, precision, pr_auc)

def plot_roc_curve(fpr, tpr, roc_auc):
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

def plot_precision_recall_curve(recall, precision, pr_auc):
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

if __name__ == '__main__':
    data = load_data('/Users/stleroux/price_data_hackohio.csv')
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_logistic_regression(X_train, y_train)
    evaluate_model(model, X_test, y_test)
