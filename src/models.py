from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score ,f1_score
import numpy as np

def make_model(model):
    if model == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif model == 'RandomForest':
        model = RandomForestClassifier()
    else:
        model = GaussianNB()

    return model

def evaluate_model(model, X, y, cv=5):
    accuracy_score = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precision_score = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall_score = cross_val_score(model, X, y, cv=cv, scoring='recall')
    f1_score = cross_val_score(model, X, y, cv=cv, scoring='f1')

    evaluation_results = {
        'accuracy': np.mean(accuracy_score),
        'precision': np.mean(precision_score),
        'recall': np.mean(recall_score),
        'f1': np.mean(f1_score)
    }

    print("Evaluation results (Mean ± Std)")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.2f} ± {np.std(value):.2f}")
    
    return evaluation_results
