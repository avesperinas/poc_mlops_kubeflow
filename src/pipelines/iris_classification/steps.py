"""Pipelines' steps definition."""

from kfp.dsl import component, Output, Input, Dataset, Metrics, Model


@component(
    packages_to_install=["pandas", "scikit-learn"],
    base_image="python:3.9",
)
def prepare_data(output_data: Output[Dataset]):
    import pandas as pd
    from sklearn import datasets

    # Load dataset
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df = df.dropna()

    # Save DataFrame as CSV using the output artifact
    df.to_csv(output_data.path, index=False)

    print(f"Data saved as artifact: {output_data.path}")


@component(
    packages_to_install=["pandas", "scikit-learn"],
    base_image="python:3.9",
)
def train_test_split(
    input_data: Input[Dataset],
    x_train_output: Output[Dataset],
    x_test_output: Output[Dataset],
    y_train_output: Output[Dataset],
    y_test_output: Output[Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load data from the input artifact
    final_data = pd.read_csv(input_data.path)

    target_column = 'species'
    x = final_data.loc[:, final_data.columns != target_column]
    y = final_data[target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=47)

    # Save the datasets as artifacts
    x_train.to_csv(x_train_output.path, index=False)
    x_test.to_csv(x_test_output.path, index=False)
    pd.DataFrame(y_train).to_csv(y_train_output.path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_output.path, index=False)

    print(x_train_output.path)
    print(x_test_output.path)
    print(y_train_output.path)
    print(y_test_output.path)

    print("Train-test split completed successfully")


@component(
    packages_to_install=["pandas", "scikit-learn", "joblib"],
    base_image="python:3.9",
)
def training_basic_classifier(
    x_train_input: Input[Dataset],
    y_train_input: Input[Dataset],
    model_output: Output[Model],
    metrics_output: Output[Metrics],
):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import joblib

    # Load training data
    x_train = pd.read_csv(x_train_input.path)
    y_train = pd.read_csv(y_train_input.path).iloc[:, 0]

    # Train the model
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(x_train, y_train)

    # Save the trained model
    joblib.dump(classifier, model_output.path)

    # Log some metrics (example)
    train_score = classifier.score(x_train, y_train)
    metrics_output.log_metric("train_accuracy", train_score)

    print(f"Model trained with training accuracy: {train_score}")


@component(
    packages_to_install=["pandas", "scikit-learn"],
    base_image="python:3.9",
)
def evaluate_model(
    model: Input[Model],
    x_test: Input[Dataset],
    y_test: Input[Dataset],
    metrics: Output[Metrics],
):
    import joblib
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.metrics import classification_report, roc_curve, auc
    import os
    
    # Load the model and data
    clf = joblib.load(model.path)
    
    x_test_df = pd.read_csv(x_test.path)
    y_test_df = pd.read_csv(y_test.path)
    y_true = y_test_df.iloc[:, 0].values
    
    # Make predictions and calculate metrics
    y_pred = clf.predict(x_test_df)
    y_pred_proba = clf.predict_proba(x_test_df)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Save metrics to output
    metrics_dict = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    for metric, value in metrics_dict.items():
        metrics.log_metric(metric, value) 

    print(f"Model evaluation completed successfully with accuracy: {accuracy:.4f}")
