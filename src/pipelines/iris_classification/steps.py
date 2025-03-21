"""Pipelines' steps definition."""

from kfp.dsl import component, Output, Input, Dataset, Metrics, Model, Artifact, ClassificationMetrics

@component(
    packages_to_install=["pandas", "scikit-learn", "tensorflow"],
    base_image="python:3.9",
)
def prepare_tensorboard_logs(
    tensorboard_log: Output[Artifact]
):
    import os
    import tensorflow as tf
    
    os.makedirs(tensorboard_log.path, exist_ok=True)
    print(f"TensorBoard log directory created at: {tensorboard_log.path}")


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
    metrics_output: Output[Metrics],
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
    
    # Log dataset metrics
    metrics_output.log_metric("total_samples", len(final_data))
    metrics_output.log_metric("train_samples", len(x_train))
    metrics_output.log_metric("test_samples", len(x_test))
    
    class_distribution = y.value_counts().to_dict()
    for class_label, count in class_distribution.items():
        metrics_output.log_metric(f"class_{class_label}_count", count)

    print("Train-test split completed successfully")


@component(
    packages_to_install=["pandas", "scikit-learn", "joblib", "tensorflow", "matplotlib"],
    base_image="python:3.9",
)
def training_basic_classifier(
    x_train_input: Input[Dataset],
    y_train_input: Input[Dataset],
    tensorboard_log: Input[Artifact],
    model_output: Output[Model],
    metrics_output: Output[Metrics],
):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import joblib
    import os
    import tensorflow as tf
    from datetime import datetime
    import matplotlib.pyplot as plt
    
    # Configure TensorBoard
    try:
        log_dir = os.path.join(tensorboard_log.path, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        file_writer = tf.summary.create_file_writer(log_dir)
    except:
        pass
    
    # Load training data
    x_train = pd.read_csv(x_train_input.path)
    y_train = pd.read_csv(y_train_input.path).iloc[:, 0]

    # Train the model with multiple iterations to track progress
    classifier = LogisticRegression(max_iter=500, solver='saga', C=1.0)
    
    for i in range(1, 11):
        subset_size = int(len(x_train) * (i/10))
        subset_x = x_train.iloc[:subset_size]
        subset_y = y_train.iloc[:subset_size]
        
        classifier.fit(subset_x, subset_y)
        train_score = classifier.score(subset_x, subset_y)
        
        # Registrate on TensorBoard
        try:
            with file_writer.as_default():
                tf.summary.scalar('training_accuracy', train_score, step=i)
                tf.summary.scalar('training_loss', 1.0 - train_score, step=i)
                tf.summary.scalar('learning_rate', 0.1 / i, step=i)
                
                # Corrección para guardar la figura en TensorBoard
                feature_importance = np.abs(classifier.coef_[0])
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                ax.bar(x_train.columns, feature_importance)
                ax.set_title('Feature Importance')
                plt.tight_layout()
                
                # Convertir figura a imagen para TensorBoard
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Crear imagen para TensorBoard
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)  # Añadir dimensión de batch
                
                tf.summary.image('feature_importance', image, step=i)
                plt.close(fig)
        except:
            pass
    
    classifier.fit(x_train, y_train)
    
    # Save the trained model
    joblib.dump(classifier, model_output.path)

    # Log metrics
    train_score = classifier.score(x_train, y_train)
    metrics_output.log_metric("train_accuracy", train_score)
    metrics_output.log_metric("model_converged", int(classifier.n_iter_[0] < 500))
    metrics_output.log_metric("num_iterations", int(classifier.n_iter_[0]))
    
    # Feature importance
    for i, feature in enumerate(x_train.columns):
        for j, target in enumerate(np.unique(y_train)):
            metrics_output.log_metric(f"coef_{feature}_class_{target}", float(classifier.coef_[j, i]))

    print(f"Model trained with training accuracy: {train_score}")
    print(f"TensorBoard logs saved to: {log_dir}")


@component(
    packages_to_install=["pandas", "scikit-learn", "matplotlib", "tensorflow"],
    base_image="python:3.9",
)
def evaluate_model(
    model: Input[Model],
    x_test: Input[Dataset],
    y_test: Input[Dataset],
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    tensorboard_log: Input[Artifact]
):
    import joblib
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from datetime import datetime
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
        classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
    )

    # Configure TensorBoard
    log_dir = os.path.join(tensorboard_log.path, f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    file_writer = tf.summary.create_file_writer(log_dir)

    # Load the model and data
    clf = joblib.load(model.path)

    x_test_df = pd.read_csv(x_test.path)
    y_test_df = pd.read_csv(y_test.path)
    y_true = y_test_df.iloc[:, 0].values

    # Make predictions
    y_pred = clf.predict(x_test_df)
    y_pred_proba = clf.predict_proba(x_test_df)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Log metrics to Kubeflow
    metrics_dict = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    # Log per-class metrics
    try:
        classes = np.unique(y_true)
        for cls in classes:
            cls_precision = precision_score(y_true, y_pred, labels=[cls], average=None)[0]
            cls_recall = recall_score(y_true, y_pred, labels=[cls], average=None)[0]
            cls_f1 = f1_score(y_true, y_pred, labels=[cls], average=None)[0]
            
            metrics_dict[f"class_{cls}_precision"] = float(cls_precision)
            metrics_dict[f"class_{cls}_recall"] = float(cls_recall)
            metrics_dict[f"class_{cls}_f1"] = float(cls_f1)
    except:
        pass

    # Add all metrics to Kubeflow
    for metric, value in metrics_dict.items():
        metrics.log_metric(metric, value)

    # Generate confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred).tolist()
        classification_metrics.log_confusion_matrix(
            categories=list(map(str, classes)),
            matrix=cm
        )
    
    except:
        pass
        
    print(f"Model evaluation completed successfully with accuracy: {accuracy:.4f}")
    print(f"Evaluation metrics logged to TensorBoard at: {log_dir}")
    print(f"Classification report:\n{classification_report(y_true, y_pred)}")