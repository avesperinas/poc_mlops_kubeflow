"""Pipeline's definition."""

import os
from pathlib import Path

import kfp
from kfp.compiler import Compiler
from kfp.dsl import pipeline

from src.utils import credentials
from src.pipelines.iris_classification.steps import (
    prepare_data,
    train_test_split,
    training_basic_classifier,
    evaluate_model,
)


pipeline_name = "iris-pipeline"
current_module_path = Path(__file__).relative_to(Path.cwd()).parent
pipeline_definition_path = str(current_module_path / f"{pipeline_name}.yaml")


# Define the pipeline
@pipeline(
    name=pipeline_name,
    description="A simple pipeline to classify Iris flowers.",
)
def iris_pipeline():
    
    prepare_data_task = prepare_data()

    train_test_split_task = train_test_split(
        input_data=prepare_data_task.outputs["output_data"]
    )

    training_basic_classifier_task = training_basic_classifier(
        x_train_input=train_test_split_task.outputs["x_train_output"],
        y_train_input=train_test_split_task.outputs["y_train_output"],
    )
    
    evaluate_model(
        model=training_basic_classifier_task.outputs["model_output"],
        x_test=train_test_split_task.outputs["x_test_output"],
        y_test=train_test_split_task.outputs["y_test_output"],
    )

# Compile the pipeline
os.makedirs('pipelines/iris_pipeline', exist_ok=True)
Compiler().compile(
    pipeline_func=iris_pipeline,
    package_path=pipeline_definition_path,
)

# Coonect with the client
kfp_client = kfp.Client(
    host=os.environ.get("PIPELINES_HOST"),
    verify_ssl=not credentials.skip_tls_verify,
    credentials=credentials,
)

# Run the pipeline
run = kfp_client.create_run_from_pipeline_package(
    pipeline_definition_path,
    experiment_name='iris-classification-experiment',
    enable_caching=False,
)
