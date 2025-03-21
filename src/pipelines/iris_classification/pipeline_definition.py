"""Pipeline's definition."""

import os
from pathlib import Path

import kfp
from kfp.compiler import Compiler
from kfp.dsl import pipeline

from src.utils import credentials
from src.pipelines.iris_classification.steps import (
    prepare_tensorboard_logs,
    prepare_data,
    train_test_split,
    training_basic_classifier,
    evaluate_model,
    prepare_tensorboard_logs,
)


pipeline_name = "iris-pipeline-tensorboard"
current_module_path = Path(__file__).relative_to(Path.cwd()).parent
pipeline_definition_path = str(current_module_path / f"{pipeline_name}.yaml")


# Define the pipeline
@pipeline(
    name=pipeline_name,
    description="A simple pipeline to classify Iris flowers with TensorBoard integration.",
)
def iris_pipeline():
    
    tensorboard_logs_task = prepare_tensorboard_logs()
    
    prepare_data_task = prepare_data()

    train_test_split_task = train_test_split(
        input_data=prepare_data_task.outputs["output_data"]
    )

    training_basic_classifier_task = training_basic_classifier(
        x_train_input=train_test_split_task.outputs["x_train_output"],
        y_train_input=train_test_split_task.outputs["y_train_output"],
        tensorboard_log=tensorboard_logs_task.outputs["tensorboard_log"]
    )

    evaluate_model_task = evaluate_model(
        model=training_basic_classifier_task.outputs["model_output"],
        x_test=train_test_split_task.outputs["x_test_output"],
        y_test=train_test_split_task.outputs["y_test_output"],
        tensorboard_log=tensorboard_logs_task.outputs["tensorboard_log"]
    )
    
    # evaluate_model_task.add_pod_annotation(
    #     name="kubeflow.org/tensorboard-log", 
    #     value=tensorboard_logs_task.outputs["tensorboard_log"].path
    # )

# Compile the pipeline
Compiler().compile(
    pipeline_func=iris_pipeline,
    package_path=pipeline_definition_path,
)

# Connect with the client
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

run_details = kfp_client.get_run(run.run_id)
print(f"Pipeline run ID: {run.run_id}")
print(f"Experiment name: iris-classification-experiment")
# print(f"Run status: {run_details.run.status}")
print(f"tensorboard --logdir={os.path.join(os.environ.get('MINIO_HOST', ''), 'tensorboard-logs', run.run_id)}")
