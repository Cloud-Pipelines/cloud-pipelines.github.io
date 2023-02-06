# Interactive pipelines

Cloud Pipelines SDK has an interactive mode feature that allows the users to run components one by one and build pipelines interactively, step by step.

Usually, pipelines are built statically, then compiled and sent for execution.
However, for experimentation and pipeline or component development, running components interactively provides better experience.

Interactive mode changes the behavior of the component functions. Normally, when you call a component function, it simply returns a `Task` object which is used to build pipelines statically. However, in interactive mode, when you call a component function, the component starts executing immediately and the function returns an `Execution` object instead of a `Task` object.

## Prerequisites

Install the [Cloud Pipelines SDK](https://pypi.org/project/cloud-pipelines/): `pip install cloud-pipelines`.

The default task launcher uses the local Docker installation to execute component tasks.
Make sure that Docker is installed and working. See <https://docs.docker.com/get-docker/>

## Activating the interactive mode

To activate the interactive execution mode, call `cloud_pipelines.activate_interactive_mode()`:

```python
import cloud_pipelines

cloud_pipelines.activate_interactive_mode()
```

That's it.
Now you can start executing components. You can pass the component execution outputs to other components, thus creating pipelines interactively.

Note: To deactivate interactive mode, call `cloud_pipelines.deactivate_interactive_mode()`.
Note: You can use interactive mode as Python context: `cloud_pipelines.orchestration.runners.InteractiveMode()`.

## Using interactive mode

### Creating or loading a component

Components are usually loaded from remote or local `component.yaml` files.

Methods to load or create components can be found in the `cloud_pipelines.components` module:

* `load_component_from_url`
* `load_component_from_file`
* `load_component_from_text`
* `create_component_from_func`

Each of these functions return a Python function with signature that corresponds to the component inputs.

Here we will just create a new component from a small Python function.

```python
def add_exclamation(message: str = "Hello, world") -> str:
    result = message + "!"
    print(result)
    return result

from cloud_pipelines import components

add_exclamation_op = components.create_component_from_func(add_exclamation)
```

The resulting `add_exclamation_op` is a Python callable/function.

Learn more about [creating a component from a Python function](https://cloud-pipelines.net/components/create_component_from_python_function/).

### Running the component interactively

Usually the component function just returns a `Task` object that is used for building pipelines statically. However in interactive mode, the component starts running right away and the function returns an `Execution` object.

```python
execution = add_exclamation_op(message="Hi")
```

You should see the following log:

```text
2023-02-06 07:18:37.984659: [Add exclamation] Starting container task.
2023-02-06 07:18:40.474181: [Add exclamation] Hi!
2023-02-06 07:18:40.998756: [Add exclamation] Container task completed with status: Succeeded
```

Check the component execution outputs (`Artifact` objects):

```python
output1 = execution.outputs["Output"]
print(output1.materialize())
```

You should see the following log:

```text
Hi!
```

## Orchestration

### Executions

When you run component task, the `Runner` immediately creates and returns an `Execution` object while the execution runs in the background.
`Execution` object has `.outputs` attribute that holds the dictionary of the component execution's output artifacts. The output artifacts can be passed to components which creates new executions.

The `execution.wait_for_completion()` method blocks until the execution succeeds or fails.

### Artifacts

Each `Artifact` points to already existing or future data (file or directory).

The `artifact.download([path=...])` method downloads artifact to explicit or temporary local location and returns the path.

The `artifact.type_spec` attribute holds the artifact type.

The `artifact.materialize()` method can convert the artifact data to a Python object for a small list of known types:

* `String` -> `str`
* `Integer` -> `int`
* `Float` -> `float`
* `Boolean` -> `bool`
* `JsonArray` -> `list`
* `JsonObject` -> `dict`

Extra supported types:

* `ApacheParquet` -> `pandas.DataFrame`
* `TensorflowSavedModel` -> TensorFlow or Keras module

The same types are supported when passing Python objects into a component.

Note: The `artifact.download()` and `artifact.materialize()` methods wait for the artifacts to be produced if they are not ready yet.

### Features

#### Execution caching and reuse

Successfully completed executions are put in cache.
If the same task (same components, same input arguments) is submitted for execution again, the result would be reused from cache instead of doing the work again.

#### Running execution caching and reuse

Executions that are still running are put in a special cache too.
If some task is submitted for execution while an identical task (same components, same input arguments) is being executed, then this running execution will be reused instead of doing the work again.

## Running locally and in cloud

### Local launchers

#### Docker

Docker launcher uses the local Docker installation to execute component tasks.
Make sure that Docker is installed and working. See <https://docs.docker.com/get-docker/>

This is the default task launcher, so you do not need to select it explicitly.
To configure Cloud Pipelines to use the Docker task launcher use the following code:

```python
from cloud_pipelines.orchestration.launchers.local_docker_launcher import DockerContainerLauncher
cloud_pipelines.activate_interactive_mode(
    task_launcher=DockerContainerLauncher(),
    # Optional:
    root_uri="/some/local/path",  # The location where the system can put the artifact data and the DB
)
```

### Google Cloud

There are several task launchers based on Google Cloud services.
Launchers are being actively developed and some are still in experimental stage.

Make sure that you have set up access to Google Cloud using `gcloud` (e.g. `gcloud auth login`).

Google Cloud services work with data stored in Google Cloud Storage, so you will need [Google Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets).

Set the `root_uri` to some directory in a Google Cloud Storage bucket when using task launchers based on Google Cloud services.

#### Google Cloud Batch

[Google Cloud Batch](https://cloud.google.com/batch/) is a fully managed batch service to schedule, queue, and execute batch jobs on Google's infrastructure.

To configure Cloud Pipelines to use the Google Cloud Batch task launcher use the following code:

```python
from cloud_pipelines.orchestration.launchers.google_cloud_batch_launcher import GoogleCloudBatchLauncher
cloud_pipelines.activate_interactive_mode(
    task_launcher=GoogleCloudBatchLauncher(),
    root_uri="gs://<bucket>/<root_dir>",
)
```

#### Google Cloud Vertex AI CustomJob

[Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/) helps users build, deploy, and scale machine learning (ML) models faster, with fully managed ML tools for any use case. Google Cloud Vertex AI [CustomJob](https://cloud.google.com/vertex-ai/docs/training/create-custom-job) service is the basic way to run containerized code in Vertex AI.

To configure Cloud Pipelines to use the Google Cloud Vertex AI CustomJob task launcher use the following code:

```python
from cloud_pipelines.orchestration.launchers.google_cloud_vertex_custom_job_launcher import GoogleCloudVertexAiCustomJobLauncher

cloud_pipelines.activate_interactive_mode(
    task_launcher=GoogleCloudVertexAiCustomJobLauncher(),
    root_uri="gs://<bucket>/<root_dir>",
)
```


## Examples

### Passing and materializing Pandas DataFrame

```python
import cloud_pipelines
from cloud_pipelines.components import create_component_from_func, InputPath, OutputPath

cloud_pipelines.activate_interactive_mode()


# Creating a new component
def select_columns(
    table_path: InputPath("ApacheParquet"),
    output_table_path: OutputPath("ApacheParquet"),
    column_names: list,
):
    import pandas

    df = pandas.read_parquet(table_path)
    print("Input table:")
    df.info()
    df = df[column_names]
    print("Output table:")
    df.info()
    df.to_parquet(output_table_path, index=False)

select_columns_op = create_component_from_func(
    func=select_columns,
    packages_to_install=["pandas==1.3.5", "pyarrow==10.0.1"],
)

# Using the component:

# Preparing the input data
import pandas
input_df = pandas.DataFrame(
    {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature3": ["a", "b", "c", "d", "e"],
    }
)

# Running the component
output_art = select_columns_op(
    table=input_df, column_names=["feature1", "feature2"]
).outputs["output_table"]

output_art2 = select_columns_op(
    table=output_art, column_names=["feature2"]
).outputs["output_table"]

# Checking the output data
output_df = output_art.materialize()
assert output_df.columns.to_list() == ["feature1", "feature2"]

output_df2 = output_art2.materialize()
assert output_df2.columns.to_list() == ["feature2"]

cloud_pipelines.deactivate_interactive_mode()
```

You should see the following log:

```text
2023-02-06 07:20:59.575396: [Select columns] Starting container task.
2023-02-06 07:21:18.629752: [Select columns] Input table:
2023-02-06 07:21:18.638000: [Select columns] <class 'pandas.core.frame.DataFrame'>
2023-02-06 07:21:18.639234: [Select columns] RangeIndex: 5 entries, 0 to 4
2023-02-06 07:21:18.640105: [Select columns] Data columns (total 3 columns):
2023-02-06 07:21:18.640904: [Select columns]  #   Column    Non-Null Count  Dtype
2023-02-06 07:21:18.641821: [Select columns] ---  ------    --------------  -----
2023-02-06 07:21:18.642746: [Select columns]  0   feature1  5 non-null      int64
2023-02-06 07:21:18.643615: [Select columns]  1   feature2  5 non-null      float64
2023-02-06 07:21:18.644713: [Select columns]  2   feature3  5 non-null      object
2023-02-06 07:21:18.645470: [Select columns] dtypes: float64(1), int64(1), object(1)
2023-02-06 07:21:18.646253: [Select columns] memory usage: 248.0+ bytes
2023-02-06 07:21:18.647397: [Select columns] Output table:
2023-02-06 07:21:18.648472: [Select columns] <class 'pandas.core.frame.DataFrame'>
2023-02-06 07:21:18.649203: [Select columns] RangeIndex: 5 entries, 0 to 4
2023-02-06 07:21:18.650094: [Select columns] Data columns (total 2 columns):
2023-02-06 07:21:18.650899: [Select columns]  #   Column    Non-Null Count  Dtype
2023-02-06 07:21:18.651725: [Select columns] ---  ------    --------------  -----
2023-02-06 07:21:18.652832: [Select columns]  0   feature1  5 non-null      int64
2023-02-06 07:21:18.653717: [Select columns]  1   feature2  5 non-null      float64
2023-02-06 07:21:18.654761: [Select columns] dtypes: float64(1), int64(1)
2023-02-06 07:21:18.655600: [Select columns] memory usage: 208.0 bytes
2023-02-06 07:21:21.214023: [Select columns] Container task completed with status: Succeeded
2023-02-06 07:21:21.219940: [Select columns 2] Starting container task.
2023-02-06 07:21:37.051625: [Select columns 2] Input table:
2023-02-06 07:21:37.059715: [Select columns 2] <class 'pandas.core.frame.DataFrame'>
2023-02-06 07:21:37.061031: [Select columns 2] RangeIndex: 5 entries, 0 to 4
2023-02-06 07:21:37.062099: [Select columns 2] Data columns (total 2 columns):
2023-02-06 07:21:37.063600: [Select columns 2]  #   Column    Non-Null Count  Dtype
2023-02-06 07:21:37.064602: [Select columns 2] ---  ------    --------------  -----
2023-02-06 07:21:37.065859: [Select columns 2]  0   feature1  5 non-null      int64
2023-02-06 07:21:37.066878: [Select columns 2]  1   feature2  5 non-null      float64
2023-02-06 07:21:37.068062: [Select columns 2] dtypes: float64(1), int64(1)
2023-02-06 07:21:37.069451: [Select columns 2] memory usage: 208.0 bytes
2023-02-06 07:21:37.070592: [Select columns 2] Output table:
2023-02-06 07:21:37.071812: [Select columns 2] <class 'pandas.core.frame.DataFrame'>
2023-02-06 07:21:37.072785: [Select columns 2] RangeIndex: 5 entries, 0 to 4
2023-02-06 07:21:37.073734: [Select columns 2] Data columns (total 1 columns):
2023-02-06 07:21:37.074751: [Select columns 2]  #   Column    Non-Null Count  Dtype
2023-02-06 07:21:37.075664: [Select columns 2] ---  ------    --------------  -----
2023-02-06 07:21:37.077513: [Select columns 2]  0   feature2  5 non-null      float64
2023-02-06 07:21:37.078540: [Select columns 2] dtypes: float64(1)
2023-02-06 07:21:37.079618: [Select columns 2] memory usage: 168.0 bytes
2023-02-06 07:21:39.645323: [Select columns 2] Container task completed with status: Succeeded
```

### Passing and materializing TensorFlow model

```python
import cloud_pipelines
from cloud_pipelines.components import create_component_from_func, InputPath, OutputPath

cloud_pipelines.activate_interactive_mode()


# Creating a new component
def transform_keras_model(
    model_path: InputPath("TensorflowSavedModel"),
    output_model_path: OutputPath("TensorflowSavedModel"),
):
    import tensorflow

    model = tensorflow.keras.models.load_model(filepath=model_path)
    model.summary()
    tensorflow.keras.models.save_model(model=model, filepath=output_model_path)

transform_keras_model_op = create_component_from_func(
    func=transform_keras_model,
    base_image="tensorflow/tensorflow:2.11.0",
)

# Using the component:

# Preparing the input data
import tensorflow as tf
input_model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(3,)),
    tf.keras.layers.Softmax(),
])

# Running the component
output_art = transform_keras_model_op(model=input_model).outputs["output_model"]

# Checking the output data
output_model = output_art.materialize()
predictions = output_model(tf.constant([[0.1, 0.2, 0.3]]))
print(predictions)
assert predictions.shape == (1, 5)

cloud_pipelines.deactivate_interactive_mode()
```

You should see the following log:

```text
2023-02-06 07:28:23.274859: [Transform keras model] Starting container task.
2023-02-06 07:28:26.564806: [Transform keras model] Model: "sequential_2"
2023-02-06 07:28:26.566027: [Transform keras model] _________________________________________________________________
2023-02-06 07:28:26.567347: [Transform keras model]  Layer (type)                Output Shape              Param #
2023-02-06 07:28:26.568419: [Transform keras model] =================================================================
2023-02-06 07:28:26.569426: [Transform keras model]  dense_2 (Dense)             (None, 5)                 20
2023-02-06 07:28:26.570448: [Transform keras model]
2023-02-06 07:28:26.571461: [Transform keras model]  softmax_2 (Softmax)         (None, 5)                 0
2023-02-06 07:28:26.572642: [Transform keras model]
2023-02-06 07:28:26.575428: [Transform keras model] =================================================================
2023-02-06 07:28:26.576851: [Transform keras model] Total params: 20
2023-02-06 07:28:26.578012: [Transform keras model] Trainable params: 20
2023-02-06 07:28:26.579110: [Transform keras model] Non-trainable params: 0
2023-02-06 07:28:26.580276: [Transform keras model] _________________________________________________________________
2023-02-06 07:28:27.912004: [Transform keras model] Container task completed with status: Succeeded
```

```text
tf.Tensor([[0.17777765 0.16397926 0.18161307 0.22165933 0.2549707 ]], shape=(1, 5), dtype=float32)
```

### End-to-end model training pipeline

```python
import cloud_pipelines
from cloud_pipelines import components

cloud_pipelines.activate_interactive_mode()


# Loading components:
download_from_gcs_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/d8c4cf5e6403bc65bcf8d606e6baf87e2528a3dc/components/google-cloud/storage/download/component.yaml")
select_columns_using_Pandas_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/8c78aae096806cff3bc331a40566f42f5c3e9d4b/components/pandas/Select_columns/in_CSV_format/component.yaml")
fill_all_missing_values_using_Pandas_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/23405971f5f16a41b16c343129b893c52e4d1d48/components/pandas/Fill_all_missing_values/in_CSV_format/component.yaml")
split_rows_into_subsets_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/daae5a4abaa35e44501818b1534ed7827d7da073/components/dataset_manipulation/Split_rows_into_subsets/in_CSV/component.yaml")
create_fully_connected_tensorflow_network_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/9ca0f9eecf5f896f65b8538bbd809747052617d1/components/tensorflow/Create_fully_connected_network/component.yaml")
train_model_using_Keras_on_CSV_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/c504a4010348c50eaaf6d4337586ccc008f4dcef/components/tensorflow/Train_model_using_Keras/on_CSV/component.yaml")
predict_with_TensorFlow_model_on_CSV_data_op = components.load_component_from_url("https://raw.githubusercontent.com/Ark-kun/pipeline_components/59c759ce6f543184e30db6817d2a703879bc0f39/components/tensorflow/Predict/on_CSV/component.yaml")

# The end-to-end model training pipeline

dataset_gcs_uri = "gs://ml-pipeline-dataset/Chicago_taxi_trips/chicago_taxi_trips_2019-01-01_-_2019-02-01_limit=10000.csv"
feature_columns = ["trip_seconds", "trip_miles", "pickup_community_area", "dropoff_community_area", "fare", "tolls", "extras"]  # Excluded "trip_total"
label_column = "tips"
training_set_fraction = 0.8

all_columns = [label_column] + feature_columns

dataset = download_from_gcs_op(
    gcs_path=dataset_gcs_uri
).outputs["Data"]

dataset = select_columns_using_Pandas_on_CSV_data_op(
    table=dataset,
    column_names=all_columns,
).outputs["transformed_table"]

dataset = fill_all_missing_values_using_Pandas_on_CSV_data_op(
    table=dataset,
    replacement_value="0",
    # # Optional:
    # column_names=None,  # =[...]
).outputs["transformed_table"]

split_task = split_rows_into_subsets_op(
    table=dataset,
    fraction_1=training_set_fraction,
)
training_data = split_task.outputs["split_1"]
testing_data = split_task.outputs["split_2"]

network = create_fully_connected_tensorflow_network_op(
    input_size=len(feature_columns),
    # Optional:
    hidden_layer_sizes=[10],
    activation_name="elu",
    # output_activation_name=None,
    # output_size=1,
).outputs["model"]

model = train_model_using_Keras_on_CSV_op(
    training_data=training_data,
    model=network,
    label_column_name=label_column,
    # Optional:
    # loss_function_name="mean_squared_error",
    number_of_epochs=10,
    # learning_rate=0.1,
    # optimizer_name="Adadelta",
    # optimizer_parameters={},
    # batch_size=32,
    metric_names=["mean_absolute_error"],
    # random_seed=0,
).outputs["trained_model"]

predictions = predict_with_TensorFlow_model_on_CSV_data_op(
    dataset=testing_data,
    model=model,
    # label_column_name needs to be set when doing prediction on a dataset that has labels
    label_column_name=label_column,
    # Optional:
    # batch_size=1000,
).outputs["predictions"]

# Inspecting the trained model (requires TensorFlow to be installed):
tf_model = model.materialize()
predictions = tf_model.predict([[100, 10, 0, 0, 30, 0, 0]])
print(predictions)
```

You should see the following log:

```text
2023-02-06 07:50:05.939270: [Download from GCS] Starting container task.
2023-02-06 07:50:05.958715: [Create fully connected tensorflow network] Starting container task.
2023-02-06 07:50:27.954952: [Download from GCS] Copying gs://ml-pipeline-dataset/Chicago_taxi_trips/chicago_taxi_trips_2019-01-01_-_2019-02-01_limit=10000.csv...
/ [1/1 files][  4.0 MiB/  4.0 MiB] 100% DoneCS] / [0/1 files][    0.0 B/  4.0 MiB]   0% Done
2023-02-06 07:50:29.098814: [Download from GCS] Container task completed with status: Succeeded
2023-02-06 07:50:29.105666: [Select columns using Pandas on CSV data] Starting container task.
2023-02-06 07:50:31.304809: [Create fully connected tensorflow network] Model: "sequential"
2023-02-06 07:50:31.306510: [Create fully connected tensorflow network] _________________________________________________________________
2023-02-06 07:50:31.308110: [Create fully connected tensorflow network]  Layer (type)                Output Shape              Param #
2023-02-06 07:50:31.309528: [Create fully connected tensorflow network] =================================================================
2023-02-06 07:50:31.310855: [Create fully connected tensorflow network]  dense (Dense)               (None, 10)                80
2023-02-06 07:50:31.312091: [Create fully connected tensorflow network]
2023-02-06 07:50:31.313317: [Create fully connected tensorflow network]  dense_1 (Dense)             (None, 1)                 11
2023-02-06 07:50:31.314453: [Create fully connected tensorflow network]
2023-02-06 07:50:31.315585: [Create fully connected tensorflow network] =================================================================
2023-02-06 07:50:31.316750: [Create fully connected tensorflow network] Total params: 91
2023-02-06 07:50:31.318417: [Create fully connected tensorflow network] Trainable params: 91
2023-02-06 07:50:31.319557: [Create fully connected tensorflow network] Non-trainable params: 0
2023-02-06 07:50:31.320711: [Create fully connected tensorflow network] _________________________________________________________________
2023-02-06 07:50:31.321628: [Create fully connected tensorflow network] None
2023-02-06 07:50:32.675023: [Create fully connected tensorflow network] Container task completed with status: Succeeded
2023-02-06 07:50:48.864645: [Select columns using Pandas on CSV data] Container task completed with status: Succeeded
2023-02-06 07:50:48.871253: [Fill all missing values using Pandas on CSV data] Starting container task.
2023-02-06 07:51:04.675086: [Fill all missing values using Pandas on CSV data] Container task completed with status: Succeeded
2023-02-06 07:51:04.682334: [Split rows into subsets] Starting container task.
2023-02-06 07:51:05.849733: [Split rows into subsets] Container task completed with status: Succeeded
2023-02-06 07:51:05.859774: [Train model using Keras on CSV] Starting container task.
2023-02-06 07:51:30.908289: [Train model using Keras on CSV] Epoch 1/10
250/250 [==============================] - 1s 3ms/step - loss: 466542.4375 - mean_absolute_error: 255.4936
2023-02-06 07:51:32.240018: [Train model using Keras on CSV] Epoch 2/10
250/250 [==============================] - 0s 2ms/step - loss: 438178.6250 - mean_absolute_error: 244.0726
2023-02-06 07:51:32.765705: [Train model using Keras on CSV] Epoch 3/10
250/250 [==============================] - 0s 2ms/step - loss: 420882.3125 - mean_absolute_error: 236.5558
2023-02-06 07:51:33.282513: [Train model using Keras on CSV] Epoch 4/10
250/250 [==============================] - 0s 2ms/step - loss: 400669.0625 - mean_absolute_error: 228.6952
2023-02-06 07:51:33.818343: [Train model using Keras on CSV] Epoch 5/10
250/250 [==============================] - 0s 1ms/step - loss: 389529.3125 - mean_absolute_error: 221.8236
2023-02-06 07:51:34.364360: [Train model using Keras on CSV] Epoch 6/10
250/250 [==============================] - 0s 2ms/step - loss: 369576.1875 - mean_absolute_error: 214.7483
2023-02-06 07:51:34.902718: [Train model using Keras on CSV] Epoch 7/10
250/250 [==============================] - 0s 1ms/step - loss: 359488.5938 - mean_absolute_error: 208.7065
2023-02-06 07:51:35.421358: [Train model using Keras on CSV] Epoch 8/10
250/250 [==============================] - 0s 1ms/step - loss: 344714.1250 - mean_absolute_error: 202.4835
2023-02-06 07:51:36.059533: [Train model using Keras on CSV] Epoch 9/10
250/250 [==============================] - 0s 2ms/step - loss: 332770.8438 - mean_absolute_error: 196.7609
2023-02-06 07:51:36.600948: [Train model using Keras on CSV] Epoch 10/10
250/250 [==============================] - 0s 1ms/step - loss: 320756.7500 - mean_absolute_error: 191.5481
2023-02-06 07:51:38.626267: [Train model using Keras on CSV] Container task completed with status: Succeeded
2023-02-06 07:51:38.635764: [Predict with TensorFlow model on CSV data] Starting container task.
2023-02-06 07:52:05.156470: [Predict with TensorFlow model on CSV data] Container task completed with status: Succeeded

1/1 [==============================] - 2s 2s/step
[[19.48053]]
```
