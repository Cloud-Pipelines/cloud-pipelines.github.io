---
title: Cloud Pipelines
---

# Cloud Pipelines

Cloud Pipelines project helps users build and run Machine Learning pipelines.

The Cloud Pipelines ecosystem consists of multiple projects that all work together:

* [Tangle](#tangle). A modern backend-based web app for building and running pipelines.
* [Pipeline Editor](#pipeline-editor) (legacy). The original browser-only web app for building and running pipelines.
* [SDK](https://github.com/Cloud-Pipelines/sdk). The Python SDK for creating and debugging pipeline components and pipelines.
* [ComponentSpec/`component.yaml` schema](https://github.com/Cloud-Pipelines/component_spec_schema). The formal definition of the pipeline component format.

## Tangle

Tangle is a web app that allows the users to build and run Machine Learning pipelines using drag and drop without having to set up development environment.

Unlike the older [Pipeline Editor](#pipeline-editor) app that was a client-side browser-only backendless app, the Tangle app relies on a native Cloud Pipelines backend which allows the app to be more feature-rich.

[![image](https://github.com/user-attachments/assets/0ce7ccc0-dad7-4f6a-8677-f2adcd83f558)](https://cloud-pipelines.net/pipeline-studio-app)

### Demo

Try the live demo of the [Tangle](https://cloud-pipelines.net/pipeline-studio-app) app. No registration is required to experiment with building pipelines. To install your own app instance and execute your pipelines, follow the [backend installation instructions](https://github.com/Cloud-Pipelines/tangle?tab=readme-ov-file#installation).

The app is under active development. Please check it out and report any bugs you find using [GitHub Issues](https://github.com/Cloud-Pipelines/tangle/issues).

### App features

* Start building pipelines right away
  * Intuitive visual drag and drop interface
  * No registration required to build. You own your data.
* Execute pipelines on your local machine or in Cloud
  * Easily install the app on local machine or deploy to cloud
  * Submit pipelines for execution with a single click.
  * Easily monitor all pipeline task executions, view the artifacts, read the logs.
* Fast iteration
  * Clone any pipeline run and get a new editable pipeline
  * Create pipeline -> Submit run -> Monitor run -> Clone run -> Edit pipeline -> Submit run ...
* Automatic execution caching and reuse
  * Save time and compute. Don't re-do what's done
  * Successful and even running executions are re-used from cache
* Reproducibility
  * All your runs are kept forever (on your machine) - graph, logs, metadata
  * Re-run an old pipeline run with just two clicks (Clone pipeline, Submit run)
  * Containers and strict component versioning ensure reproducibility
* Pipeline Components
  * Time-proven `ComponentSpec`/`component.yaml` format
  * A library of preloaded components
  * Fast-growing public component ecosystem
  * Add your own components (public or private)
  * Easy to create your own components manually or using the Cloud Pipelines SDK
  * Components can be written in [any language](https://github.com/Ark-kun/pipeline_components/tree/master/components/sample) (Python, Shell, R, Java, C#, etc).
  * Compatible with [Google Cloud Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction) and [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/introduction/)
  * Lots of pre-built components on GitHub: [Ark-kun/pipeline_components](https://github.com/Ark-kun/pipeline_components/tree/master/components).

### Links

[App demo](https://cloud-pipelines.net/pipeline-studio-app) (Pipeline building only. To execute pipelines, install the app locally or in Cloud.).

[Installation instructions](https://github.com/Cloud-Pipelines/tangle?tab=readme-ov-file#installation). Run the app and execute your pipelines locally or in Cloud.

[Report bugs and request features](https://github.com/Cloud-Pipelines/tangle/issues)

Source code: [Backend](https://github.com/Cloud-Pipelines/backend), [Frontend](https://github.com/Cloud-Pipelines/pipeline-studio-app), []

## Pipeline Editor

Pipeline Editor is a web app that allows the users to build and run Machine Learning pipelines using drag and drop without having to set up development environment.

### [Video](https://www.youtube.com/watch?v=7g22nupCDes)

See the [Pipeline Editor in action](https://www.youtube.com/watch?v=7g22nupCDes)
<iframe width="560" height="315" src="https://www.youtube.com/embed/7g22nupCDes?controls=0" title="Cloud Pipelines - Build machine learning pipelines without writing code" frameborder="0" allowfullscreen></iframe>

[Cloud Pipelines - Build machine learning pipelines without writing code](https://www.youtube.com/watch?v=7g22nupCDes)

### [App](https://cloud-pipelines.net/pipeline-editor)

Try the [Pipeline Editor](https://cloud-pipelines.net/pipeline-editor) now. No registration required.
[![image](https://user-images.githubusercontent.com/1829149/127566707-fceb9e41-1126-4588-b94a-c69e87fe0488.png)](https://cloud-pipelines.net/pipeline-editor)

<!-- Please check it out and report any bugs you find using [GitHub Issues](https://github.com/Cloud-Pipelines/pipeline-editor/issues).

The app is under active development, so expect some breakages as I work on the app and do not rely on the app for production. -->

### App features

* Build pipelines using drag and drop
* Execute pipelines in the cloud
  * Submit pipelines to [Google Cloud Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/) with a single click.
* Start building right away
  * No registration required
  * You own your data
* Pipeline Components
  * Time-proven `ComponentSpec`/`component.yaml` format
  * A library of preloaded components
  * Fast-growing public component ecosystem
  * Add your own components (public or private)
  * Easy to create your own components manually or using the Cloud Pipelines SDK
  * Components can be written in [any language](https://github.com/Ark-kun/pipeline_components/tree/master/components/sample) (Python, Shell, R, Java, C#, etc).
  * Compatible with [Google Cloud Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction) and [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/introduction/)
  * Lots of pre-built components on GitHub: [Ark-kun/pipeline_components](https://github.com/Ark-kun/pipeline_components/tree/master/components).
* Pipelines
  * Create, save, import and export
  * Submit for execution with a single click

### Links

[Report bugs and request features](https://github.com/Cloud-Pipelines/pipeline-editor/issues)

## Additional information

[Contact](mailto://contact@cloud-pipelines.net)

[Privacy policy](https://cloud-pipelines.net/privacy_policy)
