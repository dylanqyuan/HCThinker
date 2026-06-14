This repository contains the inference demo files and a released checkpoint for HCThinker, for academic use.

#### Resources

* Model weights: [Link](https://www.modelscope.cn/models/modelscopeUser6806/demo_3)

#### Model

This repository provides a single consolidated release checkpoint.

The checkpoint is released as a unified public model version, rather than a set of experiment-specific checkpoints tied to dataset-specific training and evaluation settings.

It is trained using GAIC-H-train, FCDB-H-train, and the available data from CPC-H and FLMS-H.

#### Files

Related inference code is included in this repository.

The main inference entry is `demo.py`. The environment file is provided as `environment.yml`.

#### Demo

`demo.py` provides a minimal inference example using the released checkpoint and an example image.

It is a demo entry point, not an installation guide, deployment tutorial, or evaluation script.

#### Scope

Not an actively maintained package, production system, deployment framework, LTS release, or support service.

Provided as-is. Runtime environment, dependencies, model paths, hardware settings, and compatibility may vary across systems.

#### License

Released under *CC BY-NC 4.0*.

Model weights hosted on ModelScope may be subject to separate terms.
