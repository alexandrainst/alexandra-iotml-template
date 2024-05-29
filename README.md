<a href="https://github.com/alexandrainst/{{ cookiecutter.project_name }}"><img src="https://github.com/alexandrainst/alexandra-iotml-template/blob/main/%7B%7Bcookiecutter.project_name%7D%7D/gfx/alexandra_logo.png" width="239" height="175" align="right" /></a>
# IoT+ML Project Template

This repository is a template for starting an Internet-of-Things project that includes machine learning aspects. This template adds extra elements on top of its parent [ml-template](https://github.com/alexandrainst/alexandra-ml-template) repository.

## Quickstart

Install Cookiecutter:
```
pip3 install cookiecutter
```

Create a project based on the template (the `-f` flag ensures that you use the newest
version of the template):
```
cookiecutter -f gh:alexandrainst/alexandra-iotml-template
```

## Features

You can checkout the [parent repo](https://github.com/alexandrainst/alexandra-ml-template) for a list of features related to the generic template. 

## Extra Tools used in this project (on top of the parent repo tools)
* [Grafana](https://grafana.com/): Visualisation of time series
* [PostgreSQL](https://www.postgresql.org/): Database

## Project Structure
```
.
├── .devcontainer
│   └── devcontainer.json
├── .editorconfig
├── .github
│   └── workflows
│       └── ci.yaml
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── config
│   ├── __init__.py
│   ├── config.yaml
│   └── hydra
│       └── job_logging
│           └── custom.yaml
├── data
│   ├── final
│   │   └── .gitkeep
│   ├── processed
│   │   └── .gitkeep
│   └── raw
│       └── .gitkeep
├── docker-compose.yml
├── docs
│   └── .gitkeep
├── gfx
│   ├── .gitkeep
│   └── alexandra_logo.png
├── makefile
├── models
│   └── .gitkeep
├── notebooks
│   └── .gitkeep
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── src
│   ├── grafana
│   │   ├── dashboards
│   │   │   ├── example_dashboard.json
│   │   │   └── example_provisioning.yaml
│   │   └── datasources
│   │       └── example_datasource_provisioning.yaml
│   ├── nodered
│   │   └── example_ml_inference.json
│   ├── nodered_dockerfile
│   ├── preprocessor
│   │   ├── app.py
│   │   └── pyproject.toml
│   ├── preprocessor_dockerfile
│   ├── scripts
│   │   ├── eval_model.py
│   │   ├── fix_dot_env_file.py
│   │   ├── train_model.py
│   │   └── your_script.py
│   ├── sql
│   │   ├── database_init.sql
│   │   └── example_views.sql
│   └── test_project
│       ├── ml_tools
│       │   ├── datasets.py
│       │   ├── models.py
│       │   └── traintest.py
│       ├── utils
│       │   └── sql.py
│       └── your_module.py
└── tests
    ├── test_datasets.py
    ├── test_dummy.py
    └── test_models.py
```
