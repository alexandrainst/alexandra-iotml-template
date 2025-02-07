<a href="https://github.com/alexandrainst/{{ cookiecutter.project_name }}">
<img
    src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/alexandra/alexandra-logo.jpeg"
    width="239"
    height="175"
    align="right"
/>
</a>

# Alexandra Institute Machine Learning Repository Template (IoTML addon)


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