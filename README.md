# MDS Thesis - Remaining Useful Life Estimation on Sensor Data

<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />

This repository contains the code and data for my Master's thesis on Remaining Useful Life (RUL) prediction. The thesis focuses on the application of machine learning and deep learning techniques to predict the remaining useful life of equipment.


## Table of Contents

- [Environment Details](#Environment-Details)
- [CMAPSSData](#cmapssdata)
- [Notebooks](#notebooks)
- [Scripts](#scripts)
- [Configs](#configs)
- [Outputs](#outputs)
- [Requirements](#requirements)
- [Usage](#usage)
- [References](#references)


## Environment Details
The code in this repository was developed using the following environment:
```
python==3.9.16
numpy==1.24.3
pandas==1.5.3
matplotlib==3.7.1
torch==2.0.1
scikit-learn==1.2.0
scipy==1.9.3
```

## CMAPSSData

The `CMAPSSData` directory contains the necessary data for the RUL prediction task. It includes the training, testing, and RUL files for different equipment categories.

The CMAPSSData directory contains the dataset files required for the RUL prediction task. 
For more information about the dataset, refer to the [readme.txt](CMAPSSData/readme.txt) file in the CMAPSSData directory.


## Notebooks

The `Notebooks` directory contains Jupyter notebooks related to different stages of the project:


- [Diffusion.ipynb](Diffusion.ipynb): Notebook for diffusion analysis.
- [EDA.ipynb](EDA.ipynb): Notebook for exploratory data analysis.
- [RVE_MVP.ipynb](RVE_MVP.ipynb): Notebook for RVE (Remaining Useful Life) MVP.
- [metric.ipynb](metric.ipynb): Notebook for metrics analysis.


## Scripts

The `Scripts` directory contains Python scripts for various purposes:



- [cond_diffusion.py](cond_diffusion.py): Python script for conditional diffusion.
- [cond_diffusion_original.py](cond_diffusion_original.py): Python script for original conditional diffusion.
- [diffusion_plotter.py](diffusion_plotter.py): Python script for diffusion plotting.
- [diffusion_tester.py](diffusion_tester.py): Python script for diffusion testing.
- [finetuner.py](finetuner.py): Python script for fine-tuning.
- [loss.py](loss.py): Python script for loss functions.
- [main.py](main.py): Python script for main execution.
- [metric.py](metric.py): Python script for metrics calculations.
- [metric_dataloader.py](metric_dataloader.py): Python script for metric dataloader.
- [models.py](models.py): Python script for model definitions.
- [multirun_results_getter.py](multirun_results_getter.py): Python script for getting multi-run results.
- [noise_tester.py](noise_tester.py): Python script for noise testing.
- [test.py](test.py): Python script for testing.
- [train.py](train.py): Python script for training

## Configs

The `Configs` directory contains YAML configuration files for different components of the project, such as data preprocessing, diffusion, model selection, loss functions, optimizers, schedulers, and more.

The project utilizes the Hydra library for handling and managing these configurations.

- `config.yaml`: General project configuration file.
- `data_preprocessor`: Configuration files for data preprocessing related settings.
- `diffusion`: Configuration files for diffusion-related settings.
- `model`: Configuration files for model selection and architecture settings.
- `loss`: Configuration files for different loss functions.
- `optimizer`: Configuration files for optimizer settings.
- `scheduler`: Configuration files for learning rate scheduler settings.
- `random_seed`: Configuration file for setting random seed.

## Outputs

The `Outputs` directory is used to store the output files generated during the project, such as trained model weights, evaluation results, and plots.

## Requirements

The `requirements.txt` file contains the necessary Python packages and dependencies to run the code in this repository.

## Usage

To use this code, follow these steps:

1. Install the required dependencies by running: `pip install -r requirements.txt`.
2. Update the necessary configurations in the `Configs` directory to match your specific needs.
3. Run the desired scripts or notebooks for data preprocessing, model training, testing, and evaluation.

Please refer to the specific scripts or notebooks for detailed instructions and usage examples.

```console
foo@bar:~/MDS-Thesis-RULPrediction$ python tshae_test.py --checkpoint_path ./best_models/FD003/tshae/
```



## References

If you use this code or find it helpful, please consider citing the following references:

[Provide references to relevant papers, articles, or resources related to your thesis]

