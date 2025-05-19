# SurvivalAnalysisCardiologyCoxKAN

This repository contains the code needed to perform the experiments to compare models DeepSurv and CoxKAN in heart failure (HF) patientsm, with clinical data from 553 patients, including 76 features per patient, and with time to first HF-related hospitalization as the primary endpoint

## Requirements

### Data
Notebook `Toy_data_generation.ipynb` generates a toy dataset with the same structure as the one used in the study so the code in this repository can be tested. The actual data used in our study are available from the corresponding author upon reasonable request.

### Conda environments 
We provide three conda environments: 
- sksurv.yml: to execute `Comparison_Cox.py`, `Toy_data_generation.ipynb` , `Results.ipynb` and `Statistical_Tests.ipynb`.
- coxkan.yml: to execute `Comparison_CoxKan.py` and `CoxKAN_features_influence.ipynb`.
- pysurvival.yml: to execute `Comparison_DeepSurv.py`.

To install conda environments: 
```
conda env create -f sksurv.yml
```
To activate the environment: 
```
conda activate sksurv
```

## Usage
In order to reproduce the results shown in the paper, folow these steps: 
### 1. Perform experiments
```
(sksurv) >> python Comparison_Cox.py
(coxkan) >> python Comparison_CoxKan.py
(pysurvival) >> python Comparison_DeepSurv.py
```
### 2. Visualize results
The performance of each of the methods is shon in `Results.ipynb`. 
### 3. Calculate feature importance with CoxKAN
Final CoxKAN fitting, pruning, simbolic fitting and feature influence visualization in `CoxKAN_features_influence.ipynb`. 
### 4. Perform statisticla test
Results for the Nemenyi test are calculated and shown in `Statistical_Tests.ipynb`. 
