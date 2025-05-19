import pandas as pd
import numpy as np
from random import random
import pickle as pk
import itertools 

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.model_selection import train_test_split

from coxkan import CoxKAN

from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import integrated_brier_score

import optuna 

def objective(trial, df_train, df_val, df_test): 
    
    #suggest values for hyperparameters 
    num_hidden = trial.suggest_int("num_hidden", 0, 1)
    hidden_dim = trial.suggest_int("hidden_dim", 1, 5)
    base_fun = trial.suggest_categorical("base_fun", ["silu", "linear"])
    grid = trial.suggest_int("grid", 3, 5)
    noise_scale = trial.suggest_float("noise_scale", 0.0, 0.2)

    lr = trial.suggest_float("lr", 0.0001, 0.1, log=True)
    steps = trial.suggest_int("steps", 50, 150)
    lamb = trial.suggest_float("lamb", 0.0, 0.015)
    lamb_entropy = trial.suggest_int("lamb_entropy", 0, 14)
    lamb_coef = trial.suggest_int("lamb_coef", 0, 5)

    if num_hidden == 0: 
        width = [len(df_train.columns)-2, 1]
    else: 
        width = [len(df_train.columns)-2, hidden_dim, 1]
        
    ckan = CoxKAN(width=width, base_fun=base_fun, grid=grid, noise_scale=noise_scale)
    
    ckan.train(df_train, df_val, duration_col='t', event_col='Ingreso', lr=lr, steps=steps, lamb=lamb, 
               lamb_entropy=lamb_entropy, lamb_coef=lamb_coef)

    return ckan.cindex(df_test)

def repeatedCrossVal(X, y, n_splits=2, n_repeats=15, random_state=12345): 
    
    results_dict = dict()
    results_dict['cindex'] = []
    results_dict['times'] = []
    results_dict['auc'] = []
    results_dict['mean_auc'] = []
    results_dict['brier'] = []
    tuned_params = []
    predictions = [] 
    studies = []

    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    y_aux = np.array([(i,t) for i,t in zip(y["Ingreso"], y["t"])], dtype=[('Ingreso','?'), ('t', '<f8')])
    y_class = np.array([_y[0] for _y in y_aux])

    for i, (train_index, test_index) in enumerate(outer_cv.split(X, y_class)): 
        print(f'\tSplit {i}')
        
        X_train = X.iloc[train_index].reset_index(drop=True)
        X_test = X.iloc[test_index].reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        y_test = y.iloc[test_index].reset_index(drop=True)
        y_class_train = y_class[train_index]

        #change input format for training 
        df_train_full = pd.concat([X_train, y_train], axis=1, join='outer').astype('float32')
        df_test = pd.concat([X_test, y_test], axis=1, join='outer').astype('float32')
        df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=42)

        study = optuna.create_study(direction='maximize') #TPE sampler is default 
        study.optimize(lambda trial: objective(trial, df_train, df_val, df_test), n_trials=200)
        studies.append(study)

        best_params = study.best_params
        tuned_params.append(best_params)
        num_hidden = best_params['num_hidden']
        hidden_dim = best_params['hidden_dim']
        base_fun = best_params['base_fun']
        grid = best_params['grid']
        noise_scale = best_params['noise_scale']
    
        lr = best_params['lr']
        steps = best_params['steps']
        lamb = best_params['lamb']
        lamb_entropy = best_params['lamb_entropy']
        lamb_coef = best_params['lamb_coef']

        if num_hidden == 0: 
            width = [len(df_train.columns)-2, 1]
        else: 
            width = [len(df_train.columns)-2, hidden_dim, 1]
        ckan_best_params = CoxKAN(width=width, base_fun=base_fun, grid=grid, noise_scale=noise_scale)
        ckan_best_params.train(df_train, df_val, duration_col='t', event_col='Ingreso', lr=lr, steps=steps, lamb=lamb, 
                               lamb_entropy=lamb_entropy, lamb_coef=lamb_coef)

        predictions.append(ckan_best_params.predict(df_test))

        #cindex
        results_dict['cindex'].append(ckan_best_params.cindex(df_test))
        
        #cumulative
        min_t = min([row['t'] for index, row in y_test.iterrows()])
        max_t = max([row['t'] for index, row in y_test.iterrows()])
        _times = np.arange(min_t, max_t, 15)
        results_dict['times'].append(_times)
        risk_scores = ckan_best_params.predict(df_test)

        aux_y_train = np.array([(i,t) for i,t in zip(y_train["Ingreso"], y_train["t"])], dtype=[('Ingreso','?'), ('t', '<f8')])
        aux_y_test = np.array([(i,t) for i,t in zip(y_test["Ingreso"], y_test["t"])], dtype=[('Ingreso','?'), ('t', '<f8')])
        _auc, _mean_auc = cumulative_dynamic_auc(aux_y_train, aux_y_test, risk_scores, _times)
        results_dict['auc'].append(_auc)
        results_dict['mean_auc'].append(_mean_auc)       

    return results_dict, tuned_params, predictions, studies
        


if __name__=="__main__": 

    random_state=12345
    n_splits=2
    n_repeats=10

    X = pk.load(open(f'data/X_admissions_general_pp=(ohe_norm)_cf_pp=(bool).df', 'rb'))
    y = pk.load(open("data/y_admissions.df", "rb"))

    results_dict, tuned_params, predictions, studies = repeatedCrossVal(X, y, 
                                                                        n_splits=n_splits, n_repeats=n_repeats, 
                                                                        random_state=random_state)

    with open(f'results/scores_CoxKAN.pk', 'wb') as f: 
        pk.dump(results_dict, f)
    with open(f'results/predictions_CoxKAN.pk', 'wb') as f: 
        pk.dump(predictions, f)
    with open(f'results/tuned_params_CoxKAN.pk', 'wb') as f: 
        pk.dump(tuned_params, f)
    with open(f'results/studies_CoxKAN.pk', 'wb') as f: 
        pk.dump(studies, f)
