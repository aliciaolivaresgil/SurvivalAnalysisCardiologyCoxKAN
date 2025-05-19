#export LD_LIBRARY_PATH=/home/aolivares/miniconda3/envs/pysurvival/lib/python3.7/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

import pandas as pd
import numpy as np
from random import random
import pickle as pk
import datetime

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.utils.metrics import concordance_index

from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import integrated_brier_score, concordance_index_censored

import optuna

def objective(trial, X_train, X_test, y_train, y_test): 
    
    #suggest values for hyperparameters
    num_hidden = trial.suggest_int('num_hidden', 1, 3)
    structure = []
    for i, hidden in enumerate(range(num_hidden)): 
        activation = trial.suggest_categorical(f'activation{i}', ['LogLog', 'Tanh', 'ReLU'])
        num_units = trial.suggest_int(f'num_units{i}', 50, 200)
        structure.append({'activation': activation, 'num_units': num_units})
        
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    lr = trial.suggest_float('lr', 0.0001, 0.1, log=True)
    num_epochs = trial.suggest_int('num_epochs', 100, 1000)
    l2_reg = trial.suggest_float('l2_reg', 0.0, 0.015)
    dropout = trial.suggest_float('dropout', 0, 1)
    
    model = NonLinearCoxPHModel(structure=structure)

    try: 
        model.fit(X_train, y_train['t'], y_train['Ingreso'], optimizer=optimizer, lr=lr, num_epochs=num_epochs, 
                  l2_reg=l2_reg, dropout=dropout)
        cindex = concordance_index(model, X_test, y_test['t'], y_test['Ingreso'])
    except ValueError: 
        cindex = 0
    
    return cindex
                           
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
    
    y_class = np.array([_y for _y in y['Ingreso']])
    
    for i, (train_index, test_index) in enumerate(outer_cv.split(X, y_class)): 
        print(f'\tSplit {i}, {datetime.datetime.now()}')
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        study = optuna.create_study(direction='maximize') #TPE sampler is default
        study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=200)
        studies.append(study)
        
        best_params = study.best_params
        tuned_params.append(best_params)
        num_hidden = best_params['num_hidden']
        structure = []
        for i, hidden in enumerate(range(num_hidden)): 
            activation = best_params[f'activation{i}']
            num_units = best_params[f'num_units{i}']
            structure.append({'activation': activation, 'num_units': num_units})
        optimizer = best_params['optimizer']
        lr = best_params['lr']
        num_epochs = best_params['num_epochs']
        l2_reg = best_params['l2_reg'] 
        
        best_model = NonLinearCoxPHModel(structure=structure)
        best_model.fit(X_train, y_train['t'], y_train['Ingreso'], optimizer=optimizer, lr=lr, 
                       num_epochs=num_epochs, l2_reg=l2_reg) 
        
        predictions.append(best_model.predict_risk(X_test))
        
        #cindex 
        aux_y_train = np.array(
            [(row['Ingreso'], row['t']) for index, row in y_train.iterrows()], dtype=[('Ingreso','?'), ('t', '<f8')])
        aux_y_test = np.array(
            [(row['Ingreso'], row['t']) for index, row in y_test.iterrows()], dtype=[('Ingreso','?'), ('t', '<f8')])

        results_dict['cindex'].append(concordance_index_censored(aux_y_test['Ingreso'], aux_y_test['t'], best_model.predict_risk(X_test))[0])
        print(results_dict['cindex'])
        
        #cumulative
        min_t = min([row['t'] for index, row in y_test.iterrows()])
        max_t = max([row['t'] for index, row in y_test.iterrows()])
        _times = np.arange(min_t, max_t, 15)
        results_dict['times'].append(_times)
        risk_scores = best_model.predict_risk(X_test)


                                
        _auc, _mean_auc = cumulative_dynamic_auc(aux_y_train, aux_y_test, risk_scores, _times)
        results_dict['auc'].append(_auc)
        results_dict['mean_auc'].append(_mean_auc) 
        
    return results_dict, tuned_params, predictions, studies
    
if __name__=="__main__": 
    
    random_state=12345
    n_splits=2
    n_repeats=10

    #X = pk.load(open(f'data/X_admissions_general_pp=(ohe_norm)_cf_pp=(bool).df', 'rb'))
    X = pd.read_csv(f'data/X_admissions_general_pp=(ohe_norm)_cf_pp=(bool).csv', sep=',')
    #y = pk.load(open("data/y_admissions.df", "rb"))
    y = pd.read_csv("data/y_admissions.csv", sep=',')
    
    results_dict, tuned_params, predictions, studies = repeatedCrossVal(X, y, 
                                                                        n_splits=n_splits, n_repeats=n_repeats, 
                                                                        random_state=random_state)
    
    with open(f'results/scores_DeepSurv.pk', 'wb') as f: 
        pk.dump(results_dict, f)
    with open(f'results/predictions_DeepSurv.pk', 'wb') as f: 
        pk.dump(predictions, f)
    with open(f'results/tuned_params_DeepSurv.pk', 'wb') as f: 
        pk.dump(tuned_params, f)
    with open(f'results/studies_DeepSurv.pk', 'wb') as f: 
        pk.dump(studies, f)

    