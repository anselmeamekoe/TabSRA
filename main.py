import argparse
import os, sys, time

import torch 
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score,mean_squared_error

from SRAModels import TabSRALinear
from DataProcessing import load_benchmark
from utils import TrainModel,TrainRegressionModel,reset_seed_,Predict, load, save,LinearScaling

def parse_args():
    parser = argparse.ArgumentParser()
    
    ## Dataset and split
    parser.add_argument("--datasets",type=str,
                        default='Churn',
                        choices=['Churn','CreditDefault','BankMarketing','AdultIncome','TelChurnBig','HelocFico','Blastchar','CreditCardFraud'],
                        help='The name of the benchmark dataset'
                        )
    parser.add_argument("--data_path",type=str,
                        default='Datasets/',
                        help='Data path'
                        )
    parser.add_argument("--for_classif",type=bool,
                        default=True,
                        help='whether we are dealing with  classification problem or not'
                        )
    parser.add_argument("--for_model",type=str,
                        default='deep',
                        choices=['deep','tree'],
                        help='data processing for models, standard scaling is used for deep learning models'
                        )
    parser.add_argument("--fold",type=int,
                        default=0,
                        help='Choose from 0 to 4, we only support 5-Fold CV'
                        )
    parser.add_argument("--seed",type=int,
                        default=42,
                        help='Seed for reproductibility'
                        )
    parser.add_argument("--device",
                        default='cpu',
                        help='device for training or inference'
                        )
    ## Model config abd Hyperparams
    parser.add_argument("--dim_head",type=int,
                        default=8,
                        help='attention head dimension'
                        )
    parser.add_argument("--batch_size",type=int,
                        default=256,
                        help='batch size for the training'
                        )
    parser.add_argument("--epochs",type=int,
                        default=10,
                        help='Number of training epochs'
                        )
    parser.add_argument("--eval_every",type=int,
                        default=5,
                        help='The number of training epochs before evaluating on the test set '
                        )
    parser.add_argument("--lr",type=float,
                        default=1e-2,
                        help='The learning rate used for the training'
                        )
    parser.add_argument("--dropout_rate",type=float,
                        default=0.0,
                        help='The neuron dropout rate used  in the Key and Query encorder during the training'
                        )
    parser.add_argument("--weight_decay",type=float,
                        default=0.0,
                        help='weight decay regularization'
                        )
    parser.add_argument("--eval_metric",type=str,
                        default='AUCROC',
                        choices=['AUCROC','AUCPR','MSE'],
                        help='The evaluation metric'
                        )
    ## Use mode
    parser.add_argument("--mode",type=str,
                        default='train',
                        choices=['train','test']
                        )
    parser.add_argument("--save_path",type=str,
                        default='',
                        help='Path for saving trained models or loading saved models'
                        )
    parser.add_argument("--load_best_eval",type=bool,
                        default=True,
                        help='whether to save/load the best weights with respect to the validation'
                        )   
    parser.add_argument("--verbose",type=int,
                    default=0,
                    help='whether to print the metric during training'
                    )  
    config_options = parser.parse_args()
    return config_options

def builModel(config_opt,static_params):
    Model = TabSRALinear(dim_input = static_params['dim_input'],
                   dim_output = static_params['dim_output'],
                   dim_head = config_opt.dim_head,
                   get_attention = static_params['get_attention'],
                   dropout_rate = config_opt.dropout_rate,
                   activation = static_params['activation'],
                   bias = static_params['bias'],
                   s =  LinearScaling(scale = config_opt.dim_head**-0.5)
                  )
    return Model
def main():
    config_options = parse_args()
    reset_seed_(config_options.seed)
    ## load datasets
    n_cats, n , feature_names, n_features,n_classes, datasets = load_benchmark(name=config_options.datasets,
                                                                               for_model=config_options.for_model,
                                                                               data_path=config_options.data_path,
                                                                               for_classif=config_options.for_classif
                                                                               )
    X_train_, X_test_, Y_train_, Y_test_ = datasets[config_options.fold]
    train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train_), torch.Tensor(Y_train_[:,1]))
    val_set = torch.utils.data.TensorDataset(torch.Tensor(X_test_), torch.Tensor(Y_test_[:,1]))
    
    static_params = {'dim_input':n_features, 'dim_output':n_classes-1,'get_attention':False, 'activation':nn.ReLU(), 'bias':True}
    Model = builModel(config_options,static_params)
    if config_options.mode=='train':
        if config_options.save_path=='':
            save_model_dir = config_options.datasets+'_'+str(config_options.fold)
        else:
            save_model_dir = config_options.save_path
        if config_options.for_classif:
            Model = TrainModel(Model, 
                               train_set = train_set,
                               test_set = val_set,
                               save_path=save_model_dir,
                               device = config_options.device,
                               epochs= config_options.epochs,
                               batch_size = config_options.batch_size,
                               lr = config_options.lr,
                               eval_every = config_options.eval_every,
                               weight_decay = config_options.weight_decay,
                               verbose=config_options.verbose,
                               load_best_eval=config_options.load_best_eval,
                               eval_metric=config_options.eval_metric
                               )
        else:
            Model = TrainRegressionModel(Model,
                               train_set = train_set,
                               test_set = val_set,
                               save_path=save_model_dir,
                               device = config_options.device,
                               epochs= config_options.epochs,
                               batch_size = config_options.batch_size,
                               lr = config_options.lr,
                               eval_every = config_options.eval_every,
                               weight_decay = config_options.weight_decay,
                               verbose=config_options.verbose,
                               load_best_eval=config_options.load_best_eval,
                               eval_metric=config_options.eval_metric
                               )
    else:
        Model = load(Model, path=config_options.save_path,device=config_options.device)


    pred_val = Predict(Model,val_set[:][0],device=config_options.device )
    pred_train = Predict(Model,train_set[:][0],device=config_options.device )
    if config_options.eval_metric=='AUCROC':
        test_perf = roc_auc_score(val_set[:][1].numpy(), pred_val)
        train_perf= roc_auc_score(train_set[:][1].numpy(), pred_train)
        print(f"--Train AUCROC = {train_perf} -- Test AUCROC = {test_perf}")
    elif config_options.eval_metric=='AUCPR':
        test_perf= average_precision_score(val_set[:][1].numpy(), pred_val)
        train_perf = average_precision_score(train_set[:][1].numpy(), pred_train)
        print(f"--Train AUCPR = {train_perf} -- Test AUCPR = {test_perf}")
    else:
        test_perf = mean_squared_error(val_set[:][1].numpy(), pred_val)
        train_perf = mean_squared_error(train_set[:][1].numpy(), pred_train)
        print(f"--Train MSE = {train_perf} -- Test MSE = {test_perf}")
if __name__=='__main__':
    main()
