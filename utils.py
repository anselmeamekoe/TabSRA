import os
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim 

def reset_seed_(seed):
    """
    Set seed for reproductibility 

    Parameters
    ----------
    seed : int
        The seed

    Returns
    -------
    None.

    """
    # Resetting SEED to fair comparison of results
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
class LinearScaling(nn.Module):
    def __init__(self,
                 scale = 8**-0.5
    ):
        """
        Apply  a linear scaling (activation) to the input data . For example s(x)=0.5*x
        
        Parameters
        ----------
        scale : (float) The scaling factor. The default is 8**-0.5.


        """
        super(LinearScaling, self).__init__()
        self.scale = scale
    def forward(self, x):
        return x*self.scale
def save(Model, path, device='cpu'):
    """
    Save the NN models

    Parameters
    ----------
    Model : The model to save 
        
    path : The save directory
    
    device: The device, 'cpu' by default

    Returns
    -------
    None.
    """
    #Model = Model.to(device)
    state = {
        'state_dict':Model.state_dict()
    }
    #dir_path = os.path.dirname(path)
    #os.makedirs(dir_path, exist_ok=True)#
    torch.save(state, path+'.pth') 
    
def load(Model, path, device='cpu'):
    """
    
    Parameters
    ----------
    Model : The NN model instanciation (not necesserely tuned)
    path : The directory to the saved weights
    device : The device, 'cpu' by default

    Returns
    -------
    Model : The model filled with the saved weights 

    """
    state = torch.load(path+'.pth', map_location=torch.device(device))
    Model.load_state_dict(state['state_dict'])
    return Model

def Predict(Model, Xtest, device='cpu'):
    """
    

    Parameters
    ----------
    Model : The NN model used for the inference
    Xtest (Torch Tensor) :  The datapoints to predict of shape (n_samples, n_features)
        
    device : The device use for the inference. Should be convenint to the one of the used model, default is 'cpu'.

    Returns
    -------
    pred_test (numpy float array ): The output score of the model of shape (n_samples, )
    att (numpy float array ): the attention weights of shape (n_samples, n_features) produce if Model.get_attention is True
    """
    Model.eval()
    Xtest = Xtest.to(device)
    with torch.no_grad():
        if Model.get_attention:
            att, pred_test = Model(Xtest)
            pred_test = pred_test.to('cpu').numpy().ravel()
            att = att.to('cpu').numpy()
            return att, pred_test
        else:
            pred_test = Model(Xtest).to('cpu').numpy().ravel()
            return pred_test

def TrainModel(Model, 
               train_set,
               test_set=None,
               device='cpu',
               epochs= 150,
               batch_size = 256,
               lr = 1e-2 , 
               weight_decay = 1e-5,
               verbose = 1,
               eval_every = 5,
               eval_metric = 'AUCROC',
               save_path = None,
               load_best_eval = True
              ):
    """
    Train NN model and save the best checkpoint based on test AUCROC or AUCPR

    Parameters
    ----------
    Model : The instanciation of the NN model to be trained 
    train_set (Torch TensorDataset): Cointtains the training datasets
    test_set (Torch TensorDataset):  optional, contains the validation datasets. The default is None
 
    device : The device to used for the Training. The default is 'cpu'.
    epochs (int) : The number of training epoch. The default is 150
    batch_size (int) : The batch size by default is 256.
    lr (float): The learning rate. The default is 1e-2.
    weight_decay (float) : Weight decay parameter for the optimizer. The default is 1e-5.
    verbose (int) : If verbosity during the training. By default  1.
    eval_every (int): Eval the model on the test set at every eval_every epochs . The default is 5.
    eval_metric (str): The evaluation metric between 'AUCROC' and 'AUCPR'. The default is 'AUCROC'.
    save_path (str): The directory for saving the model. By default None.
    load_best_eval (boolean): Whether to load the best validation model. Disable is test_set =None or eval_every=0 or save_path=None .

    Returns
    -------
    The trained Model
    """
    # Optimizer  (TODO: add directly the optimizer as argument)
    optimizer = optim.Adam(Model.parameters(), lr = lr, weight_decay = weight_decay )
    
    # Training criterion
    criterion = nn.BCELoss().to(device)
    
    # Data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, drop_last = False, shuffle = True) 
    
    best_perf = 0
    load_best_eval = False
    for epoch in range(epochs):
        Model.to(device)
        Model.train()

        for index, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            if Model.get_attention:
                att, out  = Model(x_batch)
            else:
                out  = Model(x_batch)
            loss = criterion(out, y_batch.reshape(out.size()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #Evalution on the test set 
        
        if (epoch%eval_every==0 and test_set is not None):
            Model.eval()
            Xtest = test_set[:][0].to(device)
            with torch.no_grad():
                if Model.get_attention:
                    att, pred_test = Model(Xtest)
                    pred_test = pred_test.to('cpu').numpy().ravel()
                else:
                    pred_test = Model(Xtest).to('cpu').numpy().ravel()

                test_aucroc  = roc_auc_score(test_set[:][1].numpy(), pred_test)
                test_aucpr =  average_precision_score(test_set[:][1].numpy(), pred_test) # AUC(pred, test_set[:][1].reshape(pred.size()) )

            # Checkpoint for the best model 
            if eval_metric =='AUCROC':
                tmp_perf = test_aucroc
            elif eval_metric =='AUCPR':
                tmp_perf = test_aucpr
            else:
                print("Unknown evaluation metric")
            if tmp_perf>best_perf:
                best_perf = tmp_perf
                #best_model = Model 
                if save_path is not None:
                    load_best_eval = True
                    save(Model, save_path, device = device)
            # print the evaluation performance
            if verbose:
                print(f"epoch = {epoch}--AUCROC perf = {test_aucroc} -- AUCPR perf = {test_aucpr}")
    return load(Model,save_path,device = device) if load_best_eval else Model


def TrainRegressionModel(Model, 
               train_set,
               test_set=None,
               device='cpu',
               epochs= 150,
               batch_size = 256,
               lr = 1e-2 , 
               weight_decay = 1e-5,
               verbose = 1,
               eval_every = 5,
               eval_metric = 'rmse',
               save_path = None,
               load_best_eval = True
              ):
    
    """
    Train NN model and save the best checkpoint based on test rmse 
    """
    # Optimizer  (TODO: add directly the optimizer as argument)
    optimizer = optim.Adam(Model.parameters(), lr = lr, weight_decay = weight_decay )
    
    # Training criterion
    criterion = nn.MSELoss().to(device)
    
    # Data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, drop_last = False, shuffle = True) 
    
    best_perf = np.inf
    for epoch in range(epochs):
        Model.to(device)
        Model.train()

        for index, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            if Model.get_attention:
                att, out  = Model(x_batch)
            else:
                out  = Model(x_batch)
            loss = criterion(out, y_batch.reshape(out.size()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #Evalution on the test set 
        if (epoch%eval_every==0 and test_set is not None):
            Model.eval()
            Xtest = test_set[:][0].to(device)
            with torch.no_grad():
                if Model.get_attention:
                    att, pred_test = Model(Xtest)
                    pred_test = pred_test.to('cpu').numpy().ravel()
                else:
                    pred_test = Model(Xtest).to('cpu').numpy().ravel() 
            if eval_metric =='rmse':
                test_perf  = np.sqrt(mean_squared_error(test_set[:][1].numpy(), pred_test))
            else:
                print("Unknown evaluation metric, please use rmse for regression problems")

            # Checkpoint for the best model 
            if test_perf<best_perf:
                best_perf = test_perf
                #best_model = Model 
                if save_path is not None:
                    save(Model, save_path,device = device)
            # print the evaluation performance
            if verbose:
                print(f"epoch = {epoch}--rmse perf = {test_perf}")
            
    return load(Model,save_path,device = device) if load_best_eval else Model