import os
import numpy as np
import torch
import torch.nn as nn
#from SRA import SelfReinformentAttention
from einops import rearrange 
import skorch

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

class LinearConstraint(nn.Module):
    def __init__(self,
                 dim_input,
                 dim_output,
                 bias = True,
                 positive_indice =None,
                 negative_indice = None, 
                  ):
        super(LinearConstraint, self).__init__()
        self.dim_output = dim_output
        self.classifier  =  nn.Linear(dim_input, dim_output, bias =bias)
        self.positive_indice=positive_indice
        self.negative_indice= negative_indice
    def forward(self, input_):
        
        
        if self.positive_indice is not None:
            for class_index in range(self.dim_output):
                self.classifier.weight.data[class_index][self.positive_indice] = torch.clamp(self.classifier.weight.data[class_index][self.positive_indice], min=0.0)
        if self.negative_indice is not None:
            for class_index in range(self.dim_output):
                self.classifier.weight.data[class_index][self.negative_indice] = torch.clamp(self.classifier.weight.data[class_index][self.negative_indice], max=0.0)
        result = self.classifier(input_)
        return result
    
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
    
class InputShapeSetterTabSRA(skorch.callbacks.Callback):
    def __init__(self, regression=False):
        self.regression = regression

    def on_train_begin(self, net, X, y):

        dim_input = X.shape[1]
        net.set_params(module__dim_input=dim_input,module__dim_output=1)  

### SRA block
class Vec2matEncoder(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_head=8,
        n_hidden_encoder=2,
        dropout_rate=0.0,
        activation = nn.Identity(), 
        bias = True
    ):
        super(Vec2matEncoder, self).__init__()
        # Dimensions
        self.dim_input = dim_input
         
        dim_output_encoder = dim_input*dim_head
        
        # hidden layer dimensions
        dim_inners = [max(dim_input,dim_output_encoder//2**(n_hidden_encoder-i)) for i in range(n_hidden_encoder)]
        
        # linear layers  
        self.inners_linear = nn.ModuleList([
            nn.Linear(dim_input,dim,bias = bias) if i==0 else nn.Linear(dim_inners[i-1],dim,bias = bias) for i,dim in enumerate(dim_inners)
        ]
        )
        self.out_linear = nn.Linear(dim_inners[-1], dim_output_encoder,bias = bias)
        
        # activation functions
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        
        # Dropout 
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, input_):
        
        for linear_layer in self.inners_linear:
            input_ = linear_layer(input_)
            input_ = self.activation(input_)
            input_ = self.dropout(input_)

        out = self.out_linear(input_)
        out = self.sigmoid(out)
        
        # reshape to have (batch, dim_input, d_k) d_k here is the dimension of each head
        out = rearrange(out,'b (h w) -> b h w', h = self.dim_input)
        return out

class SelfReinformentAttention(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_head=8,
        n_hidden_encoder=2,
        dropout_rate=0.0,
        activation = nn.Identity(), 
        bias = True,
        no_interaction_indice= None,
        s = nn.Identity()
    ):
        super(SelfReinformentAttention, self).__init__()
        
        self.dim_input = dim_input
        self.d_k = dim_head
        self.scale = dim_head**-0.5
        
        self.activation = activation
        
        # Encoding
        self.encoder_q= Vec2matEncoder(dim_input = dim_input,
                                        dim_head=dim_head,
                                        n_hidden_encoder=n_hidden_encoder,
                                        dropout_rate=dropout_rate,
                                        activation = activation, 
                                        bias = bias
        )
        self.encoder_k= Vec2matEncoder(dim_input = dim_input,
                                        dim_head=dim_head,
                                        n_hidden_encoder=n_hidden_encoder,
                                        dropout_rate=dropout_rate,
                                        activation = activation, 
                                        bias = bias
        )
        self.s = s
        self.no_interaction_indice = no_interaction_indice
        
    def forward(self, input_):

        q = self.encoder_q(input_)
        k = self.encoder_k(input_)
        
        qk  = q*k*self.scale
        
        att = qk .sum(axis = -1)
        att = self.s(att)
        if self.no_interaction_indice:
            att[:, self.no_interaction_indice]=1
        return att
        
### For Regression  task            
class TabSRALinear_Regressor_Base(nn.Module):
    def __init__(self,
                 dim_input,
                 dim_output,
                 dim_head = 8,
                 n_hidden_encoder=2,
                 n_head = 2,
                 dropout_rate = 0.0,
                 activation = nn.ReLU(),
                 get_attention = False,
                 encoder_bias = True,
                 classifier_bias = True,
                 attention_scaler = 'sqrt',
                 positive_indice = None,
                 negative_indice = None ,
                 no_interaction_indice= None
                 
    ):
        super(TabSRALinear_Regressor_Base, self).__init__()
        
        self.get_attention = get_attention
        self.n_head = n_head
        self.n_hidden_encoder = n_hidden_encoder
        
        self.positive_indice = positive_indice
        self.negative_indice = negative_indice
        self.no_interaction_indice = no_interaction_indice
        self.attention_scaler = attention_scaler
        
        self.MHSRA = nn.ModuleList([SelfReinformentAttention(dim_input = dim_input,
                                                             dim_head = dim_head,
                                                             n_hidden_encoder = n_hidden_encoder,
                                                             dropout_rate = dropout_rate,
                                                             activation = activation,
                                                             bias = encoder_bias,
                                                             no_interaction_indice= no_interaction_indice,
                                                             s= nn.Identity() if attention_scaler=='identity' else LinearScaling(scale = dim_head**-0.5),
                                                             ) for h in range(n_head)
                                   ]
                                  )
        
        
        
        self.classifiers  = nn.ModuleList([LinearConstraint(dim_input = dim_input,
                                                            dim_output = dim_output,
                                                            bias = classifier_bias,
                                                            positive_indice = positive_indice,
                                                            negative_indice = negative_indice
                                                            )  for h in range(n_head) 
                                           ]
                                          )
           
        #self.out_activation = nn.Sigmoid() if for_classif else nn.Identity()
        
    def forward(self, input_):
        v = input_
       
        att = torch.stack([self.MHSRA[h](v) for h in range(self.n_head)])
        
        out_rep = att*v
        
        
        result = torch.stack([self.classifiers[h](out_rep[h]) for h in range(self.n_head)]) 
        
        logit =  result.sum(axis=0)
        
        if self.get_attention:
            return att, logit#self.out_activation(out)
        return logit #self.out_activation(out)
    
class LinearRegressor_Constructor(skorch.NeuralNetRegressor):
    def get_feature_attribution(self, X, device=None):
        device = device if device is not None else self.device
        Linear_module = self.module_.to(device)

        with torch.no_grad():
            X_Tensor = torch.Tensor(X).to(device)
            attribution =  X_Tensor*Linear_module.classifier.weight.data
        return attribution.to('cpu').numpy()
    def get_weights(self):
        Linear_module = self.module_
        return [Linear_module.classifier.weight.data.to('cpu').numpy()]
    
class TabSRALinearRegressor_Constructor(skorch.NeuralNetRegressor):
    def predict_inference(self, X, device=None):
        device = device if device is not None else self.device
        TabSRA_module = self.module_.to(device)
        TabSRA_module.eval()
        with torch.no_grad():
            X_Tensor = torch.Tensor(X).to(device)
            logit = TabSRA_module(X_Tensor)
            
        return logit.to('cpu').numpy()    
    def get_feature_attribution(self, X, device=None):
        device = device if device is not None else self.device
        TabSRA_module = self.module_.to(device)
        TabSRA_module.get_attention = True
        TabSRA_module.eval()
        with torch.no_grad():
            X_Tensor = torch.Tensor(X).to(device)
            att, _ = TabSRA_module(X_Tensor)
            att_coef = att[0]*TabSRA_module.classifiers[0].classifier.weight.data#.to('cpu').numpy()
            n_head = TabSRA_module.n_head
            if n_head>1:
                for head in range(1,n_head):
                    att_coef += att[head]*TabSRA_module.classifiers[head].classifier.weight.data
            attribution =  X_Tensor*att_coef
    
        self.module_.get_attention = False
        return attribution.to('cpu').numpy()
    def get_weights(self):
        TabSRA_module = self.module_
        n_head = TabSRA_module.n_head
        return [TabSRA_module.classifiers[head].classifier.weight.data.to('cpu').numpy() for head in range(n_head)]
    def get_attention(self, X, device=None):
        device = device if device is not None else self.device
        TabSRA_module = self.module_.to(device)
        TabSRA_module.get_attention = True
        TabSRA_module.eval()
        with torch.no_grad():
            X_Tensor = torch.Tensor(X).to(device)
            att, _ = TabSRA_module(X_Tensor)
            
        self.module_.get_attention = False
        return att.to('cpu').numpy()
def LinearRegressor(
                 module__bias = True,
                 optimizer = torch.optim.Adam,
                 lr = 0.05,
                 batch_size = 256,
                 max_epochs = 100,
                 iterator_train__shuffle = True,
                 verbose = 1,
                 random_state = 42,
                 callbacks = [InputShapeSetterTabSRA(regression=True)],
                 **kwargs
        ):
    reset_seed_(random_state)
    model = LinearRegressor_Constructor(
        LinearConstraint,
        module__dim_input = 1,
        module__dim_output = 1,
        module__bias = module__bias,
        optimizer = optimizer,
        lr = lr,
        batch_size = batch_size,
        max_epochs = max_epochs,
        iterator_train__shuffle = iterator_train__shuffle,
        verbose = verbose,
        callbacks = callbacks,
        **kwargs
        )
    return model 
def TabSRALinearRegressor(
                 module__dim_head = 8,
                 module__n_hidden_encoder=2,
                 module__n_head = 1,
                 module__dropout_rate = 0.0,
                 module__encoder_bias =True,
                 module__classifier_bias = True,
                 module__attention_scaler = 'sqrt',
                 optimizer = torch.optim.Adam,
                 lr = 0.05,
                 batch_size = 256,
                 max_epochs = 100,
                 iterator_train__shuffle = True,
                 verbose = 1,
                 random_state = 42,
                 callbacks = [InputShapeSetterTabSRA(regression=True)],
                 **kwargs
        ):
    reset_seed_(random_state)
    model = TabSRALinearRegressor_Constructor(
        TabSRALinear_Regressor_Base,
        module__dim_input = 1,
        module__dim_output = 1,
        module__dim_head = module__dim_head,
        module__n_hidden_encoder = module__n_hidden_encoder,
        module__n_head = module__n_head,
        module__dropout_rate = module__dropout_rate,
        module__encoder_bias = module__encoder_bias,
        module__classifier_bias =module__classifier_bias,
        module__attention_scaler = module__attention_scaler,
        optimizer = optimizer,
        lr = lr,
        batch_size = batch_size,
        max_epochs = max_epochs,
        iterator_train__shuffle = iterator_train__shuffle,
        verbose = verbose,
        callbacks = callbacks,
        **kwargs
        )
    return model

### Binary classifier
class LinearConstraint_Base(nn.Module):
    def __init__(self,
                 dim_input,
                 dim_output,
                 bias = True,
                 positive_indice =None,
                 negative_indice = None, 
                  ):
        super(LinearConstraint_Base, self).__init__()
        self.dim_output = dim_output
        self.classifier  =  nn.Linear(dim_input, dim_output, bias =bias)
        self.positive_indice=positive_indice
        self.negative_indice= negative_indice
    def forward(self, input_):
        
        
        if self.positive_indice is not None:
            for class_index in range(self.dim_output):
                self.classifier.weight.data[class_index][self.positive_indice] = torch.clamp(self.classifier.weight.data[class_index][self.positive_indice], min=0.0)
        if self.negative_indice is not None:
            for class_index in range(self.dim_output):
                self.classifier.weight.data[class_index][self.negative_indice] = torch.clamp(self.classifier.weight.data[class_index][self.negative_indice], max=0.0)
        result = self.classifier(input_)
        return result.squeeze(-1)
class TabSRALinear_Base(nn.Module):
    def __init__(self,
                 dim_input,
                 dim_output,
                 dim_head = 8,
                 n_hidden_encoder=2,
                 n_head = 2,
                 dropout_rate = 0.0,
                 activation = nn.ReLU(),
                 get_attention = False,
                 encoder_bias = True,
                 classifier_bias = True,
                 attention_scaler = 'sqrt',
                 positive_indice = None,
                 negative_indice = None,
                 no_interaction_indice= None
                 
    ):
        super(TabSRALinear_Base, self).__init__()
        
        self.get_attention = get_attention
        self.n_head = n_head
        self.n_hidden_encoder = n_hidden_encoder
        
        self.positive_indice = positive_indice
        self.negative_indice = negative_indice
        self.no_interaction_indice = no_interaction_indice
        self.attention_scaler = attention_scaler
        
        self.MHSRA = nn.ModuleList([SelfReinformentAttention(dim_input = dim_input,
                                                             dim_head = dim_head,
                                                             n_hidden_encoder = n_hidden_encoder,
                                                             dropout_rate = dropout_rate,
                                                             activation = activation,
                                                             bias = encoder_bias,
                                                             no_interaction_indice = no_interaction_indice,
                                                             s= nn.Identity() if attention_scaler=='identity' else LinearScaling(scale = dim_head**-0.5),
                                                             ) for h in range(n_head)
                                   ]
                                  )
        
        
        
        self.classifiers  = nn.ModuleList([LinearConstraint(dim_input = dim_input,
                                                            dim_output = dim_output,
                                                            bias = classifier_bias,
                                                            positive_indice = positive_indice,
                                                            negative_indice = negative_indice
                                                            )  for h in range(n_head) 
                                           ]
                                          )
           
        #self.out_activation = nn.Sigmoid() if for_classif else nn.Identity()
        
    def forward(self, input_):
        v = input_
       
        att = torch.stack([self.MHSRA[h](v) for h in range(self.n_head)])
        
        out_rep = att*v
        
        
        result = torch.stack([self.classifiers[h](out_rep[h]) for h in range(self.n_head)]) 
        
        logit =  result.sum(axis=0).squeeze(-1)
        
        if self.get_attention:
            return att, logit#self.out_activation(out)
        return logit #self.out_activation(out)
class LinearClassifier_Constructor(skorch.NeuralNetClassifier):
    def get_feature_attribution(self, X, device=None):
        device = device if device is not None else self.device
        Linear_module = self.module_.to(device)

        with torch.no_grad():
            X_Tensor = torch.Tensor(X).to(device)
            attribution =  X_Tensor*Linear_module.classifier.weight.data
        return attribution.to('cpu').numpy()
    def get_weights(self):
        Linear_module = self.module_
        return [Linear_module.classifier.weight.data.to('cpu').numpy()]    
class TabSRALinearClassifier_Constructor(skorch.NeuralNetClassifier):
    def predict_inference(self, X, device=None):
        device = device if device is not None else self.device
        TabSRA_module = self.module_.to(device)
        TabSRA_module.eval()
        out_activation = nn.Sigmoid().to(device)
        with torch.no_grad():
            X_Tensor = torch.Tensor(X).to(device)
            logit = TabSRA_module(X_Tensor)
            logit = out_activation(logit)
        return logit.to('cpu').numpy()
    def get_feature_attribution(self, X, device=None):
        device = device if device is not None else self.device
        TabSRA_module = self.module_.to(device)
        TabSRA_module.eval()
        TabSRA_module.get_attention = True
        with torch.no_grad():
            X_Tensor = torch.Tensor(X).to(device)
            att, _ = TabSRA_module(X_Tensor)
            att_coef = att[0]*TabSRA_module.classifiers[0].classifier.weight.data[0]#.to('cpu').numpy()
            n_head = TabSRA_module.n_head
            if n_head>1:
                for head in range(1,n_head):
                    att_coef += att[head]*TabSRA_module.classifiers[head].classifier.weight.data[0]
            attribution =  X_Tensor*att_coef
    
        self.module_.get_attention = False
        return attribution.to('cpu').numpy()
    def get_weights(self):
        TabSRA_module = self.module_
        n_head = TabSRA_module.n_head
        return [TabSRA_module.classifiers[head].classifier.weight.data.to('cpu').numpy()[0] for head in range(n_head)]
    def get_attention(self, X, device=None):
        device = device if device is not None else self.device
        TabSRA_module = self.module_.to(device)
        TabSRA_module.eval()
        TabSRA_module.get_attention = True
        with torch.no_grad():
            X_Tensor = torch.Tensor(X).to(device)
            att, _ = TabSRA_module(X_Tensor)
            
        self.module_.get_attention = False
        return att.to('cpu').numpy()
def LinearClassifier(
                 module__bias = True,
                 optimizer = torch.optim.Adam,
                 lr = 0.05,
                 batch_size = 256,
                 max_epochs = 100,
                 iterator_train__shuffle = True,
                 verbose = 1,
                 random_state = 42,
                 callbacks = [InputShapeSetterTabSRA(regression=False)],
                 **kwargs
        ):
    reset_seed_(random_state)
    model = LinearClassifier_Constructor(
        LinearConstraint_Base,
        module__dim_input = 1,
        module__dim_output = 1,
        module__bias = module__bias,
        optimizer = optimizer,
        lr = lr,
        batch_size = batch_size,
        max_epochs = max_epochs,
        iterator_train__shuffle = iterator_train__shuffle,
        verbose = verbose,
        callbacks = callbacks,
        **kwargs
        )
    return model 
def TabSRALinearClassifier(
                 module__dim_head = 8,
                 module__n_hidden_encoder=2,
                 module__n_head = 1,
                 module__dropout_rate = 0.0,
                 module__encoder_bias =True,
                 module__classifier_bias = True,
                 module__attention_scaler = 'sqrt',
                 optimizer = torch.optim.Adam,
                 lr = 0.05,
                 batch_size = 256,
                 max_epochs = 100,
                 iterator_train__shuffle = True,
                 verbose = 1,
                 random_state = 42,
                 callbacks = [InputShapeSetterTabSRA(regression=False)],
                 **kwargs
        ):
    reset_seed_(random_state)
    model = TabSRALinearClassifier_Constructor(
        TabSRALinear_Base,
        module__dim_input = 1,
        module__dim_output = 1,
        module__dim_head = module__dim_head,
        module__n_hidden_encoder = module__n_hidden_encoder,
        module__n_head = module__n_head,
        module__dropout_rate = module__dropout_rate,
        module__encoder_bias = module__encoder_bias,
        module__classifier_bias =module__classifier_bias,
        module__attention_scaler = module__attention_scaler,
        optimizer = optimizer,
        lr = lr,
        batch_size = batch_size,
        max_epochs = max_epochs,
        iterator_train__shuffle = iterator_train__shuffle,
        verbose = verbose,
        callbacks = callbacks,
        **kwargs
        )
    return model
