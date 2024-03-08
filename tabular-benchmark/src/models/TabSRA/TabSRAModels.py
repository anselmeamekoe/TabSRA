import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import skorch 

# Utils
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
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
class InputShapeSetterLinear(skorch.callbacks.Callback):
    def __init__(self, regression=False):
        self.regression = regression

    def on_train_begin(self, net, X, y):

        dim_input = X.shape[1]

        net.set_params(module__dim_input=dim_input,
                       module__dim_output=2 if self.regression == False else 1)


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

# Linear model with weights constraints 
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


# TabSRALinear
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
        
    def forward(self, input_):

        q = self.encoder_q(input_)
        k = self.encoder_k(input_)
        
        qk  = q*k*self.scale
        
        att = qk .sum(axis = -1)
        att = self.s(att)
        return att




class TabSRALinear(nn.Module):
    def __init__(self,
                 dim_input=20,
                 dim_output=2,
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
                 negative_indice = None 
                 
    ):
        super(TabSRALinear, self).__init__()
        
        self.get_attention = get_attention
        self.n_head = n_head
        self.n_hidden_encoder = n_hidden_encoder
        
        self.positive_indice = positive_indice
        self.negative_indice = negative_indice 
        self.attention_scaler = attention_scaler
        
        self.MHSRA = nn.ModuleList([SelfReinformentAttention(dim_input = dim_input,
                                                             dim_head = dim_head,
                                                             n_hidden_encoder = n_hidden_encoder,
                                                             dropout_rate = dropout_rate,
                                                             activation = activation,
                                                             bias = encoder_bias,
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
            return att, logit
        return logit


