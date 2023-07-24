import torch.nn as nn
from SRA import SelfReinformentAttention 
  

class TabSRALinear(nn.Module):
    def __init__(self,
                 dim_input,
                 dim_output,
                 dim_head = 8,
                 dropout_rate = 0.0,
                 activation = nn.ReLU(),
                 get_attention = True,
                 bias = True,
                 s = nn.Identity(),
                 for_classif=True
                 
    ):
        super(TabSRALinear, self).__init__()
        
        self.get_attention = get_attention
                
        self.SelfReinformentAttention = SelfReinformentAttention(dim_input = dim_input,
                                                                   dim_head = dim_head,
                                                                   dropout_rate = dropout_rate,
                                                                   activation = activation,
                                                                   bias = bias,
                                                                   s=s
                                                                 )
        self.classifier  =  nn.Linear(dim_input, dim_output, bias =bias)
        self.out_activation = nn.Sigmoid() if for_classif else nn.Identity()
        
    def forward(self, input_):
        
        v = input_
        att = self.SelfReinformentAttention(input_)
        
        result = att*v
        
        result = self.classifier(result)
        
        if self.get_attention:
            return att, self.out_activation(result)
        return self.out_activation(result)
    
class LR(nn.Module):
    def __init__(self,
                 dim_input,
                 dim_output,
                 bias = True,
                 for_classif=True
                 
    ):
        super(LR, self).__init__()
        
        # we don't have attention for this model
        self.get_attention = False

        self.classifier  =  nn.Linear(dim_input, dim_output, bias =bias)
        self.out_activation = nn.Sigmoid() if for_classif else nn.Identity()
        
    def forward(self, input_):

        result = self.classifier(input_)
        return self.out_activation(result)