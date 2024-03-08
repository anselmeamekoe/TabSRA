import torch.nn as nn
from einops import rearrange

class Vec2matEncoder(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_head=8,
        dropout_rate=0.0,
        activation = None, 
        bias = True
    ):
        super(Vec2matEncoder, self).__init__()
        
        self.dim_input = dim_input
        
      
        dim_inner = dim_input*dim_head
        dim_inner1 = int(dim_inner//4)
        dim_inner2 = int(dim_inner//2)
        
        self.linear1 = nn.Linear(dim_input, dim_inner1,bias = bias )
        self.linear2 = nn.Linear(dim_inner1, dim_inner2,bias = bias)
        self.linear = nn.Linear(dim_inner2, dim_inner,bias = bias)
        
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_):
        
        if self.activation is not None:
            out = self.dropout(self.activation(self.linear1(input_)))
            out = self.dropout(self.activation(self.linear2(out)))
        else:
            out = self.dropout(self.linear1(input_))
            out = self.dropout(self.linear2(out))
        out = self.linear(out)
        out = self.sigmoid(out)
        
        # reshape to have (batch, dim_input, d_k) d_k here is the dimension of each head
        out = rearrange(out,'b (h w) -> b h w', h = self.dim_input)
        return out
class SelfReinformentAttention(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_head=8,
        dropout_rate=0.0,
        activation = None, 
        bias = True,
        s = nn.Identity()
    ):
        super(SelfReinformentAttention, self).__init__()
        
        self.dim_input = dim_input
        self.d_k = dim_head
        self.scale = dim_head**-0.5
        
        self.activation = activation
        
        # Encoding
        self.encoder_qk= Vec2matEncoder(dim_input = dim_input,
                                        dim_head=dim_head,
                                        dropout_rate=dropout_rate,
                                        activation = activation, 
                                        bias = bias
        )
        self.s = s
        
    def forward(self, input_):

        q = self.encoder_qk(input_)
        k = self.encoder_qk(input_)
        
        qk  = q*k*self.scale
        
        att = qk .sum(axis = -1)
        att = self.s(att)
        return att
