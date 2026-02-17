import torch.nn as nn
import torch

# Optimizers?

class ScratchPercpetron(nn.Module):
    def __init__(self,dim_in,bias=True):
        super(ScratchPercpetron,self).__init__()
        self.dim_in = dim_in
        self.bias = bias
        self.weights = nn.Parameter(torch.empty(1,self.dim_in),requires_grad=True)
        self.bias_wt = nn.Parameter(torch.empty(1),requires_grad=True)

    def forward(self,x):
        if self.bias:
            return x@self.weights.T + self.bias_wt

class ScratchLinear(nn.Module):
    def __init__(self,dim_in,dim_out,bias=True):
        super(ScratchLinear,self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.bias = bias

        self.linear = nn.ModuleList([ScratchPercpetron(self.dim_in,self.bias) \
                                        for k in range(self.dim_out)])

    def forward(self,x):
        out = []
        for i,l in enumerate(self.linear):
            out.append(l(x))
        return torch.stack(out,dim=-1)

class ScratchLayerNorm(nn.Module):
    def __init__(self,normalization_dims,bias,epsilon=10**-5):
        super(ScratchLayerNorm,self).__init__()
        self.normalization_dims = normalization_dims
        self.bias = bias
        self.gamma = nn.Parameter(torch.ones(self.normalization_dims),requires_grad=True) # imp to init like this as this is mult factor
        if self.bias:
            self.beta = nn.Parameter(torch.zeros(self.normalization_dims),requires_grad=True) # imp to init like this as this add factor
            
        self.epsilon = epsilon

    def forward(self,x):
        dims = x.shape[:-len(self.normalization_dims)]
        mean = torch.mean(x.reshape(*dims,-1),dim=-1)
        new_shape = (*mean.shape, *(1,) * len(self.normalization_dims))
        mean = mean.reshape(new_shape)
        var = torch.var(x.reshape(*dims,-1),dim=-1,correction=0).reshape(new_shape) # B,1 # correction is defautl 1
        #https://en.wikipedia.org/wiki/Bessel%27s_correction
        if self.bias:
            norm = (x-mean)/(var + self.epsilon)**0.5*self.gamma + self.beta
        else:
            norm = (x-mean)/(var + self.epsilon)**0.5*self.gamma
        return norm

class ScratchRMSNorm(nn.Moudle):
    pass

if __name__=="__main__":
    x = torch.randn(8,64)
    perceptron = ScratchPercpetron(64,bias=True)
    o = perceptron(x)
    print(o.shape)

    linear = ScratchLinear(64,32,bias=True)
    ol = linear(x)
    print(ol.shape)

    x = torch.randn(8,12,10,768)
    layernorm = ScratchLayerNorm((768,),True)
    torchnorm = nn.LayerNorm(768)
    print(torch.allclose(layernorm(x),torchnorm(x),atol=1e-6))

    x = torch.randn(8,12,10,768)
    layernorm = ScratchLayerNorm((10,768,),True)
    torchnorm = nn.LayerNorm((10,768))
    print(torch.allclose(layernorm(x),torchnorm(x),atol=1e-6))


        

    