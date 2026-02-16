import torch
def get_multiplier_mat(m,i,d,base=10000):
    theta = torch.tensor(m*base**(-2*(i-1)/d),dtype=torch.float32)
    multiplier = torch.tensor([[torch.cos(theta),-torch.sin(theta)],[torch.sin(theta),torch.cos(theta)]])
    return multiplier.T

def rotary_embedding(m,x,base=10000):
    B, D = x.shape
    for i,k in enumerate(range(0,D,2)):
        x[:,k:k+2] = x[:,k:k+2]@get_multiplier_mat(m,i+1,D,base)
    return x
   
if __name__ == "__main__":
    x = torch.randn(8,64)
    o = rotary_embedding(1,x)
    print(o.shape)