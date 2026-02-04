import torch

def online_softmax(row,tile_size):
    local_max = -torch.inf
    sum_till_now = 0
    for pos in range(0,len(row),tile_size):
        
        delta = local_max - torch.tensor(max(local_max,max(row[pos:pos+tile_size]).item()))
        # print(local_max,delta)
        
        local_max = torch.tensor(max(local_max,max(row[pos:pos+tile_size]).item()))
        tile = torch.exp(row[pos:pos+tile_size] - local_max)
        # print(tile)
        contribution_from_earlier = sum_till_now*torch.exp(delta)
        # print(contribution_from_earlier)
        sum_till_now = torch.sum(tile,dim=-1) + contribution_from_earlier
        # print(sum_till_now)
        contribution_from_earlier = sum_till_now
        # print(local_max,sum_till_now)
    
    return torch.exp(row - local_max)/sum_till_now

if __name__=="__main__":
    a = torch.tensor([[1,2,3,4,5,6,7]],dtype=torch.float32)
    s= online_softmax(a[0],3)
    print(s)
    print(torch.softmax(a[0],dim=-1))