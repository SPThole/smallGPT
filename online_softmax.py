import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

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

def parallel_online_softmax(tile):
    local_max = torch.max(tile)
    tile = tile - local_max
    tile = torch.exp(tile)
    sum_tile = torch.sum(tile)
    return local_max, sum_tile

def parallel_process_tiles_mp(row, max_workers=None, tile_size=3):
    chunks = [row[i : i + tile_size] for i in range(0, len(row), tile_size)]
    
    global_max = -float('inf')
    tile_stats = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(parallel_online_softmax, chunks))

    for i, (l, s) in enumerate(results):
        global_max = max(l, global_max)
        tile_stats[i] = {'local_max': l, 'tile_contribution': s}

    return global_max, tile_stats

def final_softmax(row, tile_stats, global_max):
    total_sum = 0
    for i in tile_stats:
        l = tile_stats[i]['local_max']
        s = tile_stats[i]['tile_contribution']
        total_sum += s * torch.exp(torch.tensor(l - global_max))
    
    return torch.exp(row - global_max) / total_sum


def parallel_online_softmax(tile):
    local_max = torch.max(tile)
    tile = tile - local_max
    tile = torch.exp(tile)
    sum_tile = torch.sum(tile)
    return local_max, sum_tile

def parallel_process_tiles(row, parallel_online_softmax,max_workers=None,tile_size=3):
    start_c = 0
    local_max = -float('inf')
    tile_stats = {}
    if tile_stats is None:
        tile_stats = {}
    chunks = [row[i:i+tile_size] for i in range(0, len(row), tile_size)]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(parallel_online_softmax, chunks))

    for offset, (l, s) in enumerate(results):
        local_max = max(l, local_max)
        tile_stats[start_c + offset] = {'local_max': l, 'tile_contribution': s}

    return local_max, tile_stats

def final_softmax(row, tile_stats, global_max):
    sums = 0
    for k,v in tile_stats.items():
        sums = sums + v['tile_contribution']*torch.exp(v['local_max']-global_max)
    return torch.exp(row - global_max) / sums

if __name__=="__main__":
    a = torch.tensor([[2,4,12]],dtype=torch.float32)
    s= online_softmax(a[0],3)
    print(s)
    print(torch.softmax(a[0],dim=-1))


    a = torch.randn(1,1024)
    import time
    s = time.time()
    max_element, tile_stats=parallel_process_tiles_mp(a[0], max_workers=10, tile_size=300)
    output = final_softmax(a[0], tile_stats, max_element)
    t = time.time()
    print(f"Time taken by multiprocessing: {t-s:.4f} seconds")


    s = time.time()
    max_element, tile_stats=parallel_process_tiles(a[0], parallel_online_softmax, max_workers=10, tile_size=300)
    output = final_softmax(a[0], tile_stats, max_element)
    t = time.time()
    print(f"Time taken by threads: {t-s:.4f} seconds")



    s= time.time()
    output_torch = torch.softmax(a[0],dim=-1)
    t = time.time()
    print(f"Time taken by torch: {t-s:.4f} seconds")