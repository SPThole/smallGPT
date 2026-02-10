import torch
    
def parallel_online_softmax(tile):
    local_max = torch.max(tile,dim=-1)
    # print(local_max.values.shape, tile.shape)
    tile = tile - local_max.values.unsqueeze(-1) # broadcasting
    tile = torch.exp(tile)
    sum_tile = torch.sum(tile,dim=-1)
    # print("in func",local_max.values.shape, sum_tile.shape)
    return local_max.values, sum_tile

def parallel_process_tiles(row, parallel_online_softmax,max_workers=None,tile_size=3):
    start_c = 0
    local_max = -float('inf')*torch.ones(row.shape[:-1])
    tile_stats = {}
    if tile_stats is None:
        tile_stats = {}
    # print(row.shape)
    chunks = [row[:,:,:,i:i+tile_size] for i in range(0, row.shape[-1], tile_size)]
    # print([c.shape for c in chunks])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(parallel_online_softmax, chunks))

    for offset, (l, s) in enumerate(results):
        # print(l.shape,local_max.shape)
        local_max = torch.max(torch.cat((local_max.unsqueeze(-1), l.unsqueeze(-1)), dim=-1), dim=-1).values
        print("local max after update",local_max.shape)
        tile_stats[start_c + offset] = {'local_max': l, 'tile_contribution': s}

    return local_max, tile_stats

def final_softmax(row, tile_stats, global_max):
    sums = 0
    # global_max = global_max.unsqueeze(-1)  
    for k,v in tile_stats.items():
        sums = sums + v['tile_contribution']*torch.exp(v['local_max']-global_max)
    # print("final sums",sums.shape)
    # print("global max",global_max.shape,row.shape)
    return torch.exp(row - global_max.unsqueeze(-1)) / sums.unsqueeze(-1)


if __name__ == "__main__":
    a = torch.randn(8,10,8,8)
    o = parallel_process_tiles(a, parallel_online_softmax, max_workers=4, tile_size=2)
    # print(o[0].shape,o[1][0]['tile_contribution'].shape, o[1][0]['local_max'].shape)
    s= final_softmax(a,o[1],o[0])
    print(s)
    print(torch.softmax(a,dim=-1))