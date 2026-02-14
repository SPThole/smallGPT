import torch
    
def online_softmax_mult_v(tile,V):
    local_max = torch.max(tile,dim=-1) # 8*10*4
   
    tile = tile - local_max.values.unsqueeze(-1) # broadcasting. tile is 1*4*1*tile_size and V is 1*4*1*tile_size
    tile = torch.exp(tile) #B,S, num_heads//gqa, tile_size  V shape : B, S, num_heads//gqa, head_dim
    # print("tile and v",tile.shape,V.shape)
    sum_tile = torch.sum(tile,dim=-1)
    tile = tile@V # B, num_heads//gqa, S, :chunk_size  mult B, num_heads//gqa, :chunk_size, head_dim -> S, head_dim
    # print("in func",local_max.values.shape, sum_tile.shape,tile.shape)
    return local_max.values, sum_tile,tile

def parallel_process_tiles(row,V, online_softmax_mult_v,max_workers=None,tile_size=3):
    start_c = 0
    local_max = -float('inf')*torch.ones(row.shape[:-1])
    tile_stats = {}
    if tile_stats is None:
        tile_stats = {}
    # print(row.shape)
    chunks = [row[:,:,:,i:i+tile_size] for i in range(0, row.shape[-1], tile_size)]
    # print([c.shape for c in chunks])
    from concurrent.futures import ThreadPoolExecutor
    V_chunks = [V[:,:,i:i+tile_size,:] for i in range(0, V.shape[-2], tile_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(online_softmax_mult_v, chunks,V_chunks))

    for offset, (l, s, t) in enumerate(results):
        # print(l.shape,local_max.shape)
        local_max = torch.max(torch.cat((local_max.unsqueeze(-1), l.unsqueeze(-1)), dim=-1), dim=-1).values
        # print("local max after update",local_max.shape)
        tile_stats[start_c + offset] = {'local_max': l, 'tile_contribution': s,'tile':t}
        # print("tile stats", tile_stats[start_c + offset]['tile'].shape)
    return local_max, tile_stats


def final_softmax(tile_stats, global_max):
    sums = 0
    row = 0
    # global_max = global_max.unsqueeze(-1)  
    for k,v in tile_stats.items():
        sums = sums + v['tile_contribution']*torch.exp(v['local_max']-global_max)
        row = row + v['tile']*torch.exp(v['local_max']-global_max).unsqueeze(-1) 
    return row / sums.unsqueeze(-1)


if __name__ == "__main__":
    a = torch.randn(8,10,8,8)
    V = torch.randn(8,10,8,8)
    o = parallel_process_tiles(a,V, online_softmax_mult_v, max_workers=4, tile_size=2)
    # print(o[0].shape,o[1][0]['tile_contribution'].shape, o[1][0]['local_max'].shape)
    s= final_softmax(o[1],o[0])
    print(s)
    # Comparing with regular softmax @V
    QK2 = torch.softmax(a,dim=-1)
    QKV2 = QK2@V
    print(QKV2)
    print(torch.allclose(s,QKV2,atol=1e-5))