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

def flash_attention(Q, K, V, tile_size=2, attention_mask=None):
    B, N, S, D = Q.shape
    QKV = torch.zeros(B,N,S,D)
    for qi in range(0, Q.shape[-2], tile_size):
        # print('qi',qi)
        l_max = torch.ones(B,N,tile_size)*-float('inf')
        g_max = torch.ones(B,N,tile_size)*-float('inf')
        Q_chunk = Q[:,:,qi:qi+tile_size,:]
        l_sum = torch.zeros(B,N,tile_size,1)
        QKV_chunk = torch.zeros(B,N,tile_size,D)
        old_gmax = torch.ones_like(g_max)*-float('inf')
        for ki in range(0,K.shape[-2],tile_size):

            K_chunk = K[:,:,ki:ki+tile_size,:] # B, N, T, D
            V_chunk = V[:,:,ki:ki+tile_size,:] # B, N, T, D
            QK_chunk = Q_chunk@K_chunk.transpose(-2,-1)/D**0.5 # B, N, T, T
            
            
            
            if attention_mask!=None:
                # print()
                mask = attention_mask[qi:qi+tile_size,ki:ki+tile_size]==1
                QK_chunk = torch.where(mask,QK_chunk,-torch.inf)
            l_max = torch.max(QK_chunk,dim=-1).values # B,N,T
            stack_max = torch.stack((l_max, g_max), dim=-1) # B,N,T,2

            g_max = torch.max(stack_max,dim=-1).values # B,N,T
            
            QK_chunk = QK_chunk - g_max.unsqueeze(-1) # B, N, T , T - B, N, T,1 
            QK_chunk = torch.exp(QK_chunk) # B, N, T, T

            scaling = torch.exp(old_gmax-g_max).unsqueeze(-1) # B, N, T, 1
            old_gmax = g_max
            contribution_from_earlier_tile = l_sum*scaling # B, N, T, 1 * B, N, T, 1 -> B, N, T, 1
            l_sum = torch.sum(QK_chunk,dim=-1,keepdim=True) + contribution_from_earlier_tile # B, N, T, 1
            
            QKV_contribution_from_earlier_tile = QKV_chunk*scaling # B, N, T, D * B, N, T, 1 -> B, N, T, D
            # this is like adding the contribution from earlier tile in getting V earlier, it was only first tile*softmax weight was added
            QKV_chunk = QK_chunk@V_chunk + QKV_contribution_from_earlier_tile # B, N, T, T mult B, N, T, D -> B, N, T, D
        QKV[:,:,qi:qi+tile_size,:] = QKV_chunk / (l_sum) # B, N, T, D / B, N, T, 1 -> B, N, T, D
    return QKV



if __name__ == "__main__":
    a = torch.randn(8,10,8,8)
    V = torch.randn(8,10,8,8)
    o = parallel_process_tiles(a,V, online_softmax_mult_v, max_workers=4, tile_size=2)
    # print(o[0].shape,o[1][0]['tile_contribution'].shape, o[1][0]['local_max'].shape)
    s= final_softmax(o[1],o[0])
    # print(s)
    # Comparing with regular softmax @V
    QK2 = torch.softmax(a,dim=-1)
    QKV2 = QK2@V
    # print(QKV2)
    print(torch.allclose(s,QKV2,atol=1e-5))

    Q = torch.randn((8,12,10,64))
    K = torch.randn((8,12,10,64))
    V = torch.randn((8,12,10,64))
    
    oo = flash_attention(Q,K,V, tile_size=2,attention_mask=None)
    QK2 = torch.softmax(Q@K.transpose(-2,-1)/(64**0.5),dim=-1)
    QKV2 = QK2@V

    QKV3 = torch.nn.functional.scaled_dot_product_attention(Q,K,V,attn_mask=None)
    print(torch.allclose(oo,QKV2,atol=1e-5))
    print(torch.allclose(QKV3,oo,atol=1e-5))


    attention_mask = torch.tril(torch.ones(10,10))

    oo = flash_attention(Q,K,V, tile_size=2,attention_mask=attention_mask)
    additive_mask = torch.where(attention_mask == 1, 0.0, -float('inf'))
    QKV3 = torch.nn.functional.scaled_dot_product_attention(Q,K,V,attn_mask=additive_mask)
    print(torch.allclose(QKV3,oo,atol=1e-5))

