import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time

# Updated the earlier code with efficiency changes
# 1. Cache implimentation with buffer --> concatenation is inefficient as it creates new tensors and involves copying data. Instead, we can directly write to the appropriate index in the pre-allocated cache tensors.
# 2. Buffering with register_buffer for masking --> otherwise it would have created the mask tensor in every forward pass which is inefficient, now it is created once and stored as buffer and used in every forward pass.
# 3. Use of broadcasting in positional embedding and KV cache to avoid interleaved repeating which is inefficient
# 4. batched generate function implemented
# 5. Create scalar in scaling factor instead of tensor to avoid creating new tensor and graph for autodiff in every forward pass which is inefficient
# 6. torch.no_grad() decorator for generate function to avoid tracking gradients in generation which is inference and inefficient


def count_params(model):
    """Counts the params"""
    num_params, shapes = 0,[]
    for _,w in model.named_parameters():
        try:
            num_params = num_params + w.shape[0]*w.shape[1]
        except:
            num_params = num_params + w.shape[0]
        shapes.append([_,w.shape])
    df = pd.DataFrame(shapes,columns=['Name',"Shape"])
    return num_params,df


class GPT2ModelBlock(nn.Module):
    def __init__(self,
                 embedding_size,
                 head_dim,
                 num_heads,
                 up_proj_size=4*512, 
                 gqa_factor=1,
                 context_length=1024
                 ):
        super().__init__()

        self.embedding_size = embedding_size

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.up_proj_size= up_proj_size
        self.gqa_factor = gqa_factor
        self.context_length = context_length
        self.cache = {}
        self.layer_norm_1 = nn.LayerNorm(self.embedding_size)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_size)
        self.WK = nn.Linear(self.embedding_size,self.num_heads*self.head_dim//self.gqa_factor)
        self.WV = nn.Linear(self.embedding_size,self.num_heads*self.head_dim//self.gqa_factor)
        self.WQ = nn.Linear(self.embedding_size,self.num_heads*self.head_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.QKV_linear = nn.Linear(self.num_heads*self.head_dim,self.embedding_size)
        self.up_proj = nn.Linear(self.embedding_size,self.up_proj_size)
        self.down_proj = nn.Linear(self.up_proj_size,self.embedding_size)
        # self.first_pass =  True
        self.register_buffer('mask', torch.tril(torch.ones(self.context_length, self.context_length))==0)
      

    def attention_block(self,x,use_cache=False,train_mode=True):
        # X --> Batch size, Sequence Length, Embedding Size
        B, S, E = x.shape
        # print('init',B,S,E)
        if use_cache and not train_mode:
            
            if "K_cache" not in self.cache.keys():
                K = self.WK(x)
                V = self.WV(x)
                Q = self.WQ(x)

                self.cache["K_cache"] = torch.zeros(B,self.context_length,self.num_heads//self.gqa_factor,self.head_dim).to(x.device)
                self.cache["V_cache"] = torch.zeros(B,self.context_length,self.num_heads//self.gqa_factor,self.head_dim).to(x.device)
                self.cache["pos"] = S

                # Q shape --> (B, S, num_heads, head_dim) ==> (8,10,12,64)
                # K shape --> (B, S, num_heads/fac, head_dim) ==> (8,10,6,64)
                # V shape --> (B, S, num_heads/fac, head_dim)  ==> (8,10,6,64)
                Q = Q.reshape(B,S,self.num_heads,self.head_dim).permute(0,2,1,3) 
                K = K.reshape(B,-1,self.num_heads//self.gqa_factor,self.head_dim)
                V = V.reshape(B,-1,self.num_heads//self.gqa_factor,self.head_dim)

                self.cache["K_cache"][:,0:S,:,:] = K # for example 0 to 10 as prefix
                self.cache["V_cache"][:,0:S,:,:] = V # SAVED BEFORE REPEATING
                
                # Instead of repeat we can broadcast K and V during attention score calculation to save memory and computation, but for simplicity we are doing repeat here.
                Q = Q.reshape(B,self.num_heads//self.gqa_factor,self.gqa_factor,S,self.head_dim) # (B, num_heads/GQA, GQA,  S, head_dim)
                K = K.permute(0,2,1,3).unsqueeze(2)
                V = V.permute(0,2,1,3).unsqueeze(2)
                
                # K = torch.repeat_interleave(K,self.gqa_factor,dim=2).permute(0,2,1,3)
                # V = torch.repeat_interleave(V,self.gqa_factor,dim=2).permute(0,2,1,3)

            else:
                # s = time.time()
                Q = self.WQ(x[:,-1:,:])
                # t = time.time()
                # print('s1',t-s)
                # s = time.time()
                Q = Q.reshape(B,1,self.num_heads,self.head_dim).permute(0,2,1,3)
                Q = Q.reshape(B,self.num_heads//self.gqa_factor,self.gqa_factor,-1,self.head_dim)
                # t = time.time()
                # print('s2',t-s)
                # s = time.time()
                K_last = self.WK(x[:,-1:,:])
                V_last = self.WV(x[:,-1:,:])
                # t = time.time()
                # print('s3',t-s)
                # s = time.time()
                K_last = K_last.reshape(B,-1,self.num_heads//self.gqa_factor,self.head_dim)
                V_last = V_last.reshape(B,-1,self.num_heads//self.gqa_factor,self.head_dim)
                # t = time.time()
                # print('s4',t-s)
                # s = time.time()
                self.cache['K_cache'][:,self.cache['pos']:self.cache['pos']+1,:,:] = K_last
                self.cache['V_cache'][:,self.cache['pos']:self.cache['pos']+1,:,:] = V_last
                self.cache['pos'] = self.cache['pos'] + 1


                # t = time.time()
                # print('s5',t-s)
                # s = time.time()
                # Concatenation is inefficient as it creates new tensors and involves copying data. Instead, we can directly write to the appropriate index in the pre-allocated cache tensors.
                # K = torch.cat((self.cache['K_cache'],K_last),dim=-2)
                # V = torch.cat((self.cache['V_cache'],V_last),dim=-2)

                # doing just  self.cache['K_cache'][:,:,self.cache['pos'],:] --> will squeeze the dimension, wrong slicing
                # t = time.time()
                # print('s6',t-s)
                # self.cache['K_cache'] = K
                # self.cache['V_cache'] = V

                K = self.cache['K_cache'][:,:self.cache['pos'],:,:].permute(0,2,1,3).unsqueeze(2)
                V = self.cache['V_cache'][:,:self.cache['pos'],:,:].permute(0,2,1,3).unsqueeze(2)

        else:
            Q = self.WQ(x)
            Q = Q.reshape(B,S,self.num_heads,self.head_dim).permute(0,2,1,3)
            Q = Q.reshape(B,self.num_heads//self.gqa_factor,self.gqa_factor,S,self.head_dim)
            K = self.WK(x)
            V = self.WV(x)
            K = K.reshape(B,-1,self.num_heads//self.gqa_factor,self.head_dim)
            K = K.permute(0,2,1,3).unsqueeze(2)
            # K = torch.repeat_interleave(K,self.gqa_factor,dim=2).permute(0,2,1,3)
            V = V.reshape(B,-1,self.num_heads//self.gqa_factor,self.head_dim)
            V = V.permute(0,2,1,3).unsqueeze(2)
            # V = torch.repeat_interleave(V,self.gqa_factor,dim=2).permute(0,2,1,3)

        # print(Q.shape, K.shape, V.shape)

        # earlier i used tensor, here which will create copies whenever the forward is pass is happening lots of inefficient, so using scalar value instead
        QK = Q@K.transpose(-2,-1)/self.head_dim**0.5 # Instead of creating tensor create scalar as tensor always creates graph for autodiff
        # QK = QK.reshape(B,-1,S,S)
        # print(QK.shape)
        if not use_cache:
            # full prefix masking - since no cache
            # mask = torch.tril(torch.ones(S, S)) == 0.0 --> converting this to buffer to avoid creating this tensor again and again in every forward pass which is inefficient

            QK = QK.masked_fill(self.mask[:Q.shape[3],:Q.shape[3]], float('-inf')) # 2 to 3 as we used broadcast trick
        else:
            # incremental decoding - only mask the last token with previous tokens, since future tokens are not there in cache
            if Q.shape[3]==1:
                pass
            else:
                # when using cache but this is prefix given
                # mask = torch.tril(torch.ones(Q.shape[2], Q.shape[2])) == 0.0

                QK = QK.masked_fill(self.mask[:Q.shape[3],:Q.shape[3]], float('-inf'))

        att_score = torch.softmax(QK,dim=-1)
        # att_score = att_score.reshape(B,self.num_heads//self.gqa_factor,self.gqa_factor,S,S)
        # print(att_score.shape,V.shape)
        if train_mode:
            att_score = self.dropout(att_score)
        # V = V.squeeze(2) # (B, num_heads/GQA, 1, S, head_dim) --> (B, num_heads/GQA, S, head_dim) by broadcasting
        QKV = att_score@V
        # print(QKV.shape)
        QKV = QKV.reshape(B,self.num_heads,-1,self.head_dim).permute(0,2,1,3).reshape(B,-1,self.num_heads*self.head_dim)
        return QKV
    
    def forward(self, x, use_cache=False, train_mode=True):
        x1 = self.layer_norm_1(x)

        x1 = self.attention_block(x1,use_cache,train_mode)
        x1 = self.QKV_linear(x1)
        if train_mode:
            x1 = self.dropout(x1)
        x = x + x1
        x2 = self.layer_norm_2(x)
        x2 = self.up_proj(x2)
        x2 = self.gelu(x2)
        
        x2 = self.down_proj(x2)
        if train_mode:
            x2 = self.dropout(x2)
        x = x + x2

        return x

    def clear_cache(self):
        self.cache = {}
    



class GPT2Model(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 context_length,
                 head_dim,
                 num_heads,
                 num_layers,
                 up_proj_size=4*512, 
                 gqa_factor=1,
                 tied_embedding = True
                 ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding = nn.Embedding(context_length, embedding_size)
        self.tied_embedding = tied_embedding
        self.loss_fct = nn.CrossEntropyLoss()
        self.track_len  = 0

        self.layers = nn.ModuleList([
            GPT2ModelBlock(
                embedding_size,
                head_dim,
                num_heads,
                up_proj_size,
                gqa_factor,
                context_length=context_length
            ) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_size)
        self.head = nn.Linear(embedding_size, vocab_size, bias=False)
        if self.tied_embedding:
            self.head.weight = self.token_embedding.weight
    
    def clear_kv_cache(self):
        for layer in self.layers:
            layer.clear_cache()

    def forward(self, input_ids, labels=None, use_cache=False, train_mode=True,curr_len=None):
        # x = input_ids
        B, S = input_ids.shape
        
        # print(self.track_len)
        token_emb = self.token_embedding(input_ids)
        if S>1 or train_mode:
            # pos_emb = self.positional_embedding(torch.arange(S,device=input_ids.device).unsqueeze(0).repeat(B,1))
            pos_emb = self.positional_embedding(torch.arange(S,device=input_ids.device).unsqueeze(0)) # broadcasting will take care of repeat here as well, like KV
        else:
            pos_emb = self.positional_embedding(torch.tensor([[curr_len-1]],device=input_ids.device))
        x = token_emb + pos_emb
        for layer in self.layers:
            x = layer(x,use_cache,train_mode)
        x = self.ln_f(x)
        logits = self.head(x)

        if labels is not None:
            # earlier it was creating loss function inside forward which is inefficient as it creates new object everyd time
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss
        return logits
    
    @torch.no_grad() # important so that pytorch doesnt track diff gradients in generation which is inference
    def generate(self,tokenized_context,max_new_tokens=10,use_cache=True):
        input_ids = tokenized_context['input_ids']
        output = []
        train_mode = False
        curr_len = input_ids.shape[-1]
        for k in range(max_new_tokens):
            output_logits = self.forward(input_ids,labels=None,use_cache=use_cache,train_mode=train_mode,curr_len=curr_len)
            out_token = torch.argmax(output_logits[:,-1,:],dim=-1).reshape(-1,1)
            if use_cache:
                input_ids = out_token
                # print(input_ids.shape)
            else:
                input_ids = torch.cat((input_ids,out_token.reshape(-1,1)),dim=-1)
            output.append(out_token)
            
            curr_len = curr_len+1
            # print(curr_len)
            # print(output.shape)
        self.clear_kv_cache()
       
        return torch.cat(output,dim=-1)
    
    

if __name__ == "__main__":
    model = GPT2Model(
    vocab_size=50257,
    embedding_size=768,
    context_length=1024,
    head_dim=64,
    num_heads=12,
    num_layers=12,
    up_proj_size=3072,
    gqa_factor=1,
    tied_embedding=True)

    counts, details = count_params(model)
    print(f"TOTAL PARAM COUNT IS {counts/10**6} M")
    print(details.head(30))

    import time
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenized_input = tokenizer(["Hello, how are you?"], return_tensors='pt')
    s = time.time()
    out = model.generate(tokenized_input,max_new_tokens=300,use_cache=False)
    t = time.time()
    print("Time taken without cache for 300 tokens",t-s)
    s = time.time()
    out = model.generate(tokenized_input,max_new_tokens=300,use_cache=True)
    t = time.time()
    print("Time taken with cache for 300 tokens",t-s)
    

    
