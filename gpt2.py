import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


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
    """
    A single block of GPT2 Model consisting of Attention and MLP with LayerNorm and Residual Connections
    """
    def __init__(self,
                 embedding_size,
                 head_dim,
                 num_heads,
                 up_proj_size=4*512, 
                 gqa_factor=1,
                 ):
        super().__init__()

        self.embedding_size = embedding_size

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.up_proj_size= up_proj_size
        self.gqa_factor = gqa_factor
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

    def attention_block(self,x,use_cache=False,train_mode=True):
        """Computes the attention block for a given input tensor."""
        # X --> Batch size, Sequence Length, Embedding Size
        B, S, E = x.shape
        # print('init',B,S,E)
        if use_cache and not train_mode:
            
            if "K_cache" not in self.cache.keys():
                # self.cache["K_cache"] = torch.zeros(B,S,self.num_heads,self.head_dim).permute(0,2,1,3)
                K = self.WK(x)
                V = self.WV(x)
                Q = self.WQ(x)
                Q = Q.reshape(B,S,self.num_heads,self.head_dim).permute(0,2,1,3) 
                K = K.reshape(B,-1,self.num_heads//self.gqa_factor,self.head_dim)
                V = V.reshape(B,-1,self.num_heads//self.gqa_factor,self.head_dim)

                K = torch.repeat_interleave(K,self.gqa_factor,dim=2).permute(0,2,1,3)
                V = torch.repeat_interleave(V,self.gqa_factor,dim=2).permute(0,2,1,3)

                self.cache["K_cache"] = K
                self.cache["V_cache"] = V


            else:
                # s = time.time()
                Q = self.WQ(x[:,-1:,:])
                # t = time.time()
                # print('s1',t-s)
                # s = time.time()
                Q = Q.reshape(B,1,self.num_heads,self.head_dim).permute(0,2,1,3)
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
                K_last = torch.repeat_interleave(K_last,self.gqa_factor,dim=2).permute(0,2,1,3)
                V_last = torch.repeat_interleave(V_last,self.gqa_factor,dim=2).permute(0,2,1,3)
                # t = time.time()
                # print('s5',t-s)
                # s = time.time()
                K = torch.cat((self.cache['K_cache'],K_last),dim=-2)
                V = torch.cat((self.cache['V_cache'],V_last),dim=-2)
                # t = time.time()
                # print('s6',t-s)
                self.cache['K_cache'] = K
                self.cache['V_cache'] = V

        else:
            # s = time.time()
            Q = self.WQ(x)
            # t = time.time()
            # print(t-s)
            K = self.WK(x)
            V = self.WV(x)
            s = time.time()
            Q = Q.reshape(B,S,self.num_heads,self.head_dim).permute(0,2,1,3)
            K = K.reshape(B,-1,self.num_heads//self.gqa_factor,self.head_dim)
            K = torch.repeat_interleave(K,self.gqa_factor,dim=2).permute(0,2,1,3)
            V = V.reshape(B,-1,self.num_heads//self.gqa_factor,self.head_dim)
            V = torch.repeat_interleave(V,self.gqa_factor,dim=2).permute(0,2,1,3)

        # print(Q.shape, K.shape, V.shape)
        QK = Q@K.transpose(-2,-1)/torch.sqrt(torch.tensor(self.head_dim))
        
        if not use_cache:
            mask = torch.tril(torch.ones(S, S)) == 0.0
            QK = QK.masked_fill(mask, float('-inf'))
        else:
            if Q.shape[2]==1:
                pass
            else:
                mask = torch.tril(torch.ones(Q.shape[2], Q.shape[2])) == 0.0
                QK = QK.masked_fill(mask, float('-inf'))

            #No masking needed

        att_score = torch.softmax(QK,dim=-1)
        if train_mode:
            att_score = self.dropout(att_score)
        QKV = att_score@V

        QKV = QKV.permute(0,2,1,3).reshape(B,-1,self.num_heads*self.head_dim)
        return QKV
    
    def forward(self, x, use_cache=False, train_mode=True):
        """Forward pass for the GPT2 Model Block."""
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
    """GPT2 Model consisting of multiple GPT2ModelBlocks"""
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
        self.track_len  = 0

        self.layers = nn.ModuleList([
            GPT2ModelBlock(
                embedding_size,
                head_dim,
                num_heads,
                up_proj_size,
                gqa_factor,
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
        """Forward pass for the GPT2 Model."""
        x = input_ids
        B, S = input_ids.shape
        
        # print(self.track_len)
        token_emb = self.token_embedding(input_ids)
        if S>1 or train_mode:
            pos_emb = self.positional_embedding(torch.arange(S).unsqueeze(0).repeat(B,1))
        else:
            pos_emb = self.positional_embedding(torch.tensor([[curr_len-1]]))
        x = token_emb + pos_emb
        for layer in self.layers:
            x = layer(x,use_cache,train_mode)
        x = self.ln_f(x)
        logits = self.head(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss
        return logits
    

    def generate(self,tokenizer,input_context,max_new_tokens=10,use_cache=True):
        """Generate text"""
        tokenized_context = tokenizer(input_context)
        input_ids = tokenized_context['input_ids']
        output = []
        train_mode = False
        curr_len = len(input_ids)
        if use_cache:
            for k in range(max_new_tokens):
                output_logits = self.forward(torch.tensor(input_ids).unsqueeze(0),labels=None,use_cache=use_cache,train_mode=train_mode,curr_len=curr_len)
                out_token = torch.argmax(output_logits[:,-1,:],dim=-1)
                input_ids = [out_token.item()]
                # print(len(input_ids))
                output.append(out_token.item())
                curr_len = curr_len+1
                self.clear_kv_cache()
        else:
            for k in range(max_new_tokens):
                output_logits = self.forward(torch.tensor(input_ids).unsqueeze(0),labels=None,use_cache=use_cache,train_mode=train_mode,curr_len=curr_len)
                out_token = torch.argmax(output_logits[:,-1,:],dim=-1)
                input_ids = input_ids + [out_token.item()]
                # print(len(input_ids))
                output.append(out_token.item())
                curr_len = curr_len+1
              
        # self.track_len = 0
        return output
    

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
    s = time.time()
    out = model.generate(tokenizer,"hi, How can I help you?",100,use_cache=False)
    t = time.time()
    print("Time taken without cache for 10 tokens",t-s)
    s = time.time()
    out = model.generate(tokenizer,"hi, How can I help you?",100,use_cache=True)
    t = time.time()
    print("Time taken with cache for 10 tokens",t-s)
    

    
