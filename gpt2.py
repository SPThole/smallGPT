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
    A single block of GPT2 Model consisting of Attention and MLP with LayerNorm and Residual Connections"""
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
        
        self.layer_norm_1 = nn.LayerNorm(self.embedding_size)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_size)
        self.WK = nn.Linear(self.embedding_size,self.num_heads*self.head_dim//self.gqa_factor)
        self.WV = nn.Linear(self.embedding_size,self.num_heads*self.head_dim//self.gqa_factor)
        self.WQ = nn.Linear(self.embedding_size,self.num_heads*self.head_dim)
        self.dropout = nn.Dropout(0.1)
        self.QKV_linear = nn.Linear(self.num_heads*self.head_dim,self.embedding_size)
        self.up_proj = nn.Linear(self.embedding_size,self.up_proj_size)
        self.down_proj = nn.Linear(self.up_proj_size,self.embedding_size)

    def attention_block(self,x):
        """Computes the attention block for a given input tensor."""
        # X --> Batch size, Sequence Length, Embedding Size
        B, S, E = x.shape

        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        Q = Q.reshape(B,S,self.num_heads,self.head_dim).permute(0,2,1,3)
        K = K.reshape(B,S,self.num_heads//self.gqa_factor,-1)
        K = torch.repeat_interleave(K,self.gqa_factor,dim=2).permute(0,2,1,3)
        V = V.reshape(B,S,self.num_heads//self.gqa_factor,-1)
        V = torch.repeat_interleave(V,self.gqa_factor,dim=2).permute(0,2,1,3)

        QK = Q@K.transpose(-2,-1)/torch.sqrt(torch.tensor(self.head_dim)//self.gqa_factor)
        mask = torch.tril(torch.ones(S, S)) == 0.0
        QK = QK.masked_fill(mask, float('-inf'))

        att_score = torch.softmax(QK,dim=-1)
        att_score = self.dropout(att_score)
        QKV = att_score@V
        QKV = QKV.permute(0,2,1,3).reshape(B,S,self.num_heads*self.head_dim)
        return QKV
    
    def forward(self, x):
        """Forward pass for the GPT2 Model Block."""
        x1 = self.layer_norm_1(x)

        x1 = self.attention_block(x1)
        x1 = self.QKV_linear(x1)
        x1 = self.dropout(x1)
        x = x + x1
        x = self.layer_norm_2(x)
        x1 = self.up_proj(x)
        x1 = nn.GELU()(x1)
        x1 = self.down_proj(x1)
        x1 = self.dropout(x1)
        x = x + x1

        return x
    
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

    def forward(self, input_ids, labels):
        """Forward pass for the GPT2 Model."""
        x = input_ids
        B, S = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.positional_embedding(torch.arange(S).unsqueeze(0).repeat(B,1))
        x = token_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss
        return logits
    

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

    
