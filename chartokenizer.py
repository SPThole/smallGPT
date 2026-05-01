from tqdm import tqdm
import torch

class CharTokenize:
    def __init__(self):
        self.special_tokens = {
                              'pad_token':'<PAD>',
                              'eos_token':'<EOS>',
                              'bos_token':'<BOS>',
                              'unk_token':'<UNK>'
                              }
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}

    def tokenize(self,text):
        return list(text)

    def encode(self, text, return_torch=False, max_len=None, pad=False):
        unk_id = self.vocab[self.special_tokens['unk_token']]
        ids = [self.vocab.get(t, unk_id) for t in text]

        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            elif pad:
                pad_id = self.vocab[self.special_tokens['pad_token']]
                ids.extend([pad_id] * (max_len - len(ids)))
        
        if return_torch:
            return torch.tensor(ids, dtype=torch.long)
        return ids
    
    def decode(self, tokens: list[int]) -> str:
        return "".join([self.inv_vocab[t] for t in tokens])

    def train(self, texts):
        unique = set()
        for t in tqdm(texts):
            unique.update(t) # Faster in-place update
            
        # Add special tokens if they exist
        if hasattr(self, 'special_tokens'):
            unique.update(self.special_tokens.values())

        # Using sorted(unique) is better so the IDs are deterministic
        self.vocab = {char: i for i, char in enumerate(sorted(unique))}
        
        # Consistent naming (self.vocab instead of self.stoi)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        return self