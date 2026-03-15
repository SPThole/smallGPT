from tqdm import tqdm


class CharTokenize:
    def __init__(self):
        self.special_tokens = {
                              'pad_token':'<PAD>',
                              'eos_token':'<EOS>',
                              'bos_token':'<BOS>',
                              'unk_token':'<UNK>'
                              }

    def tokenize(self,text):
        return list(text)

    def encode(self, text):
        unk_id = self.vocab[self.special_tokens['unk_token']]
        return [self.vocab.get(t, unk_id) for t in text]
    
    def decode(self,tokens):
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