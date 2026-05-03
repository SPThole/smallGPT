from tqdm import tqdm
import torch

class TokenizerOutput(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

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

    @property
    def pad_token(self):
        return self.special_tokens['pad_token']

    @property
    def eos_token(self):
        return self.special_tokens['eos_token']

    @property
    def bos_token(self):
        return self.special_tokens['bos_token']

    @property
    def unk_token(self):
        return self.special_tokens['unk_token']

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def eos_token_id(self):
        return self.vocab[self.eos_token]

    @property
    def bos_token_id(self):
        return self.vocab[self.bos_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token]

    def tokenize(self,text):
        return list(text)

    def __call__(self, text, return_tensors=None, **kwargs):
        if "max_length" in kwargs and "max_len" not in kwargs:
            kwargs["max_len"] = kwargs.pop("max_length")
        ids = self.encode(text, **kwargs)
        attention_mask = [0 if token_id == self.pad_token_id else 1 for token_id in ids]

        if return_tensors == "pt":
            return TokenizerOutput({
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            })

        return TokenizerOutput({"input_ids": ids, "attention_mask": attention_mask})

    def encode(self, text, return_torch=False, max_len=None, pad=False):
        unk_id = self.unk_token_id
        ids = [self.vocab.get(t, unk_id) for t in text]

        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            elif pad:
                ids.extend([self.pad_token_id] * (max_len - len(ids)))
        
        if return_torch:
            return torch.tensor(ids, dtype=torch.long)
        return ids
    
    def decode(self, tokens: list[int], skip_special_tokens=False) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        special_tokens = set(self.special_tokens.values())
        chars = []
        for token in tokens:
            char = self.inv_vocab[token]
            if skip_special_tokens and char in special_tokens:
                continue
            chars.append(char)
        return "".join(chars)

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(encoded_inputs, dict):
            encoded_inputs = [encoded_inputs]

        input_ids = [
            item["input_ids"].tolist() if isinstance(item["input_ids"], torch.Tensor) else list(item["input_ids"])
            for item in encoded_inputs
        ]

        if padding is False:
            target_length = None
        elif max_length is not None:
            target_length = max_length
        else:
            target_length = max(len(ids) for ids in input_ids)

        if target_length is not None and pad_to_multiple_of is not None:
            remainder = target_length % pad_to_multiple_of
            if remainder:
                target_length += pad_to_multiple_of - remainder

        padded_input_ids = []
        attention_masks = []
        for ids in input_ids:
            if target_length is not None:
                ids = ids[:target_length]
                pad_length = target_length - len(ids)
            else:
                pad_length = 0

            padded_input_ids.append(ids + [self.pad_token_id] * pad_length)
            attention_masks.append([1] * len(ids) + [0] * pad_length)

        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_masks,
        }

        for key in encoded_inputs[0]:
            if key in batch:
                continue

            values = [
                item[key].tolist() if isinstance(item[key], torch.Tensor) else list(item[key])
                for item in encoded_inputs
            ]
            if target_length is not None:
                padded_values = []
                for value in values:
                    value = value[:target_length]
                    padded_values.append(value + [self.pad_token_id] * (target_length - len(value)))
                values = padded_values
            batch[key] = values

        if return_tensors == "pt":
            return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

        return batch

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
