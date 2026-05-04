import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import os
import json
import argparse
import pickle
from datasets import load_dataset,concatenate_datasets
from gpt2_eff import *
from chartokenizer import *
import numpy as np
from transformers import DataCollatorForLanguageModeling
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer,max_length=256,pad_token_id=0):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized_text = self.tokenizer.encode(text, return_torch=True)
        # Pad or truncate the tokenized text to the maximum length
        if len(tokenized_text) < self.max_length:
            tokenized_text = torch.cat([tokenized_text, torch.full((self.max_length - len(tokenized_text),), self.pad_token_id)])
            # tokenized_text = torch.cat([tokenized_text,torch.tensor([self.tokenizer.eos_token_id])])
        else:
            tokenized_text = tokenized_text[:self.max_length]
            # tokenized_text = torch.cat([tokenized_text,torch.tensor([self.tokenizer.eos_token_id])])
        return {'input_ids': tokenized_text}


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'labels': labels}

class SimpleTrainer:
    def __init__(self, model, 
                 train_dataloader, 
                 eval_dataloader,
                 tokenizer,
                 sample_prompts,
                 optimizer, 
                 train_log_steps, 
                 eval_steps,
                 gen_max_new_tokens,
                 device
                 ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.sample_prompts = sample_prompts
        self.optimizer = optimizer
        self.device = device
        self.train_log_steps = train_log_steps
        self.eval_steps = eval_steps
        self.gen_max_new_tokens = gen_max_new_tokens

    def train(self, numepochs):
        self.num_epochs = numepochs
        for epoch in range(numepochs):
            self.model.train()
            total_loss = 0

            pbar = tqdm(
                enumerate(self.train_dataloader, start=1),
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch + 1}/{numepochs}",
                unit="step",
            )

            for step, batch in pbar:
                input_ids = batch["input_ids"][:, :-1].to(self.device)
                labels = batch["labels"][:, 1:].to(self.device)

                self.optimizer.zero_grad()

                _, loss = self.model(input_ids, labels=labels)

                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()
                total_loss += current_loss
                avg_loss = total_loss / step

                # Live loss shown inside tqdm progress bar
                pbar.set_postfix(
                    current_loss=f"{current_loss:.4f}",
                    avg_loss=f"{avg_loss:.4f}",
                )

                if step % self.eval_steps == 0:
                    eval_loss = self.evaluate()
                    tqdm.write(
                        f"Epoch [{epoch + 1}/{numepochs}] "
                        f"Step [{step}/{len(self.train_dataloader)}] "
                        f"Evaluation Loss: {eval_loss:.4f}"
                    )
                    self.generate_samples()
                    self.model.train()

            # Print once after each epoch finishes
            epoch_avg_loss = total_loss / len(self.train_dataloader)
            tqdm.write(
                f"Epoch [{epoch + 1}/{numepochs}] done | "
                f"Average Loss: {epoch_avg_loss:.4f}"
            )

    def evaluate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", unit="batch", leave=False):
                input_ids = batch["input_ids"][:, :-1].to(self.device)
                labels = batch["labels"][:, 1:].to(self.device)

                _, loss = self.model(input_ids, labels=labels)
                total_loss += loss.item()

        return total_loss / len(self.eval_dataloader)

    @torch.no_grad()
    def generate_samples(self):
        self.model.eval()
        tqdm.write("\nGeneration samples:")

        samples = random.sample(self.sample_prompts, min(len(self.sample_prompts), 5))

        for prompt in samples:
            tokenized_prompt = self.tokenizer.encode(prompt, return_torch=True).to(self.device)

            generated = self.model.generate(
                {"input_ids": tokenized_prompt.unsqueeze(0)},
                max_new_tokens=self.gen_max_new_tokens,
                use_cache=False,
                temperature=0.4,
                top_k=10,
            )

            full_tokens = torch.cat([tokenized_prompt.cpu(), generated[0].cpu()])
            text = self.tokenizer.decode(full_tokens, skip_special_tokens=True)

            tqdm.write(f"Prompt: {prompt!r} -> {text!r}")

    def save_checkpoint(self,checkpoint_dir,model_name,tokenizer):
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Model configuration to save alongside checkpoint
        model_config = {
            'model_type': model_name,  # MLA version
            'vocab_size': len(tokenizer.vocab),
            'embedding_size': self.model.embedding_size,
            'context_length': self.model.context_length,
            'head_dim': self.model.head_dim,
            'num_heads': self.model.num_heads,
            'num_layers': self.model.num_layers,
            'up_proj_size': self.model.up_proj_size,
            'gqa_factor': self.model.gqa_factor,
            'tied_embedding': self.model.tied_embedding,
            'pad_token_id': tokenizer.vocab[tokenizer.special_tokens['pad_token']],
        }

        optimizer_config = {
                        'optimizer_type': 'Adam',
                        'learning_rate': 5e-4,
                    }

        # Complete checkpoint with all metadata
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': model_config,
            'optimizer_config': optimizer_config,
            'epoch': self.num_epochs,
            'device': str(self.device),
        }

        # Save checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pt'))

        # Save model config as JSON for easy inspection
        with open(os.path.join(checkpoint_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)

        # Save optimizer config as JSON
        with open(os.path.join(checkpoint_dir, 'optimizer_config.json'), 'w') as f:
            json.dump(optimizer_config, f, indent=2)

        # Save tokenizer
        with open(os.path.join(checkpoint_dir, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)

        print(f"✓ Checkpoint saved to {checkpoint_dir}/")
        print(f"  - checkpoint.pt (model + optimizer states + configs)")
        print(f"  - model_config.json")
        print(f"  - optimizer_config.json")
        print(f"  - tokenizer.pkl")

    def load_checkpoint(self, checkpoint_dir,device):
        checkpoint_dir = 'arithmetic_gpteff_checkpoint_7epoch'
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
        with open(os.path.join(checkpoint_dir, 'model_config.json'), 'r') as f:
            model_config = json.load(f)
        with open(os.path.join(checkpoint_dir, 'optimizer_config.json'), 'r') as f:
            optimizer_config = json.load(f)
        with open(os.path.join(checkpoint_dir, 'tokenizer.pkl'), 'rb') as f:
            tokenizer = pickle.load(f)

        # Recreate model from saved config
        model = GPT2Model(
            vocab_size=model_config['vocab_size'],
            embedding_size=model_config['embedding_size'],
            context_length=model_config['context_length'],
            head_dim=model_config['head_dim'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            up_proj_size=model_config['up_proj_size'],
            gqa_factor=model_config['gqa_factor'],
            tied_embedding=model_config['tied_embedding'],
            pad_token_id=model_config['pad_token_id']
        )

        # Load saved state dicts
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # Recreate optimizer and load state
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Resume training info
        print(f"✓ Loaded checkpoint from {checkpoint_dir}/")
        print(f"  Model type: {model_config['model_type']}")
        print(f"  Resumed from epoch: {checkpoint['epoch']}")

        
def process_row(row):
    return {'text': row['input'] + ' Answer:' + row['output'] + ' <EOS>'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the text dataset file, should have input and output fields and should be in a hf dataset format and should have train split")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--train_log_steps", type=int, default=100, help="Number of steps between logging training loss")
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of steps between evaluations")
    parser.add_argument("--gen_max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate during sample generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--context_length", type=int, default=256, help="Context length for the model")

    # filter data
    args = parser.parse_args()
    data = load_dataset(args.dataset_path, split="train").select(range(100))
    logger.info(f"Loaded dataset with {len(data)} samples")
    data = data.train_test_split(test_size=0.1)
    data = data.map(process_row, 
                batch_size=32000)
    
    logger.info(f"Processed dataset, sample processed text: {data['train'][0]['text']}")
    sample_texts = [k.split("=")[0] + '=' for k in data['test']['text'][:1000]]
    data = data.filter(lambda x: len(x['text']) < 256)
    
    logger.info(f"Filtered dataset to remove samples with text length >= 256, remaining samples: {len(data['train'])}")

    # train tokenizer
    texts = list(data['train']['text'])
    tokenizer = CharTokenize()
    tokenizer = tokenizer.train(texts)
    logger.info(f"Trained tokenizer on the dataset, vocab size: {len(tokenizer.vocab)}")

    model = GPT2Model(
    vocab_size=len(tokenizer.vocab),
    embedding_size=128,
    context_length=256,
    head_dim=64,
    num_heads=8,
    num_layers=3,
    up_proj_size=4*512, 
    gqa_factor=1,
    tied_embedding = True,
    pad_token_id = tokenizer.pad_token_id
)
    logger.info(f"Initialized GPT2Model with vocab size {len(tokenizer.vocab)} and context length {args.context_length}")

    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    logger.info(f"Initialized optimizer with learning rate {args.learning_rate}")


    train_dataset = TextDataset(list(data['train']['text']), tokenizer, max_length=128, pad_token_id=tokenizer.pad_token_id)
    eval_dataset = TextDataset(list(data['test']['text']), tokenizer, max_length=128, pad_token_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    logger.info(f"Initialized dataloaders with train batches: {len(train_dataloader)}, eval batches: {len(eval_dataloader)}")

    trainer = SimpleTrainer(
                    model=model,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_dataloader,
                    tokenizer=tokenizer,
                    sample_prompts=sample_texts,
                    optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
                    train_log_steps=args.train_log_steps,
                    eval_steps=args.eval_steps,
                    device=args.device,
                    gen_max_new_tokens=args.gen_max_new_tokens
                )
    logger.info("Starting training...")
    trainer.train(args.epochs)
    logger.info("Training completed.")
    
    trainer.save_checkpoint('dummy_epoch','gpt2_eff',tokenizer)
    logger.info("Checkpoint saved.")






    
    



