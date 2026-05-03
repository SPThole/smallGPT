import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random


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
                 device):
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
