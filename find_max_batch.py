import torch
import gc
from gpt2_eff import GPT2Model

def find_max_batch():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")

    # Model configuration from scratch.ipynb
    config = {
        "vocab_size": 8192,
        "embedding_size": 768,
        "context_length": 128,
        "head_dim": 64,
        "num_heads": 12,
        "num_layers": 12,
        "up_proj_size": 3072,
        "gqa_factor": 1,
        "tied_embedding": True
    }

    try:
        model = GPT2Model(**config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Exponential search for batch size
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    last_success = 0

    for b in batch_sizes:
        print(f"\nTesting batch size: {b}...", end=" ", flush=True)
        try:
            # Create dummy data
            input_ids = torch.randint(0, config["vocab_size"], (b, config["context_length"])).to(device)
            labels = input_ids.clone()

            # Shift for causal LM (matching our trainer logic)
            input_ids_shifted = input_ids[:, :-1]
            labels_shifted = labels[:, 1:]

            # Forward pass
            model.train()
            optimizer.zero_grad()
            logits, loss = model(input_ids_shifted, labels=labels_shifted)

            # Backward pass
            loss.backward()
            optimizer.step()

            print("Success!")
            last_success = b
            
            # Cleanup to avoid cumulative memory usage
            del input_ids, labels, input_ids_shifted, labels_shifted, logits, loss
            if device == "mps":
                torch.mps.empty_cache()
            gc.collect()

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "allocat" in msg:
                print(f"OOM (Out Of Memory) at batch size {b}")
            else:
                print(f"Error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    print(f"\nRecommended Maximum Batch Size: {last_success}")
    if last_success >= 64:
        print("Note: Fast training is likely, but monitor system 'Memory Pressure' in Activity Monitor.")

if __name__ == "__main__":
    find_max_batch()
