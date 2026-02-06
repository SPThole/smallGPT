# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch


# def speculative_decoding(input_text,tokenizer,main_model,assistive_model,speculation,max_new_tokens):
#     tokenized_text = tokenizer(input_text)
#     input_ids = tokenized_text['input_ids']
#     len_input = len(input_ids)
#     generated_tokens = 0

#     while generated_tokens<max_new_tokens:
#         # print(input_ids)
#         generated_assist_tokens = assistive_model.generate(input_ids = torch.tensor(input_ids).unsqueeze(0), 
#                 attention_mask = torch.tensor([[1]*len(input_ids)]),
#                                 max_new_tokens=speculation,)[0][-speculation:]
#         # print("generated_assist_tokens,",generated_assist_tokens)
#         assisted_input = input_ids + generated_assist_tokens.tolist()
#         assisted_main_output = main_model(input_ids = torch.tensor(assisted_input).unsqueeze(0),
#                 attention_mask = torch.tensor([[1]*len(assisted_input)]))
#         # print("assisted_input",assisted_input,assisted_main_output.logits[0].shape)
        
#         assisted_main_output = torch.argmax(assisted_main_output.logits[0][len(input_ids)-1:-1],dim=-1)
#         # print("assisted_main_output",assisted_main_output)
#         accpeted = 0
#         for k in range(len(assisted_main_output)):
#             if assisted_main_output[k]==generated_assist_tokens[k]:
#                 # print('token accepted',assisted_main_output[k])
#                 input_ids = input_ids + [generated_assist_tokens[k].item()]
#                 generated_tokens = generated_tokens+1
#                 accpeted = accpeted + 1
#             else:
#                 input_ids = input_ids + [assisted_main_output[k].item()]
#                 generated_tokens = generated_tokens+1
#                 # print('token rejected',assisted_main_output[k],'instead of',generated_assist_tokens[k])
#                 break
#         # print('current generated length',generated_tokens, 'accepted tokens percentage', accpeted/generated_tokens if generated_tokens > 0 else 0)
    
#     return input_ids[len_input:]
    

# if __name__ == "__main__":
#     import time
#     tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")  
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     main_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
    
#     assistive_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
#     s = time.time()
#     input_text = "India is a country where,"
#     out = speculative_decoding(input_text,tokenizer,main_model,assistive_model,2,100) 
#     print(tokenizer.decode(out))
#     print('time take by speculation',time.time()-s)

#     s = time.time()
#     out = main_model.generate(input_ids = torch.tensor(tokenizer(input_text)['input_ids']).unsqueeze(0),
#                 attention_mask = torch.tensor([[1]*len(tokenizer(input_text)['input_ids'])]),max_new_tokens=100)[0]
#     print(tokenizer.decode(out))
#     print('time taken by direct decoding',time.time()-s)
    
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def speculative_decoding(input_text, tokenizer, main_model, assistive_model, 
                         speculation, max_new_tokens):
    tokenized_text = tokenizer(input_text, return_tensors="pt")
    input_ids = tokenized_text['input_ids']
    len_input = input_ids.shape[1]
    generated_tokens = 0
    
    # Initialize KV caches
    main_past_kv = None
    
    with torch.no_grad():
        # Prefill: run main model on the prompt to build its KV cache
        prefill_output = main_model(
            input_ids=input_ids, 
            use_cache=True
        )
        main_past_kv = prefill_output.past_key_values
        
        while generated_tokens < max_new_tokens:
            remaining = max_new_tokens - generated_tokens
            current_speculation = min(speculation, remaining)
            
            # --- Draft phase: use assistive model to propose tokens ---
            draft_tokens = assistive_model.generate(
                input_ids=input_ids,
                max_new_tokens=current_speculation,
            )[0, input_ids.shape[1]:]  # only new tokens
            
            num_draft = len(draft_tokens)
            if num_draft == 0:
                break
                
            draft_tokens = draft_tokens.unsqueeze(0)  # [1, K]
            
            # --- Verify phase: run main model on draft tokens using cache ---
            verify_output = main_model(
                input_ids=draft_tokens,
                past_key_values=main_past_kv,
                use_cache=True,
            )
            # Logits shape: [1, K, vocab_size]
            # Position 0 predicts what comes after last accepted token
            # Position i predicts what comes after draft_tokens[i]
            verify_tokens = torch.argmax(verify_output.logits[0], dim=-1)  # [K]
            
            new_past_kv = verify_output.past_key_values
            
            # --- Accept/reject ---
            accepted = 0
            for k in range(num_draft):
                if verify_tokens[k] == draft_tokens[0, k]:
                    accepted += 1
                    generated_tokens += 1
                else:
                    # Reject: take the main model's token instead
                    generated_tokens += 1
                    break
            
            if accepted == num_draft:
                # All accepted — append all draft tokens
                new_tokens = draft_tokens[0].tolist()
                input_ids = torch.cat([
                    input_ids, draft_tokens
                ], dim=1)
                # Cache is already correct for all positions
                main_past_kv = new_past_kv
                
            else:
                # Partial acceptance
                accepted_tokens = draft_tokens[0, :accepted].tolist()
                rejected_replacement = [verify_tokens[accepted].item()]
                new_tokens = accepted_tokens + rejected_replacement
                
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([new_tokens]).long()
                ], dim=1)
                
                # Trim KV cache to only include accepted positions
                # Keep original cache + accepted positions (not the rejected ones)
                main_past_kv = _trim_kv_cache(
                    new_past_kv, 
                    keep_length=len_input + generated_tokens - 1
                )
                # Re-run the replacement token through the model
                replacement_tensor = torch.tensor([[rejected_replacement[0]]])
                rerun = main_model(
                    input_ids=replacement_tensor,
                    past_key_values=main_past_kv,
                    use_cache=True,
                )
                main_past_kv = rerun.past_key_values
    
    return input_ids[0, len_input:].tolist()


def _trim_kv_cache(past_kv, keep_length):
    """Trim KV cache to keep only first `keep_length` positions."""
    trimmed = []
    for layer_k, layer_v in past_kv:
        trimmed.append((
            layer_k[:, :, :keep_length, :],
            layer_v[:, :, :keep_length, :],
        ))
    return tuple(trimmed)


if __name__ == "__main__":
    import time
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    main_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
    assistive_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    main_model.eval()
    assistive_model.eval()
    
    input_text = "India is a country where,"
    
    with torch.no_grad():
        s = time.time()
        out = speculative_decoding(
            input_text, tokenizer, main_model, assistive_model, 
            speculation=5, max_new_tokens=100
        )
        print(tokenizer.decode(out))
        print('time taken by speculation', time.time() - s)
        
        s = time.time()
        input_ids = tokenizer(input_text, return_tensors="pt")['input_ids']
        out = main_model.generate(input_ids=input_ids, max_new_tokens=100)[0]
        print(tokenizer.decode(out))
        print('time taken by direct decoding', time.time() - s)
