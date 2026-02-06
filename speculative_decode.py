from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def speculative_decoding(input_text,tokenizer,main_model,assistive_model,speculation,max_new_tokens):
    tokenized_text = tokenizer(input_text)
    input_ids = tokenized_text['input_ids']
    len_input = len(input_ids)
    generated_tokens = 0

    while generated_tokens<max_new_tokens:
        print(input_ids)
        generated_assist_tokens = assistive_model.generate(input_ids = torch.tensor(input_ids).unsqueeze(0), 
                                max_new_tokens=speculation,)[0][-speculation:]
        print("generated_assist_tokens,",generated_assist_tokens)
        assisted_input = input_ids + generated_assist_tokens.tolist()
        assisted_main_output = main_model(input_ids = torch.tensor(assisted_input).unsqueeze(0),
                attention_mask = torch.tensor([[1]*len(assisted_input)]))
        print("assisted_input",assisted_input,assisted_main_output.logits[0].shape)
        
        assisted_main_output = torch.argmax(assisted_main_output.logits[0][len(input_ids)-1:-1],dim=-1)
        print("assisted_main_output",assisted_main_output)
        accpeted = 0
        for k in range(len(assisted_main_output)):
            if assisted_main_output[k]==generated_assist_tokens[k]:
                print('token accepted',assisted_main_output[k])
                input_ids = input_ids + [generated_assist_tokens[k].item()]
                generated_tokens = generated_tokens+1
                accpeted = accpeted + 1
            else:
                input_ids = input_ids + [assisted_main_output[k].item()]
                generated_tokens = generated_tokens+1
                print('token rejected',assisted_main_output[k],'instead of',generated_assist_tokens[k])
                break
        print('current generated length',generated_tokens, 'accepted tokens percentage', accpeted/generated_tokens if generated_tokens > 0 else 0)
    
    return input_ids[len_input:]
    

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")  
    tokenizer.pad_token = tokenizer.eos_token
    main_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
    assistive_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
    out = speculative_decoding("twinkle twinkle little",tokenizer,main_model,assistive_model,2,30) 
    print(tokenizer.decode(out))
    
