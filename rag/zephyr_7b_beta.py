import json, os
import torch
from typing import Any, Generator, List, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
torch.set_default_dtype(torch.float16) 
import sys
import re

from llama_index.schema import Document

from huggingface_hub import login

STAGING = False

def save_list_to_json(lst, filename):
    """ Save Files """
    with open(filename, 'w') as file:
        json.dump(lst, file)

def rm_file(file_path):
    """ Delete Files """
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} removed successfully.")

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message['role'] == 'system':
            prompt += f"<|system|>\n{message['content']}\n"
        elif message['role'] == 'user':
            prompt += f"<|user|>\n{message['content']}\n"
        elif message['role'] == 'assistant':
            prompt += f"<|assistant|>\n{message['content']}\n"
    # Ensure we start with a system prompt
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n\n" + prompt
    # Add the final assistant prompt
    prompt += "<|assistant|>\n"
    return prompt

def run_query(tokenizer, model, messages, temperature=0.1, max_new_tokens=512, **kwargs):
    # Convert messages to prompt
    prompt = messages_to_prompt(messages)
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    
    # Set up generation configuration
    generation_config = GenerationConfig(do_sample=True, temperature=temperature, **kwargs)
    
    # Generate the response
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens
        )
        
        # Decode the generated response
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=False)
        
        # Extract the assistant's response
        if "<|assistant|>\n" in output:
            response = output.split("<|assistant|>\n")[-1].strip()
        else:
            response = output[len(prompt):].strip()
        response = extract_answer(response)
        return response

def extract_answer(text):
    # Use regex to find the word 'Answer:' followed by the actual answer
    match = re.search(r'Answer:\s*([^\n.,(]*)', text, re.DOTALL)
    
    if match:
        s = match.group(1).strip()
        if "Insufficient information" in s:
            return "Insufficient information"
        else:
            return s
    else:
        return "Insufficient Information"

def initialise_and_run_model(save_name, input_stage_1, model_name, sample_size):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        offload_folder="./offload"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto"
    )

    prefix = """Below is a question followed by some context from different sources. 
    Extract the information from the presented texts.
    Please answer the question with concise answer that is strictly limited to either a named entity (in full), or a 'Yes' or 'No' response. 
    If the provided information is insufficient to answer the question, 
    respond 'Insufficient Information'.
    The response must start with "Answer: "answer" ... ." """

    print('Loading Stage 1 Ranking')
    with open(input_stage_1, 'r') as file:
        doc_data = json.load(file)

    print('Remove saved file if exists.')
    rm_file(save_name)

    num_samples = int(len(doc_data) * sample_size)
    doc_data = doc_data[:num_samples]

    save_list = []
    for d in tqdm(doc_data):
        retrieval_list = d['retrieval_list']
        context = '--------------'.join(e['text'] for e in retrieval_list)
        messages = [
            {"role": "system", "content": prefix},
            {"role": "user", "content": f"Question: {d['query']}\n\nContext:\n\n{context}"}
        ]
        response = run_query(tokenizer, model, messages)
        save = {
            'query': d['query'],
            'prompt': messages_to_prompt(messages),
            'model_answer': response,
            'gold_answer': d['answer'],
            'question_type': d['question_type']
        }
        save_list.append(save)

    # Save Results
    print('Query processing completed. Saving the results.')  
    save_list_to_json(save_list, save_name)

if __name__ == '__main__':
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    input_stage_1 = "output/baai_llm_embedder_reranker.json"
    output_file = "output/zephyr_7b_beta.json"
    sample_size = 1
    
    initialise_and_run_model(output_file, input_stage_1, model_name, sample_size)
