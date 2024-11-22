import json, os
import torch
from typing import Any, Generator, List, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
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

def wr_dict(filename,dic):
  """ Write Files """
  try:
    if not os.path.isfile(filename):
      data = []
      data.append(dic)
      with open(filename, 'w') as f:
        json.dump(data, f)
    else:      
      with open(filename, 'r') as f:
        data = json.load(f)
        data.append(dic)
      with open(filename, 'w') as f:
          json.dump(data, f)
  except Exception as e:
    print("Save Error:", str(e))
  return
            
def rm_file(file_path):
  """ Delete Files """
  if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File {file_path} removed successfully.")

def _depth_first_yield(json_data: Any, levels_back: int, collapse_length: 
                       Optional[int], path: List[str], ensure_ascii: bool = False,
                      ) -> Generator[str, None, None]:
  """ Do depth first yield of all of the leaf nodes of a JSON.
      Combines keys in the JSON tree using spaces.
      If levels_back is set to 0, prints all levels.
      If collapse_length is not None and the json_data is <= that number
      of characters, then we collapse it into one line.
  """
  if isinstance(json_data, (dict, list)):
    # only try to collapse if we're not at a leaf node
    json_str = json.dumps(json_data, ensure_ascii=ensure_ascii)
    if collapse_length is not None and len(json_str) <= collapse_length:
      new_path = path[-levels_back:]
      new_path.append(json_str)
      yield " ".join(new_path)
      return
    elif isinstance(json_data, dict):
      for key, value in json_data.items():
        new_path = path[:]
        new_path.append(key)
        yield from _depth_first_yield(value, levels_back, collapse_length, new_path)
    elif isinstance(json_data, list):
      for _, value in enumerate(json_data):
        yield from _depth_first_yield(value, levels_back, collapse_length, path)
    else:
      new_path = path[-levels_back:]
      new_path.append(str(json_data))
      yield " ".join(new_path)


class JSONReader():
  """JSON reader.
     Reads JSON documents with options to help suss out relationships between nodes.
  """
  def __init__(self, is_jsonl: Optional[bool] = False,) -> None:
    """Initialize with arguments."""
    super().__init__()
    self.is_jsonl = is_jsonl

  def load_data(self, input_file: str) -> List[Document]:
    """Load data from the input file."""
    documents = []
    with open(input_file, 'r') as file:
      load_data = json.load(file)
    for data in load_data:
      metadata = {"title": data['title'], 
                  "published_at": data['published_at'],
                  "source":data['source']}
      documents.append(Document(text=data['body'], metadata=metadata))
    return documents

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

def run_query(tokenizer, model, messages, temperature=0.1, max_new_tokens=50, **kwargs,):
  # messages = [ {"role": "user", "content": messages}, ]
  # prompt = messages_to_prompt(messages)
  input_ids = tokenizer(messages, return_tensors="pt").input_ids.cuda()
  generation_config = GenerationConfig(do_sample=True, temperature=temperature,
                                       **kwargs,)
  with torch.no_grad():
    generation_output = model.generate(
                      input_ids=input_ids,
                      generation_config=generation_config,
                      #pad_token_id=tokenizer.unk_token_id,
                      pad_token_id=tokenizer.eos_token_id,
                      return_dict_in_generate=True,
                      output_scores=True,
                      max_new_tokens=max_new_tokens,
                      )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    # Use the commented print below to determing the split token and any other random
    # text that you will need to remove to ensure that exact match evaluation works.
    # Extract the assistant's response
    if "<|end_of_turn|> GPT4 Correct Assistant:" in output:
        response = output.split("<|end_of_turn|> GPT4 Correct Assistant:")[-1].strip()
    else:
        response = output[len(prompt):].strip()
    response = extract_answer(response)
    # response = output.split("[/INST]")[-1].strip()
    # response = response.replace(r".</s>", "")
    # response = response.replace(r"</s>", "")
    return response

def initialise_and_run_model(save_name, input_stage_1, model_name, sample_size):

  model = AutoModelForCausalLM.from_pretrained(model_name,
                                               device_map="auto",
                                               offload_folder="./offload")

  tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            device_map="auto",
                                            padding_side="left")

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
      
    prompt = f"{prefix}\n\nQuestion: {d['query']}\n\nContext:\n\n{context}"
    prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: "
    response = run_query(tokenizer, model, prompt)
    #print(response)
    save = {}
    save['query'] = d['query']
    save['prompt'] = prompt
    save['model_answer'] = response
    save['gold_answer'] = d['answer']
    save['question_type'] = d['question_type']
    #print(save)
    save_list.append(save)

  # Save Results
  print ('Query processing completed. Saving the results.')  
  save_list_to_json(save_list,save_name)



if __name__ == '__main__':
  rerank = False
  login(token="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", add_to_git_credential=True)

  model_name = "berkeley-nest/Starling-LM-7B-alpha"
  input_stage_1 = "output/baai_llm_embedder_reranker.json"
  output_file = "output/starling_7b_alpha.json"
  sample_size = 1
    
  initialise_and_run_model(output_file, input_stage_1, model_name, sample_size)

