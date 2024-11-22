import json, os
from tqdm import tqdm 
from copy import deepcopy
from rank_bm25 import BM25Okapi
from typing import Any, Generator, List, Dict, Optional
from nltk.tokenize import sent_tokenize

import nltk
nltk.download('punkt_tab')

from llama_index.extractors import BaseExtractor
from llama_index.schema import Document


# This is the staging flag. Set to False if you want to run on the real
# collection.

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


# The two classes are used to parse the json corpus and queries.
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
    

class CustomExtractor(BaseExtractor):
  async def aextract(self, nodes) -> List[Dict]:
    metadata_list = [
      {
        "title": (node.metadata["title"]),
        "source": (node.metadata["source"]),      
        "published_at": (node.metadata["published_at"])
      } for node in nodes
    ]
    return metadata_list
  
def bm25_rank(corpus, queries, output_name):
    print('Remove saved file if exists.')
    rm_file(output_name)
    
    # Read the corpus and queries
    reader = JSONReader()
    data = reader.load_data(corpus)

    # Split documents into smaller chunks (sentences) for better relevance
    corpus_sentences = []
    sentence_map = []
    for doc_index, doc in enumerate(data):
        sentences = sent_tokenize(doc.text)
        for sentence in sentences:
            corpus_sentences.append(sentence)
            sentence_map.append(doc_index)  # Map sentences to their original document

    tokenized_corpus = [sent.split() for sent in corpus_sentences]

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_corpus)

    # Prepare query processing
    with open(queries, 'r') as file:
        query_data = json.load(file)

    print('Query Data')
    print('--------------------------')
    print(query_data[0])
    print('--------------------------')

    retrieval_save_list = []
    print("Running BM25 Retrieval ...")
    
    # Iterate through each query and retrieve top sentences
    for data_item in tqdm(query_data):
        query = data_item['query']
        tokenized_query = query.split()

        # BM25 ranking
        scores = bm25.get_scores(tokenized_query)
        top_n_indexes = scores.argsort()[::-1][:10]  # Get top 10 results

        retrieval_list = []
        for index in top_n_indexes:
            if index < len(corpus_sentences):  # Ensure index is within range of corpus
                sentence = corpus_sentences[index]
                doc_index = sentence_map[index]  # Find original document index
                doc = data[doc_index]  # Retrieve the document metadata
                
                dic = {
                    'text': f"[Excerpt from document]\ntitle: {doc.metadata['title']}\npublished_at: {doc.metadata['published_at']}\nsource: {doc.metadata['source']}\nExcerpt:\n-----\n{sentence}",  # Excerpt is now the relevant sentence
                    'score': scores[index]
                }
                retrieval_list.append(dic)
            else:
                print(f"Warning: Index {index} is out of bounds for data length {len(data)}.")

        # Save the retrieval results
        save = {
            'query': data_item['query'],
            'answer': data_item['answer'],
            'question_type': data_item['question_type'],
            'retrieval_list': retrieval_list,
            'gold_list': data_item['evidence_list']
        }
        retrieval_save_list.append(save)

    # Save the BM25 retrieval results
    print('BM25 Retrieval complete. Saving Results')
    with open(output_name, 'w') as json_file:
        json.dump(retrieval_save_list, json_file)

if __name__ == '__main__':
    corpus = "data/corpus.json"
    queries = "data/rag.json"
    output_name = "output/bm25_ranking.json"
    bm25_rank(corpus, queries, output_name)
