# Load model directly
import os

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from dataset import OpenQADataset, GeneratedDataset

import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification



# TODO: Sequence classification
# TODO: test this since I was unable to run it
@torch.no_grad()
def main():
  device = 'cuda'
  
  read_path = "/gscratch/scrubbed/lee0618/cse447-nlp/src/data/raw_gen_text.json"
  dataset = GeneratedDataset.initialize_data(read_path)
  batch_size = 32
  dataloader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          num_workers=5,
                          collate_fn=GeneratedDataset.collate_fn)
  #Need context - ie. prompt, abstract, or something like that and I wasn't sure what the structure of the data
  #in raw_gen_data is, i think we decided to store it together but i'm not sure what format it's in
  qa_dataset = OpenQADataset(split="train")

  #file to store filtered text
  write_to_path = "/gscratch/scrubbed/lee0618/cse447-nlp/src/data/filtered_text.json"
  write_file = open(write_to_path, "w")
  
  tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/roberta-large-faithcritic")
  model = AutoModelForSequenceClassification.from_pretrained("McGill-NLP/roberta-large-faithcritic")
  GeneratedDataset.tokenizer = tokenizer

  #not super familiar with loop so def check if this is correct
  loop = tqdm(total=len(dataset) // dataloader.batch_size, position=0, leave=False)
  for i, samples in enumerate(dataloader):
    #pretty sure this is not correct: need format -> input = tokenizer(knowledge, response), so i'm thinking (prompt, gen)
    #but i'm not sure what the structure of raw_gen is so wasn't sure, will edit it --> wrote it this way for clarity
    prompts, raw_gens = samples['prompt'], samples['raw_gen']
    inputs = tokenizer(prompts, raw_gens, return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)
    outputs = model(**inputs)
    scores = [item["entailment"] for item in outputs]
    for prompt, raw_gen, score in zip(prompts, raw_gens, scores):
      if(score > 0.5):
        json.dump({"prompt": prompt, "generated_response": gen}, write_file)
        write_file.write('\n')
    loop.update(1)

#just copied it over
if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = "/gscratch/scrubbed/lee0618/cache/"
    os.environ['HF_HOME'] = "/gscratch/scrubbed/lee0618/cache/"
    main()    
