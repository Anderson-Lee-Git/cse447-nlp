import os

from dataset import OpenQADataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import json

@torch.no_grad()
def main():
    device = "cuda"
    write_to_path = "/gscratch/scrubbed/lee0618/cse447-nlp/src/data/raw_gen_text_96.json"
    write_file = open(write_to_path, "w")
    batch_size = 1
    dataset = OpenQADataset(split="train")
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=5,
                            collate_fn=OpenQADataset.q_collate_fn)
    checkpoint = "dblakely/WizardLM-13B-V1.2-fixed-tokenizer"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=os.environ["TRANSFORMERS_CACHE"]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, 
                                              padding_side="left", 
                                              cache_dir=os.environ["TRANSFORMERS_CACHE"])
    OpenQADataset.tokenizer = tokenizer
    loop = tqdm(total=len(dataset) // dataloader.batch_size, position=0, leave=False)
    output = {}
    for i, samples in enumerate(dataloader):
        inputs = samples["text_encoding"].input_ids.to(model.device)
        out = model.generate(inputs=inputs, 
                             max_new_tokens=96, 
                             do_sample=True,
                             temperature=0.9)
        generation = tokenizer.batch_decode(out, skip_special_tokens=True)
        output[i] = generation[0]
        loop.update(1)
    json.dump(output, write_file)

if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = "/gscratch/scrubbed/lee0618/cache/"
    os.environ['HF_HOME'] = "/gscratch/scrubbed/lee0618/cache/"
    main()
