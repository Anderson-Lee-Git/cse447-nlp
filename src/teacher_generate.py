import os

from dataset import OpenQADataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def main():
    device = "cuda"
    dataset = OpenQADataset(split="train")
    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            num_workers=5,
                            collate_fn=OpenQADataset.q_collate_fn)
    checkpoint = "dblakely/WizardLM-13B-V1.2-fixed-tokenizer"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=os.environ["TRANSFORMERS_CACHE"]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, 
                                              padding_side="left", 
                                              cache_dir=os.environ["TRANSFORMERS_CACHE"])
    OpenQADataset.tokenizer = tokenizer
    loop = tqdm(total=len(dataset) // dataloader.batch_size, position=0, leave=False)
    for samples in dataloader:
        inputs = samples["text_encoding"].input_ids.to(model.device)
        out = model.generate(inputs=inputs, 
                             max_new_tokens=100, 
                             do_sample=True, 
                             top_k=50,
                             top_p=0.95)
        generation = tokenizer.batch_decode(out, skip_special_tokens=True)
        loop.update(1)



if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = "/gscratch/scrubbed/lee0618/cache/"
    os.environ['HF_HOME'] = "/gscratch/scrubbed/lee0618/cache/"
    main()
