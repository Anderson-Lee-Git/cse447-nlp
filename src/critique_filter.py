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
    
    read_path = "/gscratch/scrubbed/lee0618/cse447-nlp/src/data/raw_gen_text_128.json"
    #file to store filtered text
    write_to_path = "/gscratch/scrubbed/lee0618/cse447-nlp/src/data/filtered_text_128.json"
    write_file = open(write_to_path, "w")
    filtered_text = {"data": []}
    
    tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/roberta-large-faithcritic",
                                              padding_side="left",
                                              cache_dir=os.environ["TRANSFORMERS_CACHE"])
    model = AutoModelForSequenceClassification.from_pretrained("McGill-NLP/roberta-large-faithcritic",
                                                               cache_dir=os.environ["TRANSFORMERS_CACHE"]).to(device)
    batch_size = 2
    dataset = GeneratedDataset(read_path)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=5,
                            collate_fn=GeneratedDataset.fact_check_collate_fn)
    GeneratedDataset.tokenizer = tokenizer

    # not super familiar with loop so def check if this is correct
    loop = tqdm(total=len(dataset) // dataloader.batch_size, position=0, leave=False)
    total_acc = []
    for i, samples in enumerate(dataloader):
        inputs = samples["text_encoding"].to(device)
        outputs = model(**inputs)
        pred = torch.argmax(torch.softmax(outputs.logits, dim=1), dim=1)
        for j, p in enumerate(pred):
            if p == 1:
                filtered_text["data"].append(samples["text"][j])
        batch_acc = torch.sum(pred) / samples["batch_size"]
        total_acc.append(batch_acc.item())
        loop.update(1)
        loop.set_description(f"Accuracy: {sum(total_acc) / len(total_acc)}")
    print(f"total accuracy: {sum(total_acc) / len(total_acc)}")
    json.dump(filtered_text, write_file)

#just copied it over
if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = "/gscratch/scrubbed/lee0618/cache/"
    os.environ['HF_HOME'] = "/gscratch/scrubbed/lee0618/cache/"
    main()
