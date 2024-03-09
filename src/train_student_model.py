# Load model directly
import os

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from dataset import OpenQADataset, GeneratedDataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    accuracy = torch.sum(predictions == labels) / len(predictions)
    return accuracy

def train(model: AutoModelForCausalLM,
          optimizer: torch.optim.Adam,
          dataloader: DataLoader,
          criterion: torch.nn.CrossEntropyLoss):
    loop = tqdm(total=len(dataloader), position=0, leave=True)
    losses = []
    for i, samples in enumerate(dataloader):
        text_encoding = samples["text_encoding"].to(model.device)
        attn_mask = text_encoding.attention_mask[:, 1:] # B, L
        output = model(**text_encoding)
        logits = output.logits[:, :-1] # B, L, V
        B, L, V = logits.size()
        logits = logits.reshape(B * L, V)
        attn_mask = attn_mask.reshape(B * L)
        input_ids = text_encoding.input_ids[:, 1:].reshape(B * L)
        # compute loss
        loss = criterion(logits, input_ids)
        loss = torch.mean(loss * attn_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        loop.set_description(f"Loss: {loss.item():.4f}")
        loop.update(1)
    return losses

@torch.no_grad
def evaluate(model, dataloader):
    model.eval()
    all_predictions = torch.empty(0).to(model.device)
    all_labels = torch.empty(0).to(model.device)
    for batch in tqdm(dataloader):
        B = batch["batch_size"]
        text_encoding = batch["text_encoding"] # B, 4 * L
        for k, v in text_encoding.items():
            text_encoding[k] = v.to(model.device)
        label_encoding = batch["label_encoding"].to(model.device)
        input_ids = text_encoding["input_ids"] # B, 4 * L
        B, L4 = input_ids.size()
        out = model(**text_encoding)
        logits = out.logits.view(B, 4, L4 // 4, -1) # B, 4, L, V
        input_ids = input_ids.view(B, 4, L4 // 4, 1) # B, 4, L, 1
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = torch.gather(input=log_probs[:, :, :-1],
                                        dim=-1,
                                        index=input_ids[:, :, 1:]).view(B, 4, L4 // 4 - 1)
        choice_log_probs = torch.mean(chosen_log_probs, dim=-1).view(B, 4)
        predictions = torch.argmax(choice_log_probs, dim=-1)
        all_predictions = torch.cat([all_predictions, predictions])
        all_labels = torch.cat([all_labels, label_encoding])
    accuracy = compute_accuracy(all_predictions, all_labels)
    print(f"accuracy: {accuracy.item()}")

def main():
    torch.manual_seed(42)
    device = 'cuda'
    
    read_path = "/gscratch/scrubbed/lee0618/cse447-nlp/src/data/filtered_text.json"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base",
                                              padding_side="left",
                                              cache_dir=os.environ["TRANSFORMERS_CACHE"])
    model = AutoModelForCausalLM.from_pretrained("roberta-base",
                                                cache_dir=os.environ["TRANSFORMERS_CACHE"]).to(device)
    batch_size = 32
    dataset = GeneratedDataset(read_path)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=5,
                            collate_fn=GeneratedDataset.collate_fn)
    GeneratedDataset.tokenizer = tokenizer

    criterion = CrossEntropyLoss(reduction="none")
    optimizer = Adam(params=model.parameters(), lr=0.0001)

    dataset_validation = OpenQADataset(split="validation")
    dataloader_validation = DataLoader(dataset=dataset_validation,
                                       batch_size=batch_size,
                                       num_workers=5,
                                       collate_fn=OpenQADataset.collate_fn)
    OpenQADataset.tokenizer = tokenizer

    num_epochs = 3
    losses = []
    print("No training evaluation")
    evaluate(model, dataloader_validation)
    print("Train on generated text")
    for i in range(num_epochs):
        losses += train(model, optimizer, dataloader, criterion)
        evaluate(model, dataloader_validation)

if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = "/gscratch/scrubbed/lee0618/cache/"
    os.environ['HF_HOME'] = "/gscratch/scrubbed/lee0618/cache/"
    main()
