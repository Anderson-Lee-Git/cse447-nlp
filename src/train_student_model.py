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
import matplotlib.pyplot as plt

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    accuracy = torch.sum(predictions == labels) / len(predictions)
    return accuracy

def train_distill(model: AutoModelForCausalLM,
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
        choice_log_probs = torch.sum(chosen_log_probs, dim=-1).view(B, 4)
        predictions = torch.argmax(choice_log_probs, dim=-1)
        all_predictions = torch.cat([all_predictions, predictions])
        all_labels = torch.cat([all_labels, label_encoding])
    accuracy = compute_accuracy(all_predictions, all_labels)
    print(f"Validation accuracy: {accuracy.item()}")
    return accuracy.item()

def train_multiple_choice(model: AutoModelForCausalLM,
                         optimizer: torch.optim.Adam,
                         dataloader: DataLoader,
                         criterion: torch.nn.CrossEntropyLoss):
    losses = []
    all_predictions = torch.empty(0).to(model.device)
    all_labels = torch.empty(0).to(model.device)
    loop = tqdm(total=len(dataloader), leave=True, position=0)
    for batch in dataloader:
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
        choice_log_probs = torch.log_softmax(torch.sum(chosen_log_probs, dim=-1).view(B, 4), dim=-1) # B, 4
        predictions = torch.argmax(choice_log_probs, dim=-1)
        one_hot_labels = torch.nn.functional.one_hot(label_encoding, num_classes=4) # B, 4
        loss = choice_log_probs * one_hot_labels
        loss = torch.mean(-torch.sum(loss, dim=1), dim=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_predictions = torch.cat([all_predictions, predictions])
        all_labels = torch.cat([all_labels, label_encoding])
        loop.set_description(f"Loss: {loss.item():.4f}")
        loop.update(1)
        losses.append(loss.item())
    accuracy = compute_accuracy(all_predictions, all_labels)
    print(f"Train accuracy: {accuracy.item()}")
    return losses

def main():
    torch.manual_seed(32)
    device = 'cuda'
    
    read_path = "/gscratch/scrubbed/lee0618/cse447-nlp/src/data/filtered_text.json"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base",
                                              padding_side="left",
                                              cache_dir=os.environ["TRANSFORMERS_CACHE"])
    
    batch_size = 64
    dataset = GeneratedDataset(read_path)
    dataloader_distill = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    num_workers=5,
                                    collate_fn=GeneratedDataset.collate_fn)
    GeneratedDataset.tokenizer = tokenizer

    dataset_train = OpenQADataset(split="train")
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=batch_size,
                                  num_workers=5,
                                  collate_fn=OpenQADataset.collate_fn)

    dataset_validation = OpenQADataset(split="validation")
    dataloader_validation = DataLoader(dataset=dataset_validation,
                                       batch_size=batch_size,
                                       num_workers=5,
                                       collate_fn=OpenQADataset.collate_fn)
    OpenQADataset.tokenizer = tokenizer

    num_epochs_distill = 3
    num_epochs_ft = 3
    lrs = [5e-4, 1e-4, 5e-5, 1e-5]
    # key: lr, value: list of losses
    losses_distill = {}
    losses_ft = {}
    accs_distill = {}
    accs_ft = {}
    criterion = CrossEntropyLoss(reduction="none")
    for lr in lrs:
        losses = []
        accs = []
        model = AutoModelForCausalLM.from_pretrained("roberta-base",
                                                     cache_dir=os.environ["TRANSFORMERS_CACHE"]).to(device)
        optimizer = Adam(params=model.parameters(), lr=lr)
        # print("No training evaluation")
        # evaluate(model, dataloader_validation)
        print("Train on generated text (distillation)")
        for i in range(num_epochs_distill):
            losses += train_distill(model, optimizer, dataloader_distill, criterion)
            accs.append(evaluate(model, dataloader_validation))
        losses_distill[lr] = losses
        accs_distill[lr] = accs
        losses = []
        accs = []
        print("Fine-tune on OpenbookQA multiple choice")
        for i in range(num_epochs_ft):
            losses += train_multiple_choice(model, optimizer, dataloader_train, criterion)
            accs.append(evaluate(model, dataloader_validation))
        losses_ft[lr] = losses
        accs_ft[lr] = accs
    # distillation loss
    fig = plt.figure()
    for lr in lrs:
        plt.plot(range(len(losses_distill[lr])), losses_distill[lr], label=f"lr={lr}")
    plt.title("Train loss on filtered generated text (distill)")
    plt.legend()
    plt.savefig("/gscratch/scrubbed/lee0618/cse447-nlp/src/plots/loss_distill.png")
    # ft loss
    fig = plt.figure()
    for lr in lrs:
        plt.plot(range(len(losses_ft[lr])), losses_ft[lr], label=f"lr={lr}")
    plt.title("Finetune training loss on OpenbookQA")
    plt.legend()
    plt.savefig("/gscratch/scrubbed/lee0618/cse447-nlp/src/plots/loss_ft.png")
    # distillation acc
    fig = plt.figure()
    for lr in lrs:
        plt.plot(range(len(accs_distill[lr])), accs_distill[lr], linestyle='-', marker='o', label=f"lr={lr}")
    plt.title("Validation accuracy on filtered generated text (distill)")
    plt.legend()
    plt.savefig("/gscratch/scrubbed/lee0618/cse447-nlp/src/plots/acc_distill.png")
    # ft acc
    fig = plt.figure()
    for lr in lrs:
        plt.plot(range(len(accs_ft[lr])), accs_ft[lr], linestyle='-', marker='o', label=f"lr={lr}")
    plt.title("Finetune validation accuracy on OpenbookQA")
    plt.legend()
    plt.savefig("/gscratch/scrubbed/lee0618/cse447-nlp/src/plots/acc_ft.png")

    # save to json
    json.dump({
        "losses_distill": losses_distill,
        "losses_ft": losses_ft,
        "accs_distill": accs_distill,
        "accs_ft": accs_ft
    }, open(f"/gscratch/scrubbed/lee0618/cse447-nlp/src/data/loss_acc_stat.json", "w"))


if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = "/gscratch/scrubbed/lee0618/cache/"
    os.environ['HF_HOME'] = "/gscratch/scrubbed/lee0618/cache/"
    main()
