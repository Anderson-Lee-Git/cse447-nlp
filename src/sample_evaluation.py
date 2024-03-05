from dataset import OpenQADataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    accuracy = torch.sum(predictions == labels) / len(predictions)
    return accuracy

@torch.no_grad
def evaluate(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []
    for batch in tqdm(dataloader):
        text_encoding = batch["text_encoding"]
        for k, v in text_encoding.items():
            text_encoding[k] = v.to(model.device)
        label_encoding = batch["label_encoding"].to(model.device)
        out = model(**text_encoding, labels=label_encoding)
        logits = out.logits
        predictions = torch.argmax(logits, dim=1)
        all_predictions += predictions
        all_labels += label_encoding
    all_predictions = torch.Tensor(all_predictions)
    all_labels = torch.Tensor(all_labels)
    accuracy = compute_accuracy(all_predictions, all_labels)
    print(accuracy)

def main():
    device = "cuda"
    model_name = "WizardLM/WizardLM-13B-V1.1"
    model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    OpenQADataset.tokenizer = tokenizer
    dataset_train = OpenQADataset("train")
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=128,
                                  collate_fn=OpenQADataset.collate_fn)
    evaluate(model, dataloader_train)

if __name__ == "__main__":
    
    main()
