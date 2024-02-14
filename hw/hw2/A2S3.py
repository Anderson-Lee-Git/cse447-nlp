from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from datasets import load_dataset

######################################################
#  The following code is given to you.
######################################################

@dataclass
class SST2Example:
    """
    Convert a dict of raw data into an SST2Example object that contains a text and label.
    If you're interested, you can find descriptions of dataclass at https://docs.python.org/3/library/dataclasses.html
    """
    text: str
    label: int  # 0 for negative, 1 for positive

    @staticmethod
    def from_dict(data: dict):
        text = data['text']
        label = data['label']

        return SST2Example(
            text=text,
            label=label,
        )


def initialize_datasets(tokenizer: PreTrainedTokenizerFast) -> dict:
    """
    Initialize the dataset objects for all splits based on the raw data.
    :param tokenizer: A tokenizer used to prepare the inputs for a model (see details in https://huggingface.co/docs/transformers/main_classes/tokenizer).
    :return: A dictionary of the dataset splits.
    """
    raw_data = load_dataset("gpt3mix/sst2")
    split_datasets = {}

    for split_name in raw_data.keys():
        split_data = list(raw_data[split_name])

        split_datasets[split_name] = SST2Dataset(tokenizer, split_data)

    return split_datasets

class SST2Dataset(Dataset):
    """
    Create a customized dataset object for SST-2.
    A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
    You can find a detailed tutorial on Dataset at https://pytorch.org/tutorials/beginner/basics/data_tutorial.html.
    """
    tokenizer: PreTrainedTokenizerFast = None

    def __init__(self, tokenizer: PreTrainedTokenizerFast, raw_data_list: List[dict]):
        SST2Dataset.tokenizer = tokenizer
        self.sample_list = [SST2Example.from_dict(data) for data in raw_data_list]

    def __len__(self):
        """
        Get the number of items in the dataset.
        """
        # TODO: return the number of samples in sample_list.
        return len(self.sample_list)

    def __getitem__(self, idx):
        """
        Get the idx-th item from the dataset.
        """
        # TODO: return the idx-th item in sample_list.
        return self.sample_list[idx]

    def __iter__(self):
        """
        Get an iterator for the dataset.
        """
        # TODO: return an iterator for sample_list.
        return iter(self.sample_list)

    @staticmethod
    def collate_fn(batched_samples: List[SST2Example]) -> dict:
        """
        Encode samples in batched_samples: tokenize the input texts, and turn labels into a tensor.
        :param batched_samples: A list of SST2Example samples.
        :return: A dictionary of encoded texts and their corresponding labels (in tensors).
        """
        # TODO: collect all input texts from batched_samples into a list.
        batched_text = [sample.text for sample in batched_samples]

        # TODO: collect all labels from batched_samples into a list.
        batched_label = [sample.label for sample in batched_samples]

        # Tokenize the input texts.
        text_encoding = SST2Dataset.tokenizer(batched_text,
                                              padding=True,
                                              max_length=512,
                                              truncation=True,
                                              return_tensors="pt")

        # TODO: convert data type of the labels to torch.long (Hint: using torch.LongTensor).
        label_encoding = torch.LongTensor(batched_label)

        # TODO: return dictionary of encoded texts and labels.
        return {
            "text_encoding": text_encoding,
            "label_encoding": label_encoding,
        }

"""
Load train / validation / test dataset, using `initialize_datasets` in `dataset.py`.
"""
# TODO: load pre-trained tokenizer for Roberta-base from transformers library.
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# TODO: load datasets using initialize_datasets.
datasets = initialize_datasets(tokenizer)
dataset_train, dataset_dev, dataset_test = \
    datasets["train"], datasets["validation"], datasets["test"]

# TODO: get the first data point in your validation dataset.
# Hint: (for you to debug) you returned data point should look like `SST2Example(text="It 's a lovely ...", label=0)`
val_first_element = next(iter(dataset_dev))
print(val_first_element)


# TODO: get the length of train, validation, and test datasets using `datasets` variable.
length_train = len(dataset_train)
length_val = len(dataset_dev)
length_test = len(dataset_test)

"""
To load batch of samples from `torch.Dataset` during training / inference, we use `DataLoader` class.
Below, we provide an example of loading a dataloader for the validation split of SST-2.
"""
validation_dataloader = DataLoader(datasets['validation'],
                                   batch_size=64,
                                   shuffle=False,
                                   collate_fn=SST2Dataset.collate_fn)

# TODO: load the first batch of samples from the validation dataset
# Hint: use iterator!
batch = next(iter(validation_dataloader))
print(batch)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, epoch: int):
    """
    Train the model for one epoch.
    :param model: A pre-trained model loaded from transformers. (e.g., RobertaForSequenceClassification https://huggingface.co/docs/transformers/v4.37.0/en/model_doc/roberta#transformers.RobertaForSequenceClassification)
    :param dataloader: A train set dataloader for SST2Dataset.
    :param optimizer: An instance of Pytorch optimizer. (e.g., AdamW https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
    :param epoch: An integer denoting current epoch.
    Trains model for one epoch.
    """
    # TODO: set the model to the training mode.
    model.train()
    losses = []
    all_predictions = []
    all_labels = []
    with tqdm(dataloader, desc=f"Train Ep {epoch}", total=len(dataloader)) as tq:
        for batch in tq:
            # TODO: retrieve the data from your batch and send it to GPU.
            # Hint: model.device should point to 'cuda' as you set it as such in the main function below.
            text_encoding = batch["text_encoding"].to(model.device)
            label_encoding = batch["label_encoding"].to(model.device)
            # TODO: Compute loss by running model with text_encoding and label_encoding.
            out = model(input_ids=text_encoding["input_ids"],
                         attention_mask=text_encoding["attention_mask"],
                         labels=label_encoding)
            loss = out.loss
            logits = out.logits

            # TODO: compute gradients and update parameters using optimizer.
            # Hint: you need three lines of code here!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().item())
            tq.set_postfix({"loss": loss.detach().item()}) # for printing better-looking progress bar

            predictions = torch.argmax(logits, dim=1) # Hint: should be a list [0, 1, ...] of predicted labels
            labels =  label_encoding # Hint: should be a list [0, 1, ...] of ground-truth labels

            all_predictions += predictions
            all_labels += labels
        # compute accuracy
        all_predictions = torch.Tensor(all_predictions)
        all_labels = torch.Tensor(all_labels)
        accuracy = compute_accuracy(all_predictions, all_labels)
    return accuracy, sum(losses) / len(losses)

def evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    """
    Evaluate model on the dataloader and compute the accuracy.
    :param model: A language model loaded from transformers. (e.g., RobertaForSequenceClassification https://huggingface.co/docs/transformers/v4.37.0/en/model_doc/roberta#transformers.RobertaForSequenceClassification)
    :param dataloader: A validation / test set dataloader for SST2Dataset
    :return: A floating number representing the accuracy of model in the given dataset.
    """
    # TODO: set the model to the evaluation mode.
    model.eval()

    all_predictions = []
    all_labels = []
    losses = []
    with tqdm(dataloader, desc=f"Eval", total=len(dataloader)) as tq:
        for batch in tq:
            with torch.no_grad():
                # TODO: retrieve the data from your batch and send it to GPU.
                # Hint: model.device should point to 'cuda' as you set it as such in the main function below.
                text_encoding = batch["text_encoding"].to(model.device)
                label_encoding = batch["label_encoding"].to(model.device)
                # TODO: inference with model and compute logits.
                out =  model(input_ids=text_encoding["input_ids"],
                                attention_mask=text_encoding["attention_mask"],
                                labels=label_encoding) # Hint: logit should be of size (batch_size, 2)
                logits = out.logits
                loss = out.loss

                losses.append(loss.detach().item())

                # TODO: compute list of predictions and list of labels for the current batch
                predictions = torch.argmax(logits, dim=1) # Hint: should be a list [0, 1, ...] of predicted labels
                labels =  label_encoding # Hint: should be a list [0, 1, ...] of ground-truth labels

                all_predictions += predictions
                all_labels += labels

    # compute accuracy
    all_predictions = torch.Tensor(all_predictions)
    all_labels = torch.Tensor(all_labels)
    accuracy = compute_accuracy(all_predictions, all_labels)

    print(f"Accuracy: {accuracy}")
    return accuracy, sum(losses) / len(losses)


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Given two tensors predictions and labels, compute the accuracy.
    :param predictions: torch.Tensor of size (N,)
    :param labels: torch.Tensor of size (N,)
    :return: A floating number representing the accuracy
    """
    assert predictions.size(-1) == labels.size(-1)

    # TODO: compute accuracy
    accuracy = torch.sum(predictions == labels) / len(predictions)
    return accuracy

from transformers import RobertaForSequenceClassification

def test_evaluation(path):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    batch_size = 256

    # TODO: Load your best trained model from `./checkpoints/` and report the test set accuracy.
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    model.load_state_dict(torch.load(path)["model_state_dict"])

    datasets = initialize_datasets(tokenizer)
    # TODO: Load the test dataset
    test_dataloader = DataLoader(dataset=datasets["test"],
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=SST2Dataset.collate_fn)

    # TODO: evaluate the model on the test set
    acc, _ = evaluate(model, test_dataloader)
    print(acc)

def train(config,
          train_dataloader,
          validation_dataloader,
          num_epochs=5, 
          save=False,
          plot=False):
    learning_rate = config["lr"]
    model_name = config["model_name"]
    optimizer = config["optimizer"]

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to("cuda")

    if optimizer == "AdamW":
        optimizer = AdamW(params=model.parameters(), lr=learning_rate, eps=1e-8)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    elif optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(params=model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError()
    
    # training loop.
    best_acc = 0.0
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    for epoch in range(1, num_epochs + 1):
        train_acc, train_loss = train_one_epoch(model, train_dataloader, optimizer, epoch)
        valid_acc, valid_loss = evaluate(model, validation_dataloader)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)
        # TODO: if the newly trained model checkpoint is better than the previously
        if valid_acc > best_acc:
            best_acc = valid_acc
            if save:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_acc": valid_acc
                }, f"./checkpoints/ckpt_best_lr_{config['lr']}_bs_{config['batch_size']}_epochs_{num_epochs}")
        # saved checkpoint, save the new model in `./checkpoints` folder.
        # Hint: remember to update best_acc to the accuracy of the best model so far.
    
    if plot:
        fig = plt.figure()
        plt.plot(train_loss_history, "-o", label="Train")
        plt.plot(valid_loss_history, "-o", label="Validation")
        plt.title(f"Loss curve ({config['optimizer']})")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"./loss_curve_epoch_{num_epochs}_{config['optimizer']}.png")

        fig = plt.figure()
        plt.plot(train_acc_history, "-o", label="Train")
        plt.plot(valid_acc_history, "-o", label="Validation")
        plt.title(f"Accuracy ({config['optimizer']})")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(f"./acc_curve_epoch_{num_epochs}_{config['optimizer']}.png")
    
    return best_acc

def main():
    # hyper-parameters (we provide initial set of values here, but you can modify them.)
    batch_size = 64
    learning_rate = 5e-5
    num_epochs = 20
    model_name = "roberta-base"

    torch.manual_seed(64)

    # TODO: load pre-trained model and corresponding tokenizer (given model_name above).
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # TODO: load datasets.
    datasets = initialize_datasets(tokenizer)

    # TODO: initialize that training and evaluation (validation / test) dataloaders.
    # Hint: you should use the validation dataset during hyperparameter tuning,
    # and evaluate the model on the test set once after you finalize the design choice of your model.
    # Hint: you should shuffle the training data, but not the validation data.
    train_dataloader = DataLoader(dataset=datasets["train"],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=SST2Dataset.collate_fn)
    validation_dataloader = DataLoader(dataset=datasets["validation"],
                                       batch_size=batch_size,
                                       shuffle=False,
                                       collate_fn=SST2Dataset.collate_fn)

    # optimizer experiments
    for optim in ["Adagrad"]:
        train(
            config={"lr": 5e-5, "model_name": model_name, "optimizer": optim},
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            num_epochs=10,
            save=False,
            plot=True
        )

    hyperparams = {
        "batch_size": [64, 128, 256],
        "lr": [1e-4, 5e-5, 1e-5],
        "num_epochs": [2, 5]
    }

    # hyperparameters search
    results = []
    for batch_size in hyperparams["batch_size"]:
        for num_epochs in hyperparams["num_epochs"]:
            for lr in hyperparams["lr"]:
                train_dataloader = DataLoader(dataset=datasets["train"],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=SST2Dataset.collate_fn)
                validation_dataloader = DataLoader(dataset=datasets["validation"],
                                                batch_size=batch_size,
                                                shuffle=False,
                                                collate_fn=SST2Dataset.collate_fn)
                best_acc = train(
                    config={
                        "lr": lr,
                        "model_name": model_name,
                        "optimizer": "Adagrad",
                        "batch_size": batch_size
                    },
                    train_dataloader=train_dataloader,
                    validation_dataloader=validation_dataloader,
                    num_epochs=num_epochs,
                    save=True,
                    plot=False
                )
                results.append({
                    "lr": lr,
                    "model_name": model_name,
                    "optimizer": "Adagrad",
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "best_acc": best_acc
                })
    
    for result in results:
        print(f"bs: {result['batch_size']}, lr: {result['lr']}, num_epochs: {result['num_epochs']}")
        print(f"best acc: {result['best_acc']}")

    test_evaluation(path="/gscratch/scrubbed/lee0618/nlp/checkpoints/ckpt_best_lr_5e-05_bs_128_epochs_5")

# Run the main training loop.
# NOTE: if implemented well, each training epoch will take less than 2 minutes.
main()


