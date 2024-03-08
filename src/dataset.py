from time import sleep
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass
import json
from langchain_community.retrievers import WikipediaRetriever

@dataclass
class OpenQASample:
    id: str
    question_stem: str
    choices: list[str]
    labels: list[str]
    answer_key: str

    @staticmethod
    def from_dict(data: dict):
        return OpenQASample(
            id=data["id"],
            question_stem=data["question_stem"],
            choices=data["choices"],
            labels=data["labels"],
            answer_key=data["answer_key"]
        )

class OpenQADataset(Dataset):
    tokenizer: PreTrainedTokenizerFast = None

    def __init__(self, split):
        self.data = [
            OpenQASample(**{
                "id": raw_sample["id"],
                "question_stem": raw_sample["question_stem"],
                "choices": raw_sample["choices"]["text"],
                "labels": raw_sample["choices"]["label"],
                "answer_key": raw_sample["answerKey"]
            }) for raw_sample in OpenQADataset.get_openqa(split)
        ]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def get_openqa(split):
        dataset = load_dataset("openbookqa")
        return dataset[split]
    
    @staticmethod
    def format_question(question):
        """
        Format the question. Could provide more context
        such as "Please answer the following question using 
        through reasoning"
        """
        return question
    
    @staticmethod
    def format_choices(choices, labels):
        for i in range(len(choices)):
            choices[i] = f"{labels[i]} - {choices[i]}"
        return choices

    @staticmethod
    def format_answer_keys(answer_keys):
        """
        Format answer keys from A, B, C, D to 0, 1, 2, 3
        :param: list of answer keys in integer
        """
        return [ord(a) - ord("A") for a in answer_keys]
    
    @staticmethod
    def collate_fn(batched_samples):
        B = len(batched_samples)
        batched_question = [[OpenQADataset.format_question(sample.question_stem)] * 4 for sample in batched_samples]  # B, 4
        batched_choices = [OpenQADataset.format_choices(sample.choices, sample.labels) for sample in batched_samples]  # B, 4
        batched_answer_key = [sample.answer_key for sample in batched_samples]  # B, 1
        # flatten batched_questions for tokenization
        batched_question = sum(batched_question, [])
        batched_choices = sum(batched_choices, [])
        # Tokenize the input texts.
        text_encoding = OpenQADataset.tokenizer(batched_question,
                                                batched_choices,
                                                padding=True,
                                                max_length=128,
                                                truncation=True,
                                                return_tensors="pt")
        # unflatten
        label_encoding = torch.LongTensor(OpenQADataset.format_answer_keys(batched_answer_key))  # B, 1

        return {
            "text_encoding": {k: v.view(B, 4, -1) for (k, v) in text_encoding.items()},
            "label_encoding": label_encoding,
        }

    @staticmethod
    def q_collate_fn(batched_samples):
        B = len(batched_samples)
        batched_question = [OpenQADataset.format_question(sample.question_stem) for sample in batched_samples]
        # Tokenize the input texts.
        text_encoding = OpenQADataset.tokenizer(batched_question,
                                                padding=True,
                                                max_length=128,
                                                truncation=True,
                                                return_tensors="pt")
        return {
            "text_encoding": text_encoding
        }
    
class GeneratedDataset(Dataset):
    tokenizer: PreTrainedTokenizerFast = None
    knowledge_retriever: WikipediaRetriever = None

    def __init__(self, path):
        self.data = self.initialize_data(path)
        GeneratedDataset.knowledge_retriever = WikipediaRetriever()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[str(index)]

    def initialize_data(self, path):
        f = open(path, "r")
        data = json.load(f)
        return data

    @staticmethod
    def fact_check_collate_fn(batched_samples):
        B = len(batched_samples)
        # Tokenize the input texts.
        # TODO: max length
        
        batched_knowledge = []
        batched_response = []
        for sample in batched_samples:
            docs = GeneratedDataset.knowledge_retriever.get_relevant_documents(query=sample)
            try:
                batched_knowledge.append(docs[0].metadata["summary"])
            except:
                batched_knowledge.append("")
            batched_response.append(sample)
            sleep(0.1)
        text_encoding = GeneratedDataset.tokenizer(batched_knowledge,
                                                    batched_response,
                                                    padding=True,
                                                    max_length=256,
                                                    truncation=True,
                                                    return_tensors="pt")
        return {
            "batch_size": B,
            "text_encoding": text_encoding,
            "text": batched_response
        }