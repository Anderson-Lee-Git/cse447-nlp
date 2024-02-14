from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

BEGINNING_OF_WORD = "<B>"

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[INFO] {func.__name__} took {elapsed_time} seconds to run")
        return result
    return wrapper

def read_corpus(num_lines):
    path = "/Users/anderson/Documents/CSE/cse447-nlp/hw1/code/BPE/BPE-data.txt"
    corpus = []
    with open(path, "r") as f:
        if num_lines > 0:
            lines = f.readlines()[:num_lines]
        else:
            lines = f.readlines()[num_lines:]
        for line in lines:
            line = line.strip("\n")
            tokens = line.split()
            if "\n" in tokens:
                print(tokens.index("\n"))
                print(len(tokens))
            if "" in tokens:
                print(tokens.index(""))
                print(len(tokens))
            corpus.append(tokens)
    return corpus

class BpeTokenizer:
    def __init__(self, corpus) -> None:
        self.cntr = defaultdict(int)
        self.split_map = {}
        self.corpus = corpus
        self.vocab = set()
        # Construct vocab to be all characters
        # Construct split map by characters
        print("Initialize tokenizer")
        for line in self.corpus:
            for i, token in enumerate(line):
                split = []
                for ch in token:
                    self.vocab.add(ch)
                    split.append(ch)
                if i > 0:
                    token = BEGINNING_OF_WORD + token
                    split.insert(0, BEGINNING_OF_WORD)
                    self.vocab.add(BEGINNING_OF_WORD)
                self.cntr[token] += 1
                self.split_map[token] = split
        # indexed vocab for inference
        self.vocab = list(self.vocab)
        self.apply_rules = []
    
    # Modified from: https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt#implementing-bpe
    def compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        max_freq, max_pair = 0, None
        # print("Compute pair frequencies")
        for word, freq in self.cntr.items():
            split = self.split_map[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pair_freqs[pair] += freq
                if pair_freqs[pair] > max_freq:
                    max_pair = pair
                    max_freq = pair_freqs[pair]
        return max_freq, max_pair
    
    def compute_corpus_size(self):
        size = 0
        for word, freq in self.cntr.items():
            size += len(self.split_map[word]) * freq
        return size

    def merge_pair(self, max_pair):
        # print("Merge max pairs")
        self.vocab.append(max_pair[0] + max_pair[1])
        self.apply_rules.append(max_pair)
        # print(f"max_pair = {max_pair}")
        for word, split in self.split_map.items():
            new_split = []
            idx = 0
            # print(f"old_split = {split}")
            while idx < len(split):
                if idx == len(split) - 1:
                    new_split.append(split[idx])
                    break
                elif (split[idx], split[idx+1]) == max_pair:
                    new_split.append(split[idx] + split[idx+1])
                    idx += 2
                else:
                    new_split.append(split[idx])
                    idx += 1
            # print(f"new_split = {new_split}")
            self.split_map[word] = new_split
    
    def set_eval_corpus(self, eval_corpus):
        self.eval_corpus = eval_corpus
        self.eval_split_map = {}
        s = 0
        for line in self.eval_corpus:
            for i, token in enumerate(line):
                split = []
                for ch in token:
                    split.append(ch)
                if i > 0:
                    token = BEGINNING_OF_WORD + token
                    split.insert(0, BEGINNING_OF_WORD)
                self.eval_split_map[token] = split
                s += len(split)
        print(f"initial eval corpus size = {s}")
    
    def apply(self):
        if not hasattr(self, "eval_corpus") or not self.eval_corpus:
            raise ValueError("No eval corpus set")
        for pair in self.apply_rules:
            for word, split in self.eval_split_map.items():
                new_split = []
                idx = 0
                while idx < len(split):
                    if idx == len(split) - 1:
                        new_split.append(split[idx])
                        break
                    if (split[idx], split[idx+1]) == pair:
                        new_split.append(split[idx] + split[idx+1])
                        idx += 2
                    else:
                        new_split.append(split[idx])
                        idx += 1
                self.eval_split_map[word] = new_split
    
    def lookup_eval_split(self, token, beginning_of_word=False):
        if beginning_of_word:
            token = BEGINNING_OF_WORD + token
        if token not in self.eval_split_map:
            raise ValueError("Invalid token")
        return self.eval_split_map[token]

@timer
def bpe_train(corpus):
    tokenizer = BpeTokenizer(corpus)
    threshold = 2
    max_freq, max_pair = tokenizer.compute_pair_freqs()
    corpus_size = []
    vocab_size = []
    # output helper
    max_num_dig = len(str(max_freq))
    size = tokenizer.compute_corpus_size()
    corpus_size.append(size)
    vocab_size.append(len(tokenizer.vocab))
    print(f"initial training corpus size = {corpus_size[0]}")
    print(f"initial vocab size = {len(tokenizer.vocab)}")
    while max_freq > threshold:
        # output helper
        num_dig = len(str(max_freq))
        output = " " * (max_num_dig - num_dig) + str(max_freq)
        print(f"max bigram count = {output}", end="\r")
        # actual training
        tokenizer.merge_pair(max_pair)
        size = tokenizer.compute_corpus_size()
        corpus_size.append(size)
        vocab_size.append(len(tokenizer.vocab))
        max_freq, max_pair = tokenizer.compute_pair_freqs()
    print(f"final training corpus size = {corpus_size[-1]}")
    print(f"final vocab size = {len(tokenizer.vocab)}")
    fig = plt.figure()
    plt.scatter(vocab_size, corpus_size)
    plt.xlabel("Vocabulary size")
    plt.ylabel("Training corpus size")
    plt.show()
    return tokenizer


def bpe_apply(eval_corpus, tokenizer):
    tokenizer.set_eval_corpus(eval_corpus)
    tokenizer.apply()
    s = 0
    for line in eval_corpus:
        for i, token in enumerate(line):
            s += len(tokenizer.lookup_eval_split(token, i != 0))
    print(f"final eval corpus size = {s}")
            

if __name__ == "__main__":
    corpus = read_corpus(4000)
    tokenizer = bpe_train(corpus)
    eval_corpus = read_corpus(-1000)
    bpe_apply(eval_corpus, tokenizer)
    
