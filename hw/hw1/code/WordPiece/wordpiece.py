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
            corpus.append(tokens)
    return corpus

class WordPieceTokenizer:
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
        self.apply_rules = []
    
    # Modified from: https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt#implementing-bpe
    def compute_normalized_pair_freqs(self):
        pair_freqs = defaultdict(int)
        token_freqs = self.compute_token_freqs()
        max_freq, max_pair = 0, None
        # print("Compute pair frequencies")
        for word, freq in self.cntr.items():
            split = self.split_map[word]
            if len(split) == 1: 
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pair_freqs[pair] += freq
                if pair_freqs[pair] / (token_freqs[split[i]] * token_freqs[split[i+1]]) > max_freq:
                    max_pair = pair
                    max_freq = pair_freqs[pair] / (token_freqs[split[i]] * token_freqs[split[i+1]])
        return max_freq, max_pair

    def compute_token_freqs(self):
        cntr = defaultdict(int)
        for word, freq in self.cntr.items():
            split = self.split_map[word]
            for token in split:
                cntr[token] += freq
        return cntr
    
    def compute_corpus_size(self):
        size = 0
        for word, freq in self.cntr.items():
            size += len(self.split_map[word]) * freq
        return size

    def merge_pair(self, max_pair):
        # print("Merge max pairs")
        self.vocab.add(max_pair[0] + max_pair[1])
        self.apply_rules.append(max_pair)
        for word, split in self.split_map.items():
            new_split = []
            idx = 0
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
        if not hasattr(self, "eval_corpus"):
            raise ValueError("No eval corpus set")
        for word, split in self.eval_split_map.items():
            new_split = []
            original_word = word
            start_idx = 0
            end_idx = len(word)
            while end_idx > start_idx:
                if word[start_idx:end_idx] in self.vocab:
                    new_split.append(word[start_idx:end_idx])
                    start_idx = end_idx
                    end_idx = len(word)
                else:
                    end_idx -= 1
            self.eval_split_map[original_word] = new_split

    def lookup_eval_split(self, token, beginning_of_word=False):
        if beginning_of_word:
            token = BEGINNING_OF_WORD + token
        if token not in self.eval_split_map:
            raise ValueError("Invalid token")
        return self.eval_split_map[token]

@timer
def wordpiece_train(corpus):
    tokenizer = WordPieceTokenizer(corpus)
    max_rules = 4000
    max_freq, max_pair = tokenizer.compute_normalized_pair_freqs()
    corpus_size = []
    vocab_size = []
    size = tokenizer.compute_corpus_size()
    corpus_size.append(size)
    vocab_size.append(len(tokenizer.vocab))
    for i in tqdm(range(max_rules)):
        tokenizer.merge_pair(max_pair)
        size = tokenizer.compute_corpus_size()
        corpus_size.append(size)
        vocab_size.append(len(tokenizer.vocab))
        _, max_pair = tokenizer.compute_normalized_pair_freqs()
    assert len(tokenizer.apply_rules) == max_rules
    print(f"final training corpus size = {corpus_size[-1]}")
    print(f"final vocab size = {len(tokenizer.vocab)}")
    print(f"final number of rules = {len(tokenizer.apply_rules)}")
    fig = plt.plot()
    plt.scatter(vocab_size, corpus_size)
    plt.xlabel("Vocabulary size")
    plt.ylabel("Training corpus size")
    # plt.show()
    return tokenizer
    
def wordpiece_apply(eval_corpus, tokenizer):
    tokenizer.set_eval_corpus(eval_corpus)
    tokenizer.apply()
    size = 0
    for tokens in eval_corpus:
        for i, token in enumerate(tokens):
            size += len(tokenizer.lookup_eval_split(token, i > 0))
    print(f"final eval corpus size = {size}")

if __name__ == "__main__":
    corpus = read_corpus(4000)
    tokenizer = wordpiece_train(corpus)
    # eval_corpus = read_corpus(-1000)
    # wordpiece_apply(eval_corpus, tokenizer)
    eval_corpus = ["Analysts were expecting the opposite, a deepening of the deficit.".split(),
                   "Five minutes later, a second person arrived, aged around thirty, with knife wounds.".split()]
    tokenizer.set_eval_corpus(eval_corpus)
    tokenizer.apply()
    for tokens in eval_corpus:
        split = []
        for i, token in enumerate(tokens):
            split.extend(tokenizer.lookup_eval_split(token, i > 0))
        print(split)

        