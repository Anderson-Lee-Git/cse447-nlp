from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm

UNK = "<UNK>"
STOP = "<STOP>"
START = "<START>"
THRESHOLD = 3
TRAIN_PATH = "/Users/anderson/Documents/CSE/cse447-nlp/hw1/code/N-gram/1b_benchmark.train.tokens"
VAL_PATH = "/Users/anderson/Documents/CSE/cse447-nlp/hw1/code/N-gram/1b_benchmark.dev.tokens"
TEST_PATH = "/Users/anderson/Documents/CSE/cse447-nlp/hw1/code/N-gram/1b_benchmark.test.tokens"

def read_corpus(path):
    f = open(path, "r")
    corpus = []
    for line in f.readlines():
        line = line.strip().strip('\n')
        tokens = line.split()
        tokens.append(STOP)
        corpus.append(tokens)
    f.close()
    return corpus

def unkify(corpus):
    """
    Unkify the corpus
    """
    print(f"Using threshold = {THRESHOLD}")
    cntr = Unigram.unigram(corpus)
    new_corpus = []
    for i in range(len(corpus)):
        new_sentence = []
        for j in range(len(corpus[i])):
            if cntr[corpus[i][j]] < THRESHOLD:
                new_sentence.append(UNK)
            else:
                new_sentence.append(corpus[i][j])
        new_corpus.append(new_sentence)
    return new_corpus

def count_total(cntr):
    s = 0
    for token in cntr.keys():
        s += cntr[token]
    return s

class Unigram:
    def __init__(self, corpus, k=0) -> None:
        """
        Attributes
        - cntr: Store counts of each token in corpus of structure dict
        - vocab: A list of vocabularies space with indexed order
        - k: add k smoothing factor for likelihood of each unigram
        """
        self.cntr = Unigram.unigram(corpus)
        self.total_count = count_total(self.cntr)
        self.vocab = self.cntr.keys()
        self.k = k
    
    def likelihood(self, token):
        """
        Return maximum likelihood estimate of the next 
        token to be the given token.
        """
        if token not in self.cntr.keys():
            token = UNK
        return add_k_smoothing(count=self.cntr[token],
                                total_count=self.total_count,
                                vocab_size=len(self.vocab),
                                k=self.k)
    
    @staticmethod
    def unigram(corpus):
        """
        Count frequencies of each token in corpus
        """
        flat_corpus = []
        for tokens in corpus:
            flat_corpus.extend(tokens)
        cntr = Counter(flat_corpus)
        return cntr
    
    def __repr__(self) -> str:
        return "Unigram"

class Bigram:
    def __init__(self, corpus, k=0) -> None:
        """
        Attributes
        - uni_cntr: Count of each vocab in corpus
        - cntr: Store counts of each bigram of structure dict(dict)
                where the outer key is x_{i-1} and the inner key is 
                x_{i}
        - vocab: A list of vocabularies space with indexed order
        - distribution: Cached distribution for potentially faster generation
                        Doesn't impact perplexity
        - k: add k smoothing factor for likelihood of each bigram
        """
        self.uni_cntr = Unigram.unigram(corpus)
        self.cntr = Bigram.bigram(corpus)
        self.vocab = list(self.uni_cntr.keys())
        self.distribution = {}
        self.k = k


    def get_distribution(self, prev_token):
        """
        Calculate the distribution conditioned on prev_token
        to provide faster generation. This is not used for 
        likelihood for faster perplexity evaluation.
        """
        if prev_token not in self.uni_cntr.keys() and prev_token != START:
            prev_token = UNK
        if prev_token not in self.distribution:
            self.distribution[prev_token] = [
                add_k_smoothing(count=self.cntr[prev_token][token],
                                total_count=count_total(self.cntr[prev_token]),
                                vocab_size=len(self.vocab),
                                k=self.k)
                for token in self.vocab
            ]
        return self.distribution[prev_token]
    
    def generate(self, prev_token):
        return np.random.choice(self.vocab, p=self.get_distribution(prev_token))
    
    def likelihood(self, prev_token, token):
        if token not in self.uni_cntr.keys():
            token = UNK
        if prev_token not in self.uni_cntr.keys() and prev_token != START:
            prev_token = UNK
        if prev_token not in self.cntr.keys():
            # if prev_token not exist, create 0 count for it
            self.cntr[prev_token] = defaultdict(int)
        return add_k_smoothing(count=self.cntr[prev_token][token],
                                total_count=count_total(self.cntr[prev_token]),
                                vocab_size=len(self.vocab),
                                k=self.k)
    
    @staticmethod
    def bigram(corpus):
        """
        Count frequencies of bigram in the training corpus
        as described in writeup
        """
        cntr = {START: defaultdict(int)}
        for tokens in corpus:
            for i in range(len(tokens)):
                if i == 0:
                    cntr[START][tokens[i]] += 1
                else:
                    if tokens[i-1] not in cntr:
                        cntr[tokens[i-1]] = defaultdict(int)
                    cntr[tokens[i-1]][tokens[i]] += 1
        return cntr

    def __repr__(self) -> str:
        return "Bigram"
    
class Trigram:
    def __init__(self, corpus, k=0) -> None:
        """
        Attributes
        - uni_cntr: Count of each vocab in corpus
        - cntr: Store counts of each trigram in structure of dict(dict(dict))
                where the outer key is x_{i-2}, middle key is x_{i-1}, and 
                inner key is x_{i}
        - vocab: A list of vocabularies space with indexed order
        - distribution: Cached distribution for potentially faster generation
                        Doesn't impact perplexity
        - k: add k smoothing factor for likelihood of each trigram
        """
        self.uni_cntr = Unigram.unigram(corpus)
        self.cntr = Trigram.trigram(corpus)
        self.vocab = list(self.uni_cntr.keys())
        self.distribution = {}
        self.k = k

    def get_distribution(self, prev_prev_token, prev_token):
        """
        Calculate the distribution conditioned on prev_prev_token, prev_token
        to provide faster generation. This is not used for 
        likelihood for faster perplexity evaluation.
        """
        if prev_prev_token not in self.uni_cntr.keys() and prev_prev_token != START:
            prev_prev_token = UNK
        if prev_token not in self.uni_cntr.keys() and prev_token != START:
            prev_token = UNK
        if prev_prev_token not in self.distribution:
            self.distribution[prev_prev_token] = {}
        if prev_token not in self.distribution[prev_prev_token]:
            # Cache the distribution
            self.distribution[prev_prev_token][prev_token] = \
                [
                    add_k_smoothing(count=self.cntr[prev_prev_token][prev_token][token],
                                 total_count=count_total(self.cntr[prev_prev_token][prev_token]),
                                 vocab_size=len(self.vocab),
                                 k=self.k) \
                    for token in self.vocab
                ]
        return self.distribution[prev_prev_token][prev_token]

    def generate(self, prev_prev_token, prev_token):
        return np.random.choice(self.vocab, p=self.get_distribution(prev_prev_token, prev_token))
    
    def likelihood(self, prev_prev_token, prev_token, token):
        if token not in self.uni_cntr.keys():
            token = UNK
        if prev_prev_token not in self.uni_cntr.keys() and prev_prev_token != START:
            prev_prev_token = UNK
        if prev_token not in self.uni_cntr.keys() and prev_token != START:
            prev_token = UNK
        if prev_prev_token not in self.cntr:
            self.cntr[prev_prev_token] = {}
        if prev_token not in self.cntr[prev_prev_token]:
            self.cntr[prev_prev_token][prev_token] = defaultdict(int)
        return add_k_smoothing(count=self.cntr[prev_prev_token][prev_token][token],
                                total_count=count_total(self.cntr[prev_prev_token][prev_token]),
                                vocab_size=len(self.vocab),
                                k=self.k)
    
    @staticmethod
    def trigram(corpus):
        """
        Count frequencies of triplets in training corpus 
        as described in the writeup
        """
        cntr = {START: {START: defaultdict(int)}}
        for tokens in corpus:
            for i in range(len(tokens)):
                if i == 0:
                    cntr[START][START][tokens[i]] += 1
                elif i == 1:
                    if tokens[i-1] not in cntr[START]:
                        cntr[START][tokens[i-1]] = defaultdict(int)
                    cntr[START][tokens[i-1]][tokens[i]] += 1
                else:
                    if tokens[i-2] not in cntr:
                        cntr[tokens[i-2]] = {}
                    if tokens[i-1] not in cntr[tokens[i-2]]:
                        cntr[tokens[i-2]][tokens[i-1]] = defaultdict(int)
                    cntr[tokens[i-2]][tokens[i-1]][tokens[i]] += 1
        return cntr
    
    def __repr__(self) -> str:
        return "Trigram"

class Interpolation:
    def __init__(self, lambda_1, lambda_2, lambda_3, corpus, k) -> None:
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.unigram = Unigram(corpus, k=k)
        self.bigram = Bigram(corpus, k=k)
        self.trigram = Trigram(corpus, k=k)

    def likelihood(self, prev_prev_token, prev_token, token):
        likelihood = self.unigram.likelihood(token) * self.lambda_1 + \
                     self.bigram.likelihood(prev_token, token) * self.lambda_2 + \
                     self.trigram.likelihood(prev_prev_token, prev_token, token) * self.lambda_3
        return likelihood

    def __repr__(self) -> str:
        return f"Interpolation(lambda1={self.lambda_1}, lambda2={self.lambda_2}, lambda3={self.lambda_3})"

def add_k_smoothing(count, total_count, vocab_size, k=0):
    """
    A function for maximum likelihood estimate after smoothing
    k = 0 refers to default MLE
    """
    try:
        return (count + k) / (total_count + k * vocab_size)
    except:
        # division by zero exception
        return 0

def calculate_perplexity(corpus, model):
    perplexity = 0
    log_corpus_stat(corpus)
    if isinstance(model, Unigram):
        print(f"Calculate perplexity for {model}")
        for tokens in tqdm(corpus):
            sentence_entropy = 0
            for token in tokens:
                sentence_entropy += np.log(model.likelihood(token))
            perplexity += np.exp(-sentence_entropy / len(tokens))
    elif isinstance(model, Bigram):
        print(f"Calculate perplexity for {model}")
        for tokens in tqdm(corpus):
            sentence_entropy = 0
            for i, token in enumerate(tokens):
                if i == 0:
                    prev_token = START
                sentence_entropy += np.log(model.likelihood(prev_token, token))
                prev_token = token
            perplexity += np.exp(-sentence_entropy / len(tokens))
    elif isinstance(model, Trigram) or \
        isinstance(model, Interpolation):
        print(f"Calculate perplexity for {model}")
        for tokens in tqdm(corpus):
            sentence_entropy = 0
            for i, token in enumerate(tokens):
                if i == 0:
                    prev_prev_token = prev_token = START
                elif i == 1:
                    prev_prev_token = START
                sentence_entropy += np.log(model.likelihood(prev_prev_token, prev_token, token))
                prev_prev_token = prev_token
                prev_token = token
            perplexity += np.exp(-sentence_entropy / len(tokens))
    else:
        raise NotImplementedError("Not implemented")
    return perplexity / len(corpus)

def evaluate_perplexity(eval_paths, k=0):
    """
    Evaluate perplexity with smoothing factor k
    k = 0 refers to classical perplexity without smoothing
    """
    corpus = read_corpus(TRAIN_PATH)
    corpus = unkify(corpus)

    unigram = Unigram(corpus, k=k)
    bigram = Bigram(corpus, k=k)
    trigram = Trigram(corpus, k=k)

    for eval_path in eval_paths:
        eval_corpus = read_corpus(eval_path)

        if eval_path == TRAIN_PATH:
            print("=====Train set=====")
        elif eval_path == VAL_PATH:
            print("=====Dev set=====")
        elif eval_path == TEST_PATH:
            print("=====Test set=====")
        
        unigram_perplexity = calculate_perplexity(eval_corpus, unigram)
        print(f"unigram perplexity = {unigram_perplexity}, k = {k}")

        if eval_path == TRAIN_PATH or k > 0:
            # only evaluate for not -infinity case
            bigram_perplexity = calculate_perplexity(eval_corpus, bigram)
            print(f"bigram perplexity = {bigram_perplexity}, k = {k}")

            trigram_perplexity = calculate_perplexity(eval_corpus, trigram)
            print(f"trigram perplexity = {trigram_perplexity}, k = {k}")

def perplexity_laplace(eval_paths):
    """
    Laplace perplexity utilizing perplexity function with k = 1
    """
    print("=====Laplace Perplexity=====")
    evaluate_perplexity(eval_paths, k=1)

def generate():
    corpus = read_corpus(TRAIN_PATH)
    corpus = unkify(corpus)
    trigram = Trigram(corpus, k=0)
    pp_token = START
    p_token = START
    while p_token != STOP:
        token = trigram.generate(pp_token, p_token)
        print(token, end=" ")
        pp_token = p_token
        p_token = token

def interpolation(eval_paths):
    lambdas = [(i, 0.3, 0.7 - i) for i in np.linspace(0.1, 0.6, 5)] + \
                [(0.2, 0.5, 0.3)]
    corpus = read_corpus(TRAIN_PATH)
    corpus = unkify(corpus)
    # lambdas used in constructor can be overriden
    model = Interpolation(1.0, 0.0, 0.0, corpus, k=0)
    for eval_path in eval_paths:
        eval_corpus = read_corpus(eval_path)

        if eval_path == TRAIN_PATH:
            print("=====Train set=====")
        elif eval_path == VAL_PATH:
            print("=====Dev set=====")
        elif eval_path == TEST_PATH:
            print("=====Test set=====")

        for lambda_1, lambda_2, lambda_3 in lambdas:
            # Override lambdas
            model.lambda_1 = lambda_1
            model.lambda_2 = lambda_2
            model.lambda_3 = lambda_3
            perplexity = calculate_perplexity(eval_corpus, model)
            print(f"Interpolation perplexity = {perplexity}, \
                lambda_1 = {lambda_1}, lambda_2 = {lambda_2}, lambda_3 = {lambda_3}")
    
    print("=====Test set=====")
    best_lambdas = 0.475, 0.3, 0.225
    model.lambda_1, model.lambda_2, model.lambda_3 = best_lambdas
    perplexity = calculate_perplexity(read_corpus(TEST_PATH), model)
    print(f"Interpolation perplexity = {perplexity}, \
                lambda_1 = {model.lambda_1}, lambda_2 = {model.lambda_2}, \
                lambda_3 = {model.lambda_3}")

def half_training_set():
    """
    Explore the effect of half training set
    """
    print("=====HALF TRAINING SET=====")
    corpus = read_corpus(TRAIN_PATH)
    corpus = corpus[:len(corpus) // 2]
    corpus = unkify(corpus)
    model = Interpolation(0.475, 0.3, 0.225, corpus, k=0)
    print("=====Dev set=====")
    eval_corpus = read_corpus(VAL_PATH)
    print(calculate_perplexity(eval_corpus, model))
    print("=====Test set=====")
    eval_corpus = read_corpus(TEST_PATH)
    print(calculate_perplexity(eval_corpus, model))

def set_unk_threshold(threshold):
    global THRESHOLD
    THRESHOLD = threshold

def log_corpus_stat(corpus):
    print(f"Number of sentence = {len(corpus)}")
    s = 0
    unk_cnt = 0
    for sentence in corpus:
        s += len(sentence)
        for token in sentence:
            if token == UNK:
                unk_cnt += 1
    print(f"Number of tokens = {s}")
    print(f"Number of {UNK} = {unk_cnt}")


if __name__ == "__main__":
    # Perplexity without smoothing
    # evaluate_perplexity([TRAIN_PATH, VAL_PATH, TEST_PATH], k=0)

    # Perplexity with laplace smoothing
    # perplexity_laplace([TRAIN_PATH, VAL_PATH, TEST_PATH])

    # Perplexity with k = 0.5, 0.2, 0.01
    # ks = [0.5, 0.2, 0.01]
    # for k in ks:
    #     evaluate_perplexity([TRAIN_PATH, VAL_PATH, TEST_PATH], k=k)

    # Interpolation explorations
    # interpolation([TRAIN_PATH, VAL_PATH])

    # Use half of training set
    # half_training_set()

    # Use 5 as UNK threshold
    set_unk_threshold(5)
    corpus = read_corpus(TRAIN_PATH)
    corpus = unkify(corpus)
    model = Interpolation(0.475, 0.3, 0.225, corpus, k=0)
    print(calculate_perplexity(read_corpus(VAL_PATH), model))
    print(calculate_perplexity(read_corpus(TEST_PATH), model))

    # Use 2 as UNK threshold
    set_unk_threshold(2)
    corpus = read_corpus(TRAIN_PATH)
    corpus = unkify(corpus)
    model = Interpolation(0.475, 0.3, 0.225, corpus, k=0)
    print(calculate_perplexity(read_corpus(VAL_PATH), model))
    print(calculate_perplexity(read_corpus(TEST_PATH), model))
