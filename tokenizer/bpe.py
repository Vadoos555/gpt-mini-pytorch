from collections import Counter
import re


def pre_tokenize(text: str):
    parts = re.findall(r"\s+|[^\w\s]+|\w+", text)
    
    result = []
    space = ""
    
    for part in parts:
        if part.isspace():
            space += part
        else:
            result.append(space + part)
            space = ""
    
    if space:
        result.append(space)
    
    return result


def get_pair_counts(sequences):
    counts = Counter()
    
    for tokens in sequences:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            counts[pair] += 1
    
    return counts


def get_best_pair(counts):
    return max(counts, key=counts.get)


def merge_pair(tokens, pair):
    result = []
    i = 0
    
    while i < len(tokens):
        if i < len(tokens)-1 and tokens[i]==pair[0] and tokens[i+1]==pair[1]:
            result.append(tokens[i] + tokens[i+1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    
    return result


def merge_all(sequences, pair):
    return [merge_pair(tokens, pair) for tokens in sequences]


def train_bpe(sequences, num_merges):
    merges = []
    vocab = set()
    
    for sequence in sequences:
        vocab.update(sequence)
    
    for _ in range(num_merges):
        counts = get_pair_counts(sequences)
        if not counts:
            break
        
        pair = get_best_pair(counts)
        merged = pair[0] + pair[1]
        vocab.add(merged)
        sequences = merge_all(sequences, pair)
        merges.append(pair)
    
    return sequences, merges, vocab


def apply_merges(tokens, merges):
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    
    return tokens


class BPE:
    def __init__(self):
        self.vocab = []
        self.merges = []
        
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.special_tokens = [
            "<|user|>",
            "<|assistant|>",
              "<|pad|>",
        ]
    
    def encode(self, text: str):
        tokens = []
        i = 0
        while i < len(text):
            # check special tokens
            found_special = False
            
            for special_token in self.special_tokens:
                
                if text.startswith(special_token, i):
                    tokens.append(special_token)
                    i += len(special_token)
                    found_special = True
                    break
                
            if found_special:
                continue
            
            # normal text
            start = i
            while i < len(text):
                is_special = False
                
                for special_token in self.special_tokens:
                    
                    if text.startswith(special_token, i):
                        is_special = True
                        break
                
                if is_special:
                    break
                i += 1
        
            normal_text = text[start:i]
            
            chunks = pre_tokenize(normal_text)
            
            for chunk in chunks:
                chunk_tokens = list(chunk)
                chunk_tokens = apply_merges(chunk_tokens, self.merges)
                
                tokens.extend(chunk_tokens)
        
        return [self.token_to_id[token] for token in tokens]
    
    def decode(self, ids):
        tokens = [self.id_to_token[i] for i in ids]
        
        return "".join(tokens)
    
    def train(self, sequences, num_merges):
        sequences, merges, vocab = train_bpe(sequences, num_merges)
        
        self.merges = merges
        self.vocab = sorted(vocab)
        
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        # add special tokens
        for special_token in self.special_tokens:
            token_id = len(self.vocab)
            
            self.vocab.append(special_token)
            self.token_to_id[special_token] = token_id
            self.id_to_token[token_id] = special_token
