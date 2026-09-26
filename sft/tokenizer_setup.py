from tokenizer.bpe import BPE, pre_tokenize
from sft.formatting import format_example


def train_tokenizer(data):
    sequences = []

    for example in data:
        text = format_example(example["instruction"], example["response"])

        chunks = pre_tokenize(text)
        tokens = []

        for chunk in chunks:
            tokens.extend(list(chunk))

        sequences.append(tokens)

    tokenizer = BPE()
    tokenizer.train(sequences, num_merges=100)

    return tokenizer
