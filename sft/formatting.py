def format_example(instruction, response):
    text = (
        "<|user|>\n"
        + instruction
        + "\n"
        + "<|assistant|>\n"
        + response
    )

    return text


if __name__ == '__main__':
    from sft.dataset import data
    
    example = data[0]
    
    text = format_example(example["instruction"], example["response"])
    print(text)
    