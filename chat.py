import torch

from config.model_config import GPTConfig
from model.gpt import GPT

from sft.dataset import data
from sft.evaluation import evaluate
from sft.tokenizer_setup import train_tokenizer


PATH = "checkpoints/sft_best.pt"


def main():
    tokenizer = train_tokenizer(data)
    print("vocab size:", len(tokenizer.vocab))

    config = GPTConfig(vocab_size=len(tokenizer.vocab))
    model = GPT(config)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    print(
        f"loaded SFT checkpoint "
        f"epoch {checkpoint['epoch']}"
    )

    print()
    print("GPT Mini Chat")
    print('Type "exit" to stop')
    print()

    while True:
        instruction = input("User: ")

        if instruction.lower() == "exit":
            break

        response = evaluate(model, tokenizer, instruction, max_new_tokens=30)
        
        print("Assistant:", response)


if __name__ == "__main__":
    main()