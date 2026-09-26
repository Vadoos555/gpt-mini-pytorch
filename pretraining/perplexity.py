import math


def perplexity(loss):
    return math.exp(loss)


if __name__ == '__main__':
    losses = [0.0, 0.693, 1.0, 2.0, 5.0, 9.1646, 9.2103]

    for loss in losses:
        ppl = perplexity(loss)
        
        print(f'loss = {loss:.4f}  perplexity = {ppl:.4f}')
        