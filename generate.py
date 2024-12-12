# pyright: basic


import torch
from torch.utils.data import Dataset

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed


set_seed(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64
BLOCK_SIZE = 32


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.text = text
        self.tokenizer = tokenizer
        token_ids = tokenizer(text)

        self.tokens = token_ids
        self.input_ids = []
        self.target_ids = []

        for i in range(0, token_ids.size(0) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self):
        return len(self.input_ids)

    def get_vocab_size(self):
        return len(torch.unique(self.tokens))

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(
            f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
        )


def generate(prompt="", num_samples=10, steps=20, do_sample=True):
    bpe = BPETokenizer()
    if prompt == "":
        x = torch.tensor([[bpe.encoder.encoder["<|endoftext|>"]]], dtype=torch.long)
    else:
        x = bpe(prompt).to(device)

    x = x.expand(num_samples, -1)
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

    for i in range(num_samples):
        out = bpe.decode(y[i].cpu().squeeze())
        print("-" * 80)
        print(out)


if __name__ == "__main__":
    with open("../gpt-from-scratch/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    train_dataset = TextDataset(text, BPETokenizer(), BLOCK_SIZE, 1)
    VOCAB_SIZE = train_dataset.get_vocab_size()

    x, y = train_dataset[0]
    print(f"{x}\n{y}")

    model_config = GPT.get_default_config()
    model_config.model_type = "gpt-nano"
    model_config.vocab_size = VOCAB_SIZE
    model_config.block_size = BLOCK_SIZE
    model = GPT(model_config).to(device)

    train_config = Trainer.get_default_config()
    train_config.device = device
    train_config.batch_size = BATCH_SIZE
    train_config.block_size = BLOCK_SIZE
    train_config.learning_rate = 5e-4
    train_config.max_iters = 8000
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_dataset)

    trainer.set_callback("on_batch_end", batch_end_callback)
    trainer.run()

    generate(prompt="Sophia Berger, the", num_samples=5, steps=40)
