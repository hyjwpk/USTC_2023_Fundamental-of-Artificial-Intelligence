#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math
import tqdm
import matplotlib.pyplot as plt


class char_tokenizer:
    """
    a very simple char-based tokenizer. the tokenizer turns a string into a list of integers.
    """

    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        # TODO: calculate the vocab size and create a dictionary that maps each character to a unique integer
        self.dic = {char: i for i, char in enumerate(corpus)}
        self.n_vocab = len(corpus)
        # End of your code

    def encode(self, string: str):
        # TODO: convert a string into a list of integers and return, using the dictionary you created above
        return [self.dic[char] for char in string]
        # End of your code

    def decode(self, codes: List[int]):
        # TODO: convert a list of integers into a string and return, using the dictionary you created above
        return ''.join([self.corpus[code] for code in codes])
        # End of your code

class Head(nn.Module):
    """single head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        # TODO: create three linear layers, Key, Query, and Value, each of which maps from n_embd to head_size
        #       and assign them to self.Key, self.Query, and self.Value, respectively
        self.Key = nn.Linear(n_embd, head_size)
        self.Query = nn.Linear(n_embd, head_size)
        self.Value = nn.Linear(n_embd, head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("triu", torch.triu(torch.ones(block_size, block_size), diagonal=1).bool())
        # End of your code
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, inputs):
        # TODO: implement the forward function of the head
        #       the input is a tensor of shape (batch, time, n_embd)
        #       the output should be a tensor of shape (batch, time, head_size)
        #       you may use the tril buffer defined above to mask out the upper triangular part of the affinity matrix
        Q = self.Query(inputs)
        K = self.Key(inputs)
        V = self.Value(inputs)
        dot_products = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32))
        dot_products = dot_products.masked_fill(self.triu[:inputs.shape[1], :inputs.shape[1]], -1e9)
        dot_products = self.softmax(dot_products)
        out = torch.bmm(dot_products, V)
        # End of your code
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        #TODO: implement heads and projection
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads * head_size, n_embd)
        # End of your code

    def forward(self, inputs):
        #TODO: implement the forward function of the multi-head attention
        heads_outputs = [head(inputs) for head in self.heads]
        out = torch.cat(heads_outputs, dim=-1)
        return self.projection(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        #TODO: implement the feed-forward network
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
        # End of your code

    def forward(self, inputs):
        return self.net(inputs)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        # TODO: implement the block of transformer using the MultiHeadAttention and 
        # FeedForward modules, along with the layer normalization layers
        self.self_attention = MultiHeadAttention(n_heads, n_embd // n_heads)
        self.feed_forward = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        # End of your code
    def forward(self, inputs):
        #TODO: implement the forward function of the block, you may refer to the docs of this experiment
        attention_out = self.self_attention(inputs)
        norm1_out = self.layer_norm1(inputs + attention_out)
        forward_out = self.feed_forward(norm1_out)
        inputs = self.layer_norm2(norm1_out + forward_out)
        # End of your code
        return inputs


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: create the embedding table, the stack of blocks, the layer normalization layer, 
        # and the linear layers.
        def get_position_encoding(self, n_position):
            position = torch.arange(n_position).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, n_embd, 2) * -(math.log(10000.0) / n_embd))
            pos_encoding = torch.zeros((n_position, n_embd))
            pos_encoding[:, 0::2] = torch.sin(position * div_term)
            pos_encoding[:, 1::2] = torch.cos(position * div_term)
            return pos_encoding.to(device)
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_embd)
        self.position_encoding = get_position_encoding(self, block_size)
        self.blocks = nn.ModuleList([Block(n_embd, n_heads) for _ in range(n_layers)])
        self.linear = nn.Linear(n_embd, n_vocab)
        self.softmax = nn.Softmax(dim=-1)
        # End of your code

    def forward(self, inputs, labels=None):
        # TODO: implement the forward function of the transformer

        # inputs:(batch, context)
        batch, time = inputs.shape
        embedding = self.embedding(inputs) + self.position_encoding[:time, :].unsqueeze(0)
        for block in self.blocks:
            embedding = block(embedding)
        logits = self.linear(embedding)
        probs = self.softmax(logits)
        # End of your code

        # compute the loss
        
        if labels is None:
            loss = None
        else:
            batch, time, channel = logits.shape
            logits = logits.view(batch * time, channel)
            labels = labels.view(batch * time)
            loss = F.cross_entropy(logits, labels)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        # TODO: generate new tokens from the transformer, using the inputs as the context,
        #  and return the generated tokens with length of max_new_tokens
        out = inputs
        for _ in range(max_new_tokens):
            # generates new tokens by iteratively sampling from the model's predicted probability distribution, 
            # concatenating the sampled tokens to the input sequence, and returning the updated sequence.
            batch, time = inputs.shape
            embedding = self.embedding(inputs) + self.position_encoding[:time, :].unsqueeze(0)
            for block in self.blocks:
                embedding = block(embedding)
            logits = self.linear(embedding[:, -1, :])
            probs = self.softmax(logits)
            next_token = torch.multinomial(probs, num_samples=1)
            # next_token = torch.argmax(probs, dim=-1).unsqueeze(1)
            out = torch.cat((out, next_token), dim=1)
            inputs = out[:, -block_size:]
        inputs = out
        # End of your code
        return inputs


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def generate(model):
    text = "Your suffering in this death,"
    context = torch.tensor([tokenizer.encode(text)], device=device, dtype=torch.long)
    # context = torch.zeros((1, 1), device=device, dtype=torch.long)
    print(decode(model.generate(context, max_new_tokens=256)[0].tolist()))


def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    progress_bar = tqdm.tqdm(range(max_iters))
    train_losses, val_losses = [], []

    for iter in progress_bar:
        
        if (iter + 1) % eval_interval == 0 or iter == 0:
            losses = estimate_loss(model)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            progress_bar.write(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        inputs, labels = get_batch("train")

        logits, loss = model(inputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


# define the hyperparameters
batch_size = 16
block_size = 256
max_iters = 5000 # set the number of training iterations as you like
eval_interval = 50
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 64
n_heads = 8
n_layers = 6

# read the dataset
with open("F:/OneDrive/Code/C++/AI/exp2/part_2/data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))

# initialize the vocabulary
tokenizer = char_tokenizer(chars)
encode = tokenizer.encode
decode = tokenizer.decode
n_vocab = tokenizer.n_vocab

# separate the dataset into train and validation
train_data = torch.tensor(encode(text[: -len(text) // 10]), dtype=torch.long)
val_data = torch.tensor(encode(text[-len(text) // 10 :]), dtype=torch.long)

# define the model
model = Transformer().to(device)
m_state_dict = torch.load("F:/OneDrive/Code/C++/AI/exp2/part_2/model/model.pth")
model.load_state_dict(m_state_dict)
# train(model)
# torch.save(model.state_dict(), "../model/model_2.pth")
generate(model)
