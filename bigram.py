import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8 # context length
max_iters = 3000
eval_interval = 300 # used for estimating loss
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# -------------------------

torch.manual_seed(1337)

# !curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
#encoder and decoder
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #take a string -> output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #take a list of integers -> output a string

# data
data = torch.tensor(encode(text), dtype=torch.long) #convert to tensor
n = int(0.9 * len(data)) #90% for training, 10% for validation|
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small, random batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Row i (of the embedding table) = logits for the next token given current token i
        # It’s directly acting as a conditional probability table for bigrams
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        logits = self.lm_head(tok_emb + pos_emb) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:    
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            last = idx[:, -1]                           # (B,)
            logits = self.token_embedding_table(last)   # (B, C)
            probs = F.softmax(logits, dim=-1)           # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)     # (B, T+1)
        return idx
    
model = BigramLanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model:
idx = torch.zeros((1, 1), dtype=torch.long, device=device) # starting context
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))