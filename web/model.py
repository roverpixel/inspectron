import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# 1. Hyperparameters & Setup
# -----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 32      # Context length (maximum sequence length)
batch_size = 16      # Independent sequences processed in parallel
n_embd = 64          # Embedding dimension
n_head = 4           # Number of attention heads
n_layer = 3          # Number of transformer blocks
learning_rate = 1e-3
max_iters = 1000

# Toy corpus for demonstration
text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause.
"""

# -----------------
# 2. Character-level Tokenizer
# -----------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# -----------------
# 3. Model Architecture
# -----------------
class Head(nn.Module):
    """One head of causal self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Compute attention scores: (Q @ K^T) / sqrt(d_k)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        # Apply causal mask (cannot peek into future tokens)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention run in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    """Position-wise feed-forward network with ReLU."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block (Attention + MLP)."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections with pre-norm
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token + Positional embeddings
        tok_emb = self.token_embedding_table(idx)             # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb                                 # (B, T, n_embd)
        x = self.blocks(x)                                    # (B, T, n_embd)
        x = self.ln_f(x)                                      # (B, T, n_embd)
        logits = self.lm_head(x)                              # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Autoregressively generate tokens given a prompt."""
        for _ in range(max_new_tokens):
            # Crop context to the maximum context length
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # Sample next token from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------
# 4. Training Loop & Inference
# -----------------
model = MiniLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

for step in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if (step + 1) % 200 == 0:
        print(f"Step {step + 1:4d} | Loss: {loss.item():.4f}")

# Generation demo
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with token 0
generated_tokens = model.generate(context, max_new_tokens=150)[0].tolist()
print("\n--- Generated Sample ---")
print(decode(generated_tokens))
