import torch
import torch.nn as nn

class SelfAttention_V2(nn.Module):
   def __init__(self, d_in, d_out, qkv_bias=False):
      super().__init__()
      self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
      self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias) 
      self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

   def forward(self, x):
      keys = self.W_key(x)
      queries = self.W_query(x)
      values = self.W_value(x)
      attn_scores = queries @ keys.T
      attn_weights = torch.softmax(
         attn_scores / keys.shape[1]**0.5, dim=-1)
      context_vec = attn_weights @ values
      return context_vec

# Your input data
inputs = torch.tensor([
   [0.43, 0.15, 0.89],
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64],
   [0.22, 0.58, 0.33],
   [0.77, 0.25, 0.10],
   [0.05, 0.80, 0.55]
])

d_in = inputs.shape[1]  # 3
d_out = 2

torch.manual_seed(123)
sa_v2 = SelfAttention_V2(d_in, d_out)

print("=== STEP BY STEP CALCULATION ===")
print(f"Input shape: {inputs.shape}")
print(f"Input:\n{inputs}")
print()

# Let's examine what's inside the Linear layer
print("🔍 INSIDE W_query Linear Layer:")
print(f"W_query.weight shape: {sa_v2.W_query.weight.shape}")
print(f"W_query.weight (transposed):\n{sa_v2.W_query.weight}")
print(f"W_query.bias: {sa_v2.W_query.bias}")  # None since qkv_bias=False
print()

# Now let's manually calculate what happens
print("🧮 MANUAL CALCULATION of queries = self.W_query(inputs):")

# Method 1: Using nn.Linear (what actually happens)
queries_auto = sa_v2.W_query(inputs)
print(f"queries (auto):\n{queries_auto}")

# Method 2: Manual matrix multiplication 
# Note: nn.Linear uses weight matrix differently than manual @
manual_queries = inputs @ sa_v2.W_query.weight.T  # Note the .T!
print(f"queries (manual):\n{manual_queries}")

# Verify they're the same
print(f"Are they equal? {torch.allclose(queries_auto, manual_queries)}")
print()

# Let's break it down even further
print("🔬 DETAILED BREAKDOWN:")
print("Step 1 - Input x Weight^T:")
print(f"inputs: {inputs.shape}")
print(f"W_query.weight.T: {sa_v2.W_query.weight.T.shape}")
print(f"Result: {(inputs @ sa_v2.W_query.weight.T).shape}")

# Show actual numbers for first row
print("\nFirst token calculation:")
print(f"First token: {inputs[0]}")  # [0.43, 0.15, 0.89]
print(f"Weight matrix:\n{sa_v2.W_query.weight}")

first_query = inputs[0] @ sa_v2.W_query.weight.T
print(f"First query manual: {first_query}")
print(f"First query auto: {queries_auto[0]}")