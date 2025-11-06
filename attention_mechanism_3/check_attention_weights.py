# Final execution with precise output
import torch

torch.manual_seed(123)

inputs = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]
])

print("=== STEP-BY-STEP VERIFICATION ===")
print("1. Inputs shape:", inputs.shape)
print("2. Token 2 (inputs[1]):", inputs[1])

d_in, d_out = 3, 2
W_query = torch.tensor([[0.2961, 0.5166],
                        [0.8137, 0.9738],
                        [0.4906, 0.4032]])  # From torch.rand with seed 123
W_key = torch.tensor([[0.2984, 0.5341],
                      [0.5385, 0.9329],
                      [0.9969, 0.6978]])    # From torch.rand with seed 123

print("3. W_query:\n", W_query)
print("4. W_key:\n", W_key)

# Calculate query_2
query_2 = inputs[1] @ W_query
print("5. query_2 = [0.55, 0.87, 0.66] @ W_query:")
print(f"   = 0.55*[{W_query[0,0]:.4f},{W_query[0,1]:.4f}] + 0.87*[{W_query[1,0]:.4f},{W_query[1,1]:.4f}] + 0.66*[{W_query[2,0]:.4f},{W_query[2,1]:.4f}]")
print(f"   = [{0.55*W_query[0,0] + 0.87*W_query[1,0] + 0.66*W_query[2,0]:.4f}, {0.55*W_query[0,1] + 0.87*W_query[1,1] + 0.66*W_query[2,1]:.4f}]")
print("   query_2 =", query_2)

# Calculate keys
keys = inputs @ W_key
keys_2 = keys[1]
print("6. keys_2 = inputs[1] @ W_key =", keys_2)

# Calculate dot product
attn_score_22 = query_2.dot(keys_2)
print("7. attn_score_22 = query_2 · keys_2:")
print(f"   = {query_2[0]:.4f}*{keys_2[0]:.4f} + {query_2[1]:.4f}*{keys_2[1]:.4f}")
print(f"   = {query_2[0]*keys_2[0]:.4f} + {query_2[1]*keys_2[1]:.4f}")
print(f"   = {attn_score_22:.4f}")