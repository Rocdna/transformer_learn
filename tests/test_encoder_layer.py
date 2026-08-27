import torch
from model.encoder_layer import TransformerEncoderLayer

B = 2
L = 5
D = 8

layer = TransformerEncoderLayer(
    d_model=8,
    num_heads=2,
    d_ff=32
)

x = torch.randn(B, L, D)

print("Input:", x.shape)

output = layer(x)

print("Output:", output.shape)





