Transformer Encoder From Scratch

This project implements the core components of a Transformer encoder block from scratch using NumPy, with the goal of understanding how modern language models like GPT, BERT, and LLaMA work internally.

Instead of using deep learning frameworks like PyTorch or TensorFlow, this implementation focuses on the mathematical operations behind transformers, helping build intuition about attention mechanisms and transformer architectures.

What is Implemented

The project builds a complete Transformer Encoder Block, including:

1. Multi-Head Self Attention

Implements scaled dot-product attention and multiple attention heads.

Key steps implemented:

Linear projections for Query, Key, Value

Splitting embeddings into multiple heads

Scaled dot-product attention

Softmax attention weights

Concatenation of attention heads

Final projection layer

2. Position-wise Feed Forward Network

Each token embedding passes through a small neural network:

FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

This introduces non-linearity and increases model capacity.

3. Layer Normalization

Layer normalization stabilizes training by normalizing token embeddings across the feature dimension.

4. Residual Connections

Residual connections are used around attention and feed-forward layers:

x + Attention(x)
x + FFN(x)

These help deep transformer models train effectively.

Encoder Block Architecture

The final encoder block implemented follows the standard transformer architecture:

Input
↓
Multi-Head Attention
↓
Add & LayerNorm
↓
Feed Forward Network
↓
Add & LayerNorm
↓
Output

Project Structure
transformer-from-scratch/
│
├── transformer_encoder_numpy.ipynb
├── README.md

Main components implemented in the notebook:

softmax

layer_norm

multi_head_attention

feed_forward

encoder_block

Example Usage

Example configuration used in the notebook:

Batch Size: 2
Sequence Length: 5
Embedding Dimension (d_model): 16
Number of Attention Heads: 4
Feed Forward Dimension: 64

Example run:

Input shape: (2, 5, 16)
Output shape: (2, 5, 16)

The output shape matches the input shape, which allows stacking multiple transformer layers.

Why I Built This

Transformers are the foundation of most modern AI systems, but many implementations hide the details behind frameworks.

This project was built to:

understand the mechanics of attention

implement transformers step-by-step

visualize the data flow inside a transformer block

strengthen understanding of deep learning architectures

Possible Improvements

Future extensions to this project could include:

positional encoding

causal masking (decoder attention)

full transformer encoder stack

transformer decoder implementation

PyTorch version for training

attention visualization

References

"Attention Is All You Need" (Vaswani et al., 2017)

The Illustrated Transformer by Jay Alammar

Transformer implementations in PyTorch
