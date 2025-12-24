import numpy as np


class LayerNormalization:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.gamma = None
        self.beta = None

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        return normalized * self.gamma + self.beta

class PositionalEncoding:
    def __init__(self, d_model, max_seq_length=5000):
        self.d_model = d_model

        # Create positional encoding matrix
        pe = np.zeros((max_seq_length, d_model))
        position = np.arange(0, max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe[np.newaxis, :, :]  # Add batch dimension

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.shape
        return x.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        # Linear projections
        Q = np.dot(q, self.W_q)  # (batch_size, seq_length, d_model)
        K = np.dot(k, self.W_k)
        V = np.dot(v, self.W_v)

        # Split into heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = self._softmax(scores)
        attention_output = np.matmul(attention_weights, V)

        # Combine heads
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.d_model)

        # Final linear projection
        output = np.dot(attention_output, self.W_o)

        return output, attention_weights

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        # Initialize weights
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # First linear layer
        hidden = np.dot(x, self.W1) + self.b1
        # ReLU activation
        hidden = np.maximum(0, hidden)
        # Second linear layer
        output = np.dot(hidden, self.W2) + self.b2
        return output

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

    def forward(self, x, mask=None):
        # Self attention
        attention_output, _ = self.self_attention.forward(x, x, x, mask)
        x = x + attention_output  # Residual connection
        x = self.norm1.forward(x)  # Layer normalization

        # Feed forward
        ff_output = self.feed_forward.forward(x)
        x = x + ff_output  # Residual connection
        x = self.norm2.forward(x)  # Layer normalization

        return x

class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.norm3 = LayerNormalization()

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self attention
        self_attention_output, _ = self.self_attention.forward(x, x, x, tgt_mask)
        x = x + self_attention_output
        x = self.norm1.forward(x)

        # Cross attention
        cross_attention_output, _ = self.cross_attention.forward(
            x, encoder_output, encoder_output, src_mask)
        x = x + cross_attention_output
        x = self.norm2.forward(x)

        # Feed forward
        ff_output = self.feed_forward.forward(x)
        x = x + ff_output
        x = self.norm3.forward(x)

        return x

class Transformer:
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048):
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model)

        # Create encoder and decoder layers
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

    def encode(self, src, src_mask=None):
        x = self.positional_encoding.forward(src)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward(x, src_mask)

        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.positional_encoding.forward(tgt)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer.forward(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return decoder_output

# Helper function to create masks
def create_padding_mask(seq):
    return (seq != 0).astype(np.float32)[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mask(size):
    mask = np.triu(np.ones((size, size)), k=1)
    return (1 - mask).astype(np.float32)
