import numpy as np

class Transformer:
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initialize weights
        self.wq = np.random.randn(input_dim, hidden_dim)
        self.wk = np.random.randn(input_dim, hidden_dim)
        self.wv = np.random.randn(input_dim, hidden_dim)
        self.wo = np.random.randn(hidden_dim, input_dim)
        
        self.wff1 = np.random.randn(input_dim, hidden_dim)
        self.wff2 = np.random.randn(hidden_dim, input_dim)
        
    def attention(self, q, k, v, mask=None):
        d_k = k.shape[-1]
        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = self.softmax(scores)
        output = np.matmul(attention_weights, v)
        return output
    
    def multi_head_attention(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = np.matmul(x, self.wq)
        k = np.matmul(x, self.wk)
        v = np.matmul(x, self.wv)
        
        q = q.reshape(batch_size, seq_len, self.num_heads, -1)
        k = k.reshape(batch_size, seq_len, self.num_heads, -1)
        v = v.reshape(batch_size, seq_len, self.num_heads, -1)
        
        output = self.attention(q, k, v)
        output = output.reshape(batch_size, seq_len, -1)
        output = np.matmul(output, self.wo)
        
        return output
    
    def feed_forward(self, x):
        hidden = np.maximum(0, np.matmul(x, self.wff1))  # ReLU activation
        output = np.matmul(hidden, self.wff2)
        return output
    
    def layer_norm(self, x, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x):
        for _ in range(self.num_layers):
            # Multi-head attention
            attention_output = self.multi_head_attention(x)
            x = self.layer_norm(x + attention_output)
            
            # Feed-forward network
            ff_output = self.feed_forward(x)
            x = self.layer_norm(x + ff_output)
        
        return x

# Example usage
input_dim = 512
hidden_dim = 2048
num_heads = 8
num_layers = 6
batch_size = 32
seq_len = 10

transformer = Transformer(input_dim, hidden_dim, num_heads, num_layers)
input_data = np.random.randn(batch_size, seq_len, input_dim)
output = transformer.forward(input_data)
print(output.shape)  # Should be (batch_size, seq_len, input_dim)