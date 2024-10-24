import numpy as np

class Autoencoder:
    def __init__(self, input_size, hidden_size):
        """
        Initialize autoencoder with input size and hidden layer size
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        # Xavier/Glorot initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, input_size))
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """
        Forward pass through the network
        Returns both encoded representation and reconstruction
        """
        # Encoder
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Decoder
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a1, self.a2
    
    def backward(self, X, learning_rate=0.01):
        """
        Backward pass to update weights using gradient descent
        """
        m = X.shape[0]
        
        # Calculate gradients
        # Output layer
        dz2 = self.a2 - X
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
    def train(self, X, epochs=100, learning_rate=0.01, batch_size=32):
        """
        Train the autoencoder
        """
        m = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            
            # Mini-batch training
            for i in range(0, m, batch_size):
                batch = X_shuffled[i:min(i + batch_size, m)]
                
                # Forward pass
                _, reconstruction = self.forward(batch)
                
                # Backward pass
                self.backward(batch, learning_rate)
            
            # Calculate loss for the epoch
            _, reconstruction = self.forward(X)
            loss = np.mean(np.square(X - reconstruction))
            losses.append(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return losses
    
    def encode(self, X):
        """Get encoded representation"""
        encoded, _ = self.forward(X)
        return encoded
    
    def decode(self, encoded):
        """Decode from encoded representation"""
        reconstruction = self.sigmoid(np.dot(encoded, self.W2) + self.b2)
        return reconstruction
    
    def reconstruct(self, X):
        """Get full reconstruction"""
        _, reconstruction = self.forward(X)
        return reconstruction

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(1000, 20)  # 1000 samples, 20 features
    
    # Create and train autoencoder
    autoencoder = Autoencoder(input_size=20, hidden_size=10)
    losses = autoencoder.train(X, epochs=100, learning_rate=0.1, batch_size=32)
    
    # Test reconstruction
    test_sample = X[:5]
    reconstruction = autoencoder.reconstruct(test_sample)
    
    print("\nOriginal data:")
    print(test_sample[:2])
    print("\nReconstructed data:")
    print(reconstruction[:2])