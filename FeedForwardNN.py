import torch
import torch.nn as nn
import torch.optim as optim

class FeedForwardNN(nn.Module):
    def __init__(self, input_size: int, hidden_layers_size_ls: list, output_size: int, dropout_rate: float):
        super(FeedForwardNN, self).__init__()

        layers = [] ## list which will contain all the layers of the model and would be passed on to sequential for final execution

        ## define the input layer
        layers.append(nn.Linear(input_size, hidden_layers_size_ls[0]))
        layers.append(nn.ReLU)
        layers.append(nn.Dropout(dropout_rate)) # adding a dropout to prevent overfitting

        ## define the hidden layers
        for i in range(1, len(hidden_layers_size_ls)):
            layers.append(nn.Linear(hidden_layers_size_ls[i], hidden_layers_size_ls[i+1]))
            layers.append(nn.ReLU)

        ## define the output layer
        layers.append(nn.Linear(hidden_layers_size_ls[-1], output_size))
        layers.append(nn.ReLU)
        
        ## combine all the layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        self.model(x)

## defining and running the NN
input_size = 10
hidden_layers_size_ls = [64, 32, 16]
output_size = 5
learning_rate = 0.01
n_epochs = 10
dropout_rate = 0.2

## define the model
model = FeedForwardNN(input_size=input_size, hidden_layers_size_ls=hidden_layers_size_ls, output_size=output_size, dropout_rate=dropout_rate)

## define the loss function and the optimizer - choosing adam for now
criterion = nn.MSELoss()
optimizer = optim.adam(model.parameters(), lr=learning_rate)

## training loop assuming that there's data splitted as X_train and y_train
for epoch in range(n_epochs):
    ## forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    ## backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch+1 % 10 == 0:
        print(f"Epoch [{epoch+1} / {n_epochs}], loss: {loss.item():.4f}")

with torch.no_grad():
    pred = model(x_test)