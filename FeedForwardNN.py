import torch
import torch.nn as nn
import torch.optim as optim


class FeedForwardNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_layers_size_ls: list, output_size: int, dropout_rate: float
    ):
        super(FeedForwardNN, self).__init__()

        layers = []

        # input layer
        layers.append(nn.Linear(input_size, hidden_layers_size_ls[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # hidden layers
        for i in range(len(hidden_layers_size_ls) - 1):
            layers.append(nn.Linear(hidden_layers_size_ls[i], hidden_layers_size_ls[i + 1]))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(hidden_layers_size_ls[-1], output_size))

        # combine all the layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


## defining and running the NN (example usage)
input_size = 10
hidden_layers_size_ls = [64, 32, 16]
output_size = 5
learning_rate = 0.01
n_epochs = 10
dropout_rate = 0.2

if __name__ == "__main__":
    model = FeedForwardNN(
        input_size=input_size,
        hidden_layers_size_ls=hidden_layers_size_ls,
        output_size=output_size,
        dropout_rate=dropout_rate,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # dummy training data to make the example runnable
    X_train = torch.randn(100, input_size)
    y_train = torch.randn(100, output_size)
    x_test = torch.randn(10, input_size)

    model.train()
    for epoch in range(n_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print progress each epoch with a simple progress bar
        bar_len = 30
        filled_len = int(bar_len * (epoch + 1) / n_epochs)
        bar = "=" * filled_len + "-" * (bar_len - filled_len)
        print(f"Epoch [{epoch + 1}/{n_epochs}] [{bar}] loss: {loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred = model(x_test)
