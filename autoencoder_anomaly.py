import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_autoencoder(df, sigma_threshold=3, num_epochs=50, batch_size=32, seed=42):
    # Set the random seed for reproducibility
    set_random_seed(seed)

    # Preprocess the data and scale the features
    features = df.drop(columns=["study_site_subject_id"])
    scaler = MinMaxScaler()
    processed_features = scaler.fit_transform(features)

    # Convert to PyTorch tensors
    tensor_data = torch.tensor(processed_features, dtype=torch.float32)
    dataset = TensorDataset(tensor_data, tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the autoencoder
    input_dim = processed_features.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the autoencoder
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in dataloader:
            inputs, _ = data
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Plotting the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()
    plt.show()

    # Calculating reconstruction error
    with torch.no_grad():
        reconstructed_data = model(tensor_data).numpy()

    # Inverse transform the scaled reconstructed data to original scale
    reconstructed_data_original_scale = scaler.inverse_transform(reconstructed_data)

    # Calculate reconstruction error for each feature
    feature_reconstruction_error = np.square(processed_features - reconstructed_data)
    df_feature_errors = pd.DataFrame(feature_reconstruction_error, columns=features.columns)
    df_feature_errors["study_site_subject_id"] = df["study_site_subject_id"].values

    # Calculate mean reconstruction error for each row
    reconstruction_error = np.mean(feature_reconstruction_error, axis=1)
    threshold = np.mean(reconstruction_error) + sigma_threshold * np.std(reconstruction_error)

    # Identifying anomalies
    df["reconstruction_error"] = reconstruction_error
    df["anomaly_flag"] = reconstruction_error > threshold

    # Create DataFrame for reconstructed values in original scale
    df_reconstructed = pd.DataFrame(reconstructed_data_original_scale, columns=features.columns)
    df_reconstructed["study_site_subject_id"] = df["study_site_subject_id"].values

    return df, df_feature_errors, df_reconstructed


if __name__ == "__main__":
    # Example run with synthetic data
    processed_features = np.random.randn(100, 50)
    df_example = pd.DataFrame(
        processed_features, columns=[f"feat_{i}" for i in range(processed_features.shape[1])]
    )
    df_example["study_site_subject_id"] = range(len(df_example))

    ae_res_df, error_features, df_reconstructed = train_autoencoder(df_example)
