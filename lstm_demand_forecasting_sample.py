import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 1. Data Preparation
def prepare_data(data, look_back, forecast_horizon, features):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back:i + look_back + forecast_horizon, 0])
    return np.array(X), np.array(y)

# 2. Create and compile the LSTM model
def create_model(input_shape, output_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(output_shape)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 3. Main function
def lstm_forecast(data, exog_features, look_back, forecast_horizon):
    # Combine demand data with exogenous features
    combined_data = pd.concat([data, exog_features], axis=1)
    
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data)
    
    # Prepare data for LSTM
    X, y = prepare_data(scaled_data, look_back, forecast_horizon, exog_features.shape[1])
    
    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train the model
    model = create_model((look_back, combined_data.shape[1]), forecast_horizon)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform predictions
    train_predict = scaler.inverse_transform(np.column_stack((train_predict, np.zeros((train_predict.shape[0], combined_data.shape[1]-1))))[:, 0])
    test_predict = scaler.inverse_transform(np.column_stack((test_predict, np.zeros((test_predict.shape[0], combined_data.shape[1]-1))))[:, 0])
    
    return train_predict, test_predict

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    demand = np.random.randint(50, 200, size=len(dates))
    temperature = np.random.uniform(10, 30, size=len(dates))
    is_weekend = (dates.dayofweek >= 5).astype(int)
    
    data = pd.DataFrame({'demand': demand}, index=dates)
    exog_features = pd.DataFrame({'temperature': temperature, 'is_weekend': is_weekend}, index=dates)
    
    # Run the LSTM forecast
    train_predict, test_predict = lstm_forecast(data, exog_features, look_back=30, forecast_horizon=7)
    
    print("Training set predictions shape:", train_predict.shape)
    print("Test set predictions shape:", test_predict.shape)