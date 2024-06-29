import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Generate synthetic time series data
np.random.seed(0)
time_series_data = 50 + np.cumsum(np.random.randn(1000))

# Function to create sequences for LSTM input
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Define sequence length
seq_length = 50

# Create sequences
X, y = create_sequences(time_series_data, seq_length)

# Reshape X to be 3-dimensional for LSTM input [samples, timesteps, features]
X = X.reshape(X.shape[0], seq_length, 1)

# Split data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Forecast on test data
y_pred = model.predict(X_test)

# Plotting
plt.figure(figsize=(14, 7))

# Plotting training data
plt.plot(np.arange(len(X_train)), np.squeeze(X_train[:, -1]), label='Training Data', color='blue')

# Plotting actual test data
plt.plot(np.arange(len(X_train), len(X_train) + len(y_test)), y_test, label='Actual Test Data', color='green')

# Plotting LSTM forecast
plt.plot(np.arange(len(X_train), len(X_train) + len(y_test)), y_pred, label='LSTM Forecast', color='red')

plt.title('LSTM Time Series Forecasting Example')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
