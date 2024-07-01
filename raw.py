import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Sample data
data = [
    {'bpm_value': 541, 'blink_count': 10, 'highest_blink_duration': 239, 'createdAt': {'seconds': 1719671461, 'nanoseconds': 435000000}},
    {'bpm_value': 555, 'blink_count': 12, 'highest_blink_duration': 319, 'createdAt': {'seconds': 1719671521, 'nanoseconds': 696000000}},
    {'bpm_value': 561, 'blink_count': 23, 'highest_blink_duration': 159, 'createdAt': {'seconds': 1719671581, 'nanoseconds': 18000000}},
    {'bpm_value': 509, 'blink_count': 26, 'highest_blink_duration': 139, 'createdAt': {'seconds': 1719671642, 'nanoseconds': 493000000}},
    {'bpm_value': 540, 'blink_count': 23, 'highest_blink_duration': 140, 'createdAt': {'seconds': 1719671701, 'nanoseconds': 460000000}},
    {'bpm_value': 546, 'blink_count': 14, 'highest_blink_duration': 159, 'createdAt': {'seconds': 1719671760, 'nanoseconds': 299000000}},
    {'bpm_value': 573, 'blink_count': 36, 'highest_blink_duration': 159, 'createdAt': {'seconds': 1719671760, 'nanoseconds': 328000000}},
    {'bpm_value': 543, 'blink_count': 21, 'highest_blink_duration': 140, 'createdAt': {'seconds': 1719671821, 'nanoseconds': 722000000}},
    {'bpm_value': 1023, 'blink_count': 21, 'highest_blink_duration': 179, 'createdAt': {'seconds': 1719672002, 'nanoseconds': 777000000}},
    {'bpm_value': 552, 'blink_count': 12, 'highest_blink_duration': 139, 'createdAt': {'seconds': 1719672060, 'nanoseconds': 922000000}},
    {'bpm_value': 541, 'blink_count': 22, 'highest_blink_duration': 21, 'createdAt': {'seconds': 1719672131, 'nanoseconds': 993000000}},
    {'bpm_value': 539, 'blink_count': 46, 'highest_blink_duration': 260, 'createdAt': {'seconds': 1719672180, 'nanoseconds': 445000000}},
    {'bpm_value': 573, 'blink_count': 15, 'highest_blink_duration': 4, 'createdAt': {'seconds': 1719672240, 'nanoseconds': 416000000}},
    {'bpm_value': 536, 'blink_count': 25, 'highest_blink_duration': 139, 'createdAt': {'seconds': 1719672302, 'nanoseconds': 541000000}},
    {'bpm_value': 540, 'blink_count': 16, 'highest_blink_duration': 139, 'createdAt': {'seconds': 1719672360, 'nanoseconds': 563000000}},
    {'bpm_value': 547, 'blink_count': 30, 'highest_blink_duration': 11, 'createdAt': {'seconds': 1719672463, 'nanoseconds': 95000000}},
    {'bpm_value': 510, 'blink_count': 14, 'highest_blink_duration': 94, 'createdAt': {'seconds': 1719672481, 'nanoseconds': 25000000}},
    {'bpm_value': 527, 'blink_count': 10, 'highest_blink_duration': 6, 'createdAt': {'seconds': 1719672590, 'nanoseconds': 806000000}},
    {'bpm_value': 0, 'blink_count': 31, 'highest_blink_duration': 139, 'createdAt': {'seconds': 1719672611, 'nanoseconds': 759000000}}
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract features
features = df[['bpm_value', 'blink_count', 'highest_blink_duration']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Create sequences
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

sequence_length = 3
X, y = create_sequences(scaled_features, sequence_length)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape X for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build the LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50),
    Dense(X_train.shape[2])
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=1, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Inverse transform the actual values
y_test_actual = scaler.inverse_transform(y_test)

# Calculate metrics
mse_bpm = mean_squared_error(y_test_actual[:, 0], predictions[:, 0])
mae_bpm = mean_absolute_error(y_test_actual[:, 0], predictions[:, 0])
mse_blink_count = mean_squared_error(y_test_actual[:, 1], predictions[:, 1])
mae_blink_count = mean_absolute_error(y_test_actual[:, 1], predictions[:, 1])
mse_highest_blink_duration = mean_squared_error(y_test_actual[:, 2], predictions[:, 2])
mae_highest_blink_duration = mean_absolute_error(y_test_actual[:, 2], predictions[:, 2])

print(f'BPM MSE: {mse_bpm}, BPM MAE: {mae_bpm}')
print(f'Blink Count MSE: {mse_blink_count}, Blink Count MAE: {mae_blink_count}')
print(f'Highest Blink Duration MSE: {mse_highest_blink_duration}, Highest Blink Duration MAE: {mae_highest_blink_duration}')

# Calculate average actual values for each feature in the test set
avg_actual_bpm = np.mean(y_test_actual[:, 0])
avg_actual_blink_count = np.mean(y_test_actual[:, 1])
avg_actual_highest_blink_duration = np.mean(y_test_actual[:, 2])

# Calculate accuracy
accuracy_bpm = (1 - (mae_bpm / avg_actual_bpm)) * 100
accuracy_blink_count = (1 - (mae_blink_count / avg_actual_blink_count)) * 100
accuracy_highest_blink_duration = (1 - (mae_highest_blink_duration / avg_actual_highest_blink_duration)) * 100

print(f'BPM Accuracy: {accuracy_bpm:.2f}%')
print(f'Blink Count Accuracy: {accuracy_blink_count:.2f}%')
print(f'Highest Blink Duration Accuracy: {accuracy_highest_blink_duration:.2f}%')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual[:, 0], color='blue', label='Actual BPM')
plt.plot(predictions[:, 0], color='red', label='Predicted BPM')
plt.title('LSTM Time Series Prediction for BPM')
plt.xlabel('Time')
plt.ylabel('BPM Value')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test_actual[:, 1], color='blue', label='Actual Blink Count')
plt.plot(predictions[:, 1], color='red', label='Predicted Blink Count')
plt.title('LSTM Time Series Prediction for Blink Count')
plt.xlabel('Time')
plt.ylabel('Blink Count')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test_actual[:, 2], color='blue', label='Actual Highest Blink Duration')
plt.plot(predictions[:, 2], color='red', label='Predicted Highest Blink Duration')
plt.title('LSTM Time Series Prediction for Highest Blink Duration')
plt.xlabel('Time')
plt.ylabel('Highest Blink Duration')
plt.legend()
plt.show()
