import io

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

app = Flask(__name__)
CORS(app)

# Function to create sequences for LSTM input
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        reports = data['reports']
        print(f'reports: {reports}')

        # Extract blink_count from each report and create input array
        input_data = np.array([report['blink_count'] for report in reports if report['blink_count'] is not None])

        # Check if there's sufficient data
        if len(input_data) < 2:
            return jsonify({'error': 'Not enough data to predict.'}), 400

        # Define sequence length
        seq_length = 10

        # Create sequences
        X, y = create_sequences(input_data, seq_length)

        # Reshape X to be 3-dimensional for LSTM input [samples, timesteps, features]
        X = X.reshape(X.shape[0], seq_length, 1)

        # Split data into training and prediction sets (80% training, 20% prediction)
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_pred = X[train_size:]

        # Build LSTM model
        step_per_epoch = len(X_train) // 32
        validation_steps = len(X_pred) // 32
        model = Sequential()
        model.add(Input(shape=(seq_length, 1)))  # Add input layer with defined shape
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # Perform predictions on the prediction set
        y_pred = model.predict(X_pred)

        # Prepare JSON response
        response = {
            'input_data': input_data.tolist(),
            'predictions': y_pred.flatten().tolist()
        }

        return jsonify(response)

    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/plot', methods=['POST'])
def plot():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        reports = data['reports']

        # Extract blink_count from each report and create input array
        input_data = np.array([report['blink_count'] for report in reports if report['blink_count'] is not None])

        # Check if there's sufficient data
        if len(input_data) < 2:
            return jsonify({'error': 'Not enough data to plot.'}), 400

        # Define sequence length
        seq_length = 10

        # Create sequences
        X, y = create_sequences(input_data, seq_length)

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
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

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
        plt.ylabel('Blink Count')
        plt.legend()
        plt.grid(True)

        # Save plot to a BytesIO object
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)
        # plt.close()
        plt.show()

        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
