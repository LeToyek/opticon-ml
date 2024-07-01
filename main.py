import io
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

app = Flask(__name__)
CORS(app)

def prepare_data(data):
    df = pd.DataFrame(data)
    
    df['createdAt'] = df['createdAt'].apply(lambda x: datetime.fromtimestamp(x['seconds'] + x['nanoseconds'] / 1e9))
    df.set_index('createdAt', inplace=True)

    features = ['bpm_value', 'blink_count', 'highest_blink_duration']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    SEQ_LENGTH = 10
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    return X, y, scaler, features

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(input_shape[-1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X, y):
    model.fit(X, y, epochs=20, batch_size=16)

def make_predictions(model, scaled_data, scaler, features):
    SEQ_LENGTH = 10
    last_sequence = scaled_data[-SEQ_LENGTH:]
    predictions = []

    for _ in range(10):
        pred = model.predict(last_sequence[np.newaxis, :, :])
        predictions.append(pred[0])
        last_sequence = np.append(last_sequence[1:], pred, axis=0)

    predictions = scaler.inverse_transform(predictions)

    last_timestamp = datetime.now()  # Adjust as per actual last timestamp from the dataset
    predicted_data = []
    predicted_bpm = []
    predicted_blink_duration = []

    for pred in predictions:
        last_timestamp += timedelta(minutes=1)
        predicted_data.append(int(pred[1]))
        predicted_bpm.append(int(pred[0]))
        predicted_blink_duration.append(int(pred[2]))

    return predicted_data, predicted_bpm, predicted_blink_duration

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        reports = data['reports']
        
        X, y, scaler, features = prepare_data(reports)

        model = build_model((X.shape[1], X.shape[2]))
        train_model(model, X, y)

        # Prepare JSON response
        pred_kpm, pred_bpm, pred_bd = make_predictions(model, X[-1], scaler, features)
        y_pred = model.predict(X)
        mse = calculate_mse(y, y_pred)
        print(f"MSEE: {mse}")

        response = {
            'predictions': pred_kpm,
            'predictions_bpm': pred_bpm,
            'predictions_blink_duration': pred_bd,
            'mse': mse
        }

        return jsonify(response)

    except Exception as e:
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
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
