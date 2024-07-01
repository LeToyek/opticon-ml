import io
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

    SEQ_LENGTH = 3
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
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50),
        Dense(input_shape[-1])
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X, y):
    model.fit(X, y, epochs=50, batch_size=1, validation_split=0.2)

def make_predictions(model, scaled_data, scaler, features):
    SEQ_LENGTH = 3
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

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

def calculate_accuracy(y_true, y_pred):
    mae_bpm = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_blink_count = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    mae_highest_blink_duration = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
    
    avg_actual_bpm = np.mean(y_true[:, 0])
    avg_actual_blink_count = np.mean(y_true[:, 1])
    avg_actual_highest_blink_duration = np.mean(y_true[:, 2])

    accuracy_bpm = (1 - (mae_bpm / avg_actual_bpm)) * 100
    accuracy_blink_count = (1 - (mae_blink_count / avg_actual_blink_count)) * 100
    accuracy_highest_blink_duration = (1 - (mae_highest_blink_duration / avg_actual_highest_blink_duration)) * 100

    return accuracy_bpm, accuracy_blink_count, accuracy_highest_blink_duration

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        reports = data['reports']
        
        X, y, scaler, features = prepare_data(reports)

        model = build_model((X.shape[1], X.shape[2]))
        train_model(model, X, y)

        pred_kpm, pred_bpm, pred_bd = make_predictions(model, X[-1], scaler, features)
        y_pred = model.predict(X)
        mse, mae = calculate_metrics(y, y_pred)
        accuracy_bpm, accuracy_blink_count, accuracy_highest_blink_duration = calculate_accuracy(y, y_pred)
        
        print(f"Accuracy BPM: {accuracy_bpm}")
        print(f"Accuracy Blink Count: {accuracy_blink_count}")
        print(f"Accuracy Highest Blink Duration: {accuracy_highest_blink_duration}")
        
        response = {
            'predictions': pred_kpm,
            'predictions_bpm': pred_bpm,
            'predictions_blink_duration': pred_bd,
            'mse': mse,
            'mae': mae,
            'accuracy_bpm': accuracy_bpm,
            'accuracy_blink_count': accuracy_blink_count,
            'accuracy_highest_blink_duration': accuracy_highest_blink_duration
        }

        return jsonify(response)

    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/plot', methods=['POST'])
def plot():
    try:
        data = request.get_json(force=True)
        reports = data['reports']

        input_data = np.array([report['blink_count'] for report in reports if report['blink_count'] is not None])

        if len(input_data) < 2:
            return jsonify({'error': 'Not enough data to plot.'}), 400

        seq_length = 3
        X, y = create_sequences(input_data.reshape(-1, 1), seq_length)

        X = X.reshape(X.shape[0], seq_length, 1)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = Sequential()
        model.add(LSTM(50, input_shape=(seq_length, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

        y_pred = model.predict(X_test)

        plt.figure(figsize=(14, 7))
        plt.plot(np.arange(len(X_train)), np.squeeze(X_train[:, -1]), label='Training Data', color='blue')
        plt.plot(np.arange(len(X_train), len(X_train) + len(y_test)), y_test, label='Actual Test Data', color='green')
        plt.plot(np.arange(len(X_train), len(X_train) + len(y_test)), y_pred, label='LSTM Forecast', color='red')

        plt.title('LSTM Time Series Forecasting Example')
        plt.xlabel('Time')
        plt.ylabel('Blink Count')
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
