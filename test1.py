import io
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

app = Flask(__name__)
CORS(app)

def prepare_data(data):
    df = pd.DataFrame(data)
    df['createdAt'] = df['createdAt'].apply(lambda x: datetime.fromtimestamp(x['seconds'] + x['nanoseconds'] / 1e9))
    df.set_index('createdAt', inplace=True)

    # Creating moving average features only if there are enough data points
    if len(df) >= 5:
        df['bpm_value_ma'] = df['bpm_value'].rolling(window=5).mean()
        df['blink_count_ma'] = df['blink_count'].rolling(window=5).mean()
        df['highest_blink_duration_ma'] = df['highest_blink_duration'].rolling(window=5).mean()
        # Fill NaN values resulting from the rolling mean calculation
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
    else:
        # If there are not enough data points for moving average, skip adding them
        df['bpm_value_ma'] = df['bpm_value']
        df['blink_count_ma'] = df['blink_count']
        df['highest_blink_duration_ma'] = df['highest_blink_duration']

    features = ['bpm_value', 'blink_count', 'highest_blink_duration', 'bpm_value_ma', 'blink_count_ma', 'highest_blink_duration_ma']
    
    if len(df) == 0:
        raise ValueError("Not enough data points to prepare the dataset.")

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
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(input_shape[-1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X, y):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
    return history

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
        history = train_model(model, X, y)

        # Prepare JSON response
        pred_kpm, pred_bpm, pred_bd = make_predictions(model, X[-1], scaler, features)
        y_pred = model.predict(X)
        mse = calculate_mse(y, y_pred)

        response = {
            'predictions': pred_kpm,
            'predictions_bpm': pred_bpm,
            'predictions_blink_duration': pred_bd,
            'mse': mse
        }

        return jsonify(response)

    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
