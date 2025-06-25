### Updated train_model.py with TFLite conversion and feature list export
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

# Paths and constants
DATA_PATH = 'Dataset.csv'
MODEL_DIR = './model'
OUTPUT_COLUMNS = ['Acceleration', 'Braking', 'Clutch', 'Gear', 'Steering']


def main():
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Handle duplicate 'Gear' columns
    cols = df.columns.tolist()
    gear_idxs = [i for i, c in enumerate(cols) if c == 'Gear']
    if len(gear_idxs) > 1:
        for idx in gear_idxs[:-1]:
            cols[idx] = 'Gear_in'
        df.columns = cols

    # Drop unneeded columns
    drop_cols = [
        "CurrentLapTime", "LastLapTime", "DistanceFromStart", "DistanceCovered"
    ]
    drop_cols += [c for c in df.columns if c.startswith("Opponent_")]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Ensure outputs exist
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Clean NaN/inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Prepare X and y
    input_cols = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    X = df[input_cols].values.astype(np.float32)
    y = df[OUTPUT_COLUMNS].values.astype(np.float32)

    # Save feature names for driver
    with open(f"{MODEL_DIR}/input_features.txt", 'w') as f:
        for feat in input_cols:
            f.write(feat + '\n')

    # Z-score normalization
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    np.save(f"{MODEL_DIR}/means.npy", means)
    np.save(f"{MODEL_DIR}/stds.npy", stds)
    X = (X - means) / stds

    # MinMax scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"{MODEL_DIR}/input_scaler.pkl")

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)
    joblib.dump(y_scaler, f"{MODEL_DIR}/output_scaler.pkl")

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Define model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(OUTPUT_COLUMNS), activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train
    model.fit(
        X_train, y_train,
        epochs=50, batch_size=32,
        validation_data=(X_val, y_val), verbose=1
    )

    # Save original model
    model.save(f"{MODEL_DIR}/torcs_model.h5")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(f"{MODEL_DIR}/torcs_model.tflite", 'wb') as f:
        f.write(tflite_model)

    print(f"Saved .h5, .tflite, and feature list in '{MODEL_DIR}'")


if __name__ == '__main__':
    main()