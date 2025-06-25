import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import os

# Paths and constants
DATA_PATH = 'final_data.csv'
MODEL_DIR = './model'
OUTPUT_COLUMNS = ['Acceleration', 'Braking', 'Clutch', 'Gear', 'Steering']
H5_MODEL_PATH = os.path.join(MODEL_DIR, 'torcs_model.h5')
INPUT_FEATS_PATH = os.path.join(MODEL_DIR, 'input_features.txt')
MEANS_PATH = os.path.join(MODEL_DIR, 'means.npy')
STDS_PATH = os.path.join(MODEL_DIR, 'stds.npy')
SCALER_IN_PATH = os.path.join(MODEL_DIR, 'input_scaler.pkl')
SCALER_OUT_PATH = os.path.join(MODEL_DIR, 'output_scaler.pkl')
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'torcs_model.tflite')


def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(OUTPUT_COLUMNS), activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model


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
    ] + [c for c in df.columns if c.startswith("Opponent_")]
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

    # Save feature names if first run
    if not os.path.exists(INPUT_FEATS_PATH):
        with open(INPUT_FEATS_PATH, 'w') as f:
            for feat in input_cols:
                f.write(feat + '\n')

    # Z-score normalization
    means = np.load(MEANS_PATH) if os.path.exists(MEANS_PATH) else X.mean(axis=0)
    stds = np.load(STDS_PATH) if os.path.exists(STDS_PATH) else X.std(axis=0)
    if not os.path.exists(MEANS_PATH): np.save(MEANS_PATH, means)
    if not os.path.exists(STDS_PATH): np.save(STDS_PATH, stds)
    X_norm = (X - means) / stds

    # MinMax scaling
    scaler = joblib.load(SCALER_IN_PATH) if os.path.exists(SCALER_IN_PATH) else MinMaxScaler()
    X_scaled = scaler.transform(X_norm) if os.path.exists(SCALER_IN_PATH) else scaler.fit_transform(X_norm)
    if not os.path.exists(SCALER_IN_PATH): joblib.dump(scaler, SCALER_IN_PATH)

    y_scaler = joblib.load(SCALER_OUT_PATH) if os.path.exists(SCALER_OUT_PATH) else MinMaxScaler()
    y_scaled = y_scaler.transform(y) if os.path.exists(SCALER_OUT_PATH) else y_scaler.fit_transform(y)
    if not os.path.exists(SCALER_OUT_PATH): joblib.dump(y_scaler, SCALER_OUT_PATH)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Load or build model
    if os.path.exists(H5_MODEL_PATH):
        print("Loading existing model for continued training...")
        # Load without compile to avoid deserialization issues
        model = load_model(H5_MODEL_PATH, compile=False)
        # Re-compile with explicit loss and metrics
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    else:
        print("No existing model found. Building a new model...")
        model = build_model(X_train.shape[1])

    # Continue training
    model.fit(
        X_train, y_train,
        epochs=100, batch_size=64,
        validation_data=(X_val, y_val), verbose=1
    )

    # Save model
    model.save(H5_MODEL_PATH)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    print(f"Continued training complete. Saved updates in '{MODEL_DIR}'")


if __name__ == '__main__':
    main()
