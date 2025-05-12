
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import json

app = Flask(__name__, static_folder='static')
CORS(app)

# === Load model once on startup ===
model = tf.keras.models.load_model("Bi_LSTM_model_701515_60K_40e.h5", compile=False)
class_to_ados = list(range(4, 21))

TARGET_COLUMNS = [
    'eye_gaze_rx', 'eye_gaze_ry', 'eye_gaze_rz',
    'head_gaze_rx', 'head_gaze_ry', 'head_gaze_rz',
    'skeleton_elbow_left_x', 'skeleton_elbow_left_y', 'skeleton_elbow_left_z',
    'skeleton_elbow_right_x', 'skeleton_elbow_right_y', 'skeleton_elbow_right_z',
    'skeleton_hand_left_x', 'skeleton_hand_left_y', 'skeleton_hand_left_z',
    'skeleton_hand_right_x', 'skeleton_hand_right_y', 'skeleton_hand_right_z',
    'skeleton_head_x', 'skeleton_head_y', 'skeleton_head_z',
    'skeleton_sholder_center_x', 'skeleton_sholder_center_y', 'skeleton_sholder_center_z',
    'skeleton_sholder_left_x', 'skeleton_sholder_left_y', 'skeleton_sholder_left_z',
    'skeleton_sholder_right_x', 'skeleton_sholder_right_y', 'skeleton_sholder_right_z',
    'skeleton_wrist_left_x', 'skeleton_wrist_left_y', 'skeleton_wrist_left_z',
    'skeleton_wrist_right_x', 'skeleton_wrist_right_y', 'skeleton_wrist_right_z'
]

def flatten_json(json_data):
    n = len(json_data['eye_gaze']['rx'])
    rows = []
    for i in range(n):
        row = {
            'eye_gaze_rx': json_data['eye_gaze']['rx'][i],
            'eye_gaze_ry': json_data['eye_gaze']['ry'][i],
            'eye_gaze_rz': json_data['eye_gaze']['rz'][i],
            'head_gaze_rx': json_data['head_gaze']['rx'][i],
            'head_gaze_ry': json_data['head_gaze']['ry'][i],
            'head_gaze_rz': json_data['head_gaze']['rz'][i],
        }
        for joint, values in json_data['skeleton'].items():
            for axis in ['x', 'y', 'z']:
                row[f"skeleton_{joint}_{axis}"] = values[axis][i]
        rows.append(row)
    return pd.DataFrame(rows)

def engineer_features(df):
    df['eye_head_rx_diff'] = df['eye_gaze_rx'] - df['head_gaze_rx']
    df['eye_head_ry_diff'] = df['eye_gaze_ry'] - df['head_gaze_ry']
    df['eye_head_rz_diff'] = df['eye_gaze_rz'] - df['head_gaze_rz']
    df['eye_gaze_magnitude'] = np.sqrt(df['eye_gaze_rx']**2 + df['eye_gaze_ry']**2 + df['eye_gaze_rz']**2)
    df['head_gaze_magnitude'] = np.sqrt(df['head_gaze_rx']**2 + df['head_gaze_ry']**2 + df['head_gaze_rz']**2)
    df['hand_distance'] = np.sqrt(
        (df['skeleton_hand_left_x'] - df['skeleton_hand_right_x'])**2 +
        (df['skeleton_hand_left_y'] - df['skeleton_hand_right_y'])**2 +
        (df['skeleton_hand_left_z'] - df['skeleton_hand_right_z'])**2)
    df['left_hand_height_ratio'] = df['skeleton_hand_left_y'] - df['skeleton_head_y']
    df['right_hand_height_ratio'] = df['skeleton_hand_right_y'] - df['skeleton_head_y']
    df['hand_z_asymmetry'] = df['skeleton_hand_left_z'] - df['skeleton_hand_right_z']
    df['shoulder_asymmetry'] = df['skeleton_sholder_left_y'] - df['skeleton_sholder_right_y']
    df['left_arm_extension'] = np.sqrt(
        (df['skeleton_hand_left_x'] - df['skeleton_elbow_left_x'])**2 +
        (df['skeleton_hand_left_y'] - df['skeleton_elbow_left_y'])**2 +
        (df['skeleton_hand_left_z'] - df['skeleton_elbow_left_z'])**2)
    df['right_arm_extension'] = np.sqrt(
        (df['skeleton_hand_right_x'] - df['skeleton_elbow_right_x'])**2 +
        (df['skeleton_hand_right_y'] - df['skeleton_elbow_right_y'])**2 +
        (df['skeleton_hand_right_z'] - df['skeleton_elbow_right_z'])**2)
    return df

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        json_data = json.load(file)
        df = flatten_json(json_data)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df[TARGET_COLUMNS]
        df = df.interpolate(method='linear', limit_area='inside')
        df = df.fillna(df.mean())
        df = engineer_features(df)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[df.columns] = scaler.fit_transform(df[df.columns])
        input_array = np.expand_dims(df.values, axis=0)
        probs = model.predict(input_array)[0]
        predicted_index = int(np.argmax(probs))
        predicted_score = class_to_ados[predicted_index]
        return str(predicted_score)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'mainn.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
