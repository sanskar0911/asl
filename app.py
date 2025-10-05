from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import os
import csv
import time
import base64
import numpy as np
from gesture_recognition import load_and_train_model, predict_gesture

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'  # For SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSockets

# Load model and accuracy
model, current_accuracy = load_and_train_model()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Globals (unchanged)
latest_prediction = "No gesture detected"
latest_landmarks = []
last_prediction = None
speak_enabled = True
last_spoken = 0
word_mode = False
word_buffer = []

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    global latest_prediction, latest_landmarks, last_prediction, last_spoken, word_mode, word_buffer
    
    # Decode base64 frame from client
    img_data = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process frame (identical to original gen_frames)
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            x_vals = [lm.x - base_x for lm in hand_landmarks.landmark]
            y_vals = [lm.y - base_y for lm in hand_landmarks.landmark]
            landmarks = x_vals + y_vals

            latest_landmarks = landmarks
            prediction = predict_gesture(model, landmarks)

            if prediction:
                if prediction == last_prediction:
                    if time.time() - last_spoken > 1.5:
                        latest_prediction = prediction
                        if word_mode:
                            word_buffer.append(prediction)
                        print(f"[🔊] Speaking: {prediction}")
                        emit('speak', prediction) if speak_enabled else None
                        last_spoken = time.time()
                else:
                    last_prediction = prediction

            cv2.putText(frame, f"Gesture: {latest_prediction}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        latest_prediction = "No gesture detected"
        latest_landmarks = []
        last_prediction = None

    # Encode processed frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    processed_frame = base64.b64encode(buffer).decode('utf-8')
    emit('processed_frame', f"data:image/jpeg;base64,{processed_frame}")

@app.route('/get_prediction')
def get_prediction():
    return jsonify({"prediction": latest_prediction})

@app.route('/get_accuracy')
def get_accuracy():
    return jsonify({"accuracy": round(current_accuracy * 100, 2)})

@app.route('/train_model_csv', methods=['POST'])
def train_model_csv():
    global model, current_accuracy
    try:
        print("[🧠] Training model from CSV...")
        model, current_accuracy = load_and_train_model()
        print("Accuracy:", current_accuracy)
        return jsonify({"message": f"Training complete. Accuracy: {round(current_accuracy * 100, 2)}%"})
    except Exception as e:
        print("[❌] Training failed:", e)
        return jsonify({"message": "Training failed", "error": str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    return train_model_csv()

@app.route('/toggle_speak', methods=['POST'])
def toggle_speak():
    global speak_enabled
    data = request.get_json()
    speak_enabled = data.get('enabled', True)
    return jsonify({"speak_enabled": speak_enabled})

@socketio.on('collect_frame')
def handle_collect_frame(data):
    label = request.json.get("label", "").strip() if request.json else ""
    
    if not label:
        emit('collection_status', {"status": "error", "message": "Label is required."})
        return

    try:
        # Decode frame
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        filename = "gesture_data.csv"
        file_exists = os.path.isfile(filename)
        samples = 0
        max_samples = 100

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_vals = [lm.x for lm in hand_landmarks.landmark]
                y_vals = [lm.y for lm in hand_landmarks.landmark]
                base_x = x_vals[0]
                base_y = y_vals[0]
                norm_x = [x - base_x for x in x_vals]
                norm_y = [y - base_y for y in y_vals]
                landmarks = norm_x + norm_y

                if len(landmarks) == 42:
                    with open(filename, "a", newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["label"] + [f"f{i}" for i in range(42)])
                            file_exists = True
                        writer.writerow([label] + landmarks)
                        samples += 1
                        print(f"[+] Collected sample {samples} for '{label}'.")
        
        emit('collection_status', {"status": "success", "message": f"Collected {samples} samples for '{label}'."})
    except Exception as e:
        emit('collection_status', {"status": "error", "message": f"Failed to collect data: {str(e)}"})

@app.route('/start_collection', methods=['POST'])
def start_collection():
    return jsonify({"status": "start", "message": "Send frames via WebSocket for collection."})

@app.route('/record_gesture', methods=['POST'])
def record_gesture():
    return start_collection()

@app.route('/model_accuracy')
def model_accuracy():
    return get_accuracy()

@app.route('/start_word_mode', methods=['POST'])
def start_word_mode():
    global word_mode, word_buffer
    word_mode = True
    word_buffer = []
    return jsonify({"message": "Word mode started"})

@app.route('/finish_word', methods=['POST'])
def finish_word():
    global word_mode, word_buffer
    word_mode = False
    final_word = "".join(word_buffer)
    return jsonify({"word": final_word})

if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000/")
    socketio.run(app, debug=True)