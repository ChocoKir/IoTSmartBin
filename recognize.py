import cv2
import numpy as np
import tensorflow as tf
import time

# --- SETUP ---
np.set_printoptions(suppress=True)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels from the text file
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize webcam and set resolution
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- OBJECT COUNTER SETUP ---
# Initialize dictionary to store counts
object_counts = {"01 Biodegradable": 0, "02 Non-Biodegradable": 0}
# Keep track of the last time an object was seen to prevent double-counting
last_seen = {"01 Biodegradable": 0, "02 Non-Biodegradable": 0}
timeout = 1.5 # seconds

# --- REAL-TIME RECOGNITION LOOP ---
while True:
    ret, image = camera.read()
    if not ret:
        break

    frame_height, frame_width, _ = image.shape

    # --- Detection Zone Setup ---
    square_size = 280
    start_x = (frame_width - square_size) // 2
    start_y = (frame_height - square_size) // 2
    end_x = start_x + square_size
    end_y = start_y + square_size
    roi = image[start_y:end_y, start_x:end_x] # Region of Interest

    # --- Preprocessing and Prediction ---
    resized_roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
    image_array = np.asarray(rgb_roi, dtype=np.float32).reshape(1, 224, 224, 3)
    normalized_image_array = (image_array / 127.5) - 1

    interpreter.set_tensor(input_details[0]['index'], normalized_image_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(prediction)
    full_class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # --- Live Counter Logic ---
    if confidence_score > 0.60:
        display_name = full_class_name[3:]
        current_time = time.time()
        # Only count if the timeout has passed for this specific class
        if current_time - last_seen[full_class_name] > timeout:
            object_counts[full_class_name] += 1
            last_seen[full_class_name] = current_time
    else:
        display_name = "Uncertain"

    # --- UI Display ---
    # Determine color based on confidence
    if confidence_score > 0.9:
        color = (0, 255, 0)      # Green
    elif confidence_score > 0.6:
        color = (0, 255, 255)    # Yellow
    else:
        color = (0, 0, 255)      # Red

    # Draw detection square
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 4)

    # Draw confidence bar
    bar_width = 20
    bar_height = square_size
    bar_x = end_x + 10
    bar_y = start_y
    fill_height = int(bar_height * confidence_score)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), 2)
    cv2.rectangle(image, (bar_x, bar_y + bar_height - fill_height), (bar_x + bar_width, bar_y + bar_height), color, -1)

    # Draw prediction and confidence text
    prediction_text = f"Prediction: {display_name}"
    confidence_text = f"Confidence: {int(confidence_score * 100)}%"
    cv2.putText(image, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(image, confidence_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Draw live counter dashboard at the bottom
    bio_count = object_counts["01 Biodegradable"]
    nonbio_count = object_counts["02 Non-Biodegradable"]
    cv2.putText(image, f"Biodegradable: {bio_count}", (10, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"Non-Biodegradable: {nonbio_count}", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the final window
    cv2.imshow("Smart Waste Sorter AI - Final Version", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
camera.release()
cv2.destroyAllWindows()