from picamera2 import Picamera2
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import os

# Build model path relative to this script
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "mnist_cnn.tflite")

# Load TFLite model
interpreter = Interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("Model loaded.")
print("Input details:", input_details)
print("Output details:", output_details)


def preprocess_digit(bgr_img):
    """
    From a BGR image, find the largest blob (digit), crop, threshold,
    resize to 28x28, normalize to [0,1], shape (1,28,28,1).
    Returns (input_tensor, (x, y, w, h)) or (None, None) if no digit.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    # Largest contour = our digit
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Skip tiny blobs (noise)
    if w * h < 50:
        return None, None

    roi = th[y:y + h, x:x + w]

    # Make square by padding
    side = max(w, h)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    roi_sq = cv2.copyMakeBorder(
        roi,
        pad_y, side - h - pad_y,
        pad_x, side - w - pad_x,
        cv2.BORDER_CONSTANT, value=0
    )

    # Resize to 28x28
    digit = cv2.resize(roi_sq, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to [0,1] and add batch+channel dims
    digit = digit.astype("float32") / 255.0
    digit = digit.reshape(1, 28, 28, 1)

    return digit, (x, y, w, h)


def predict_digit(preprocessed_digit):
    """
    Run inference on preprocessed digit image using TFLite.
    Returns (pred_class, confidence, inference_time_seconds).
    """
    interpreter.set_tensor(input_index, preprocessed_digit)
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()

    output = interpreter.get_tensor(output_index)[0]  # shape (10,)
    pred_class = int(np.argmax(output))
    confidence = float(output[pred_class])

    return pred_class, confidence, (t1 - t0)


def main():
    # Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # let camera warm up

    print("Camera started.")
    print("Press 'c' to capture & classify, 'q' to quit.")

    while True:
        # Capture RGB frame from camera
        frame_rgb = picam2.capture_array()
        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        display = frame_bgr.copy()

        cv2.imshow("Live - 'c' capture, 'q' quit", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            digit_input, box = preprocess_digit(frame_bgr)
            if digit_input is None:
                print("No clear digit detected, try again.")
                continue

            pred, conf, dt = predict_digit(digit_input)
            x, y, w, h = box

            # Draw bounding box & label on the display image
            cv2.rectangle(display, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            label = f"{pred} ({conf:.2f}), {dt*1000:.1f} ms"
            cv2.putText(display, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            cv2.imshow("Prediction", display)
            cv2.imwrite("last_prediction.jpg", display)
            print(f"Predicted: {pred}, conf={conf:.3f}, "
                  f"time={dt*1000:.1f} ms")

        elif key == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
