import cv2
import numpy as np
import os
import sys

# Load the DNN face detector
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_FILE = "deploy.prototxt"

if not os.path.exists(MODEL_FILE) or not os.path.exists(CONFIG_FILE):
    print("❌ Model files not found. Make sure 'res10_300x300_ssd_iter_140000.caffemodel' and 'deploy.prototxt' are in the working directory.")
    sys.exit()

net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)

def detect_face(img):
    """Detects and returns the cropped face region from an image using OpenCV DNN."""
    if img is None:
        return None
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            return img[y1:y2, x1:x2]
    return None

def get_embedding(face_img):
    """Returns a normalized embedding vector from a cropped face image."""
    resized = cv2.resize(face_img, (100, 100)).flatten()
    return resized / 255.0

def cosine_similarity(a, b):
    """Computes cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_and_process(image_path):
    """Reads image, detects face, and returns embedding."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    face = detect_face(img)
    if face is None:
        print(f"❌ No face found in image: {image_path}")
        return None
    return get_embedding(face)

def main():
    known_embed = load_and_process("person1.jpg")
    unknown_embed = load_and_process("unknown2.jpg")

    if known_embed is None or unknown_embed is None:
        sys.exit()

    similarity = cosine_similarity(known_embed, unknown_embed)
    print(f"\n🔍 Similarity Score: {similarity:.2f}")

    if similarity > 0.85:
        print("✅ Face match!")
    else:
        print("❌ Different person.")

if __name__ == "__main__":
    main()

