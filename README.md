# 🔍 Face Recognition Using OpenCV DNN and Cosine Similarity

This project demonstrates a lightweight face recognition pipeline using OpenCV's Deep Neural Network (DNN) module with a pre-trained Caffe model and cosine similarity for face matching.

## 🧠 How It Works

1. Loads a pre-trained face detection model (`res10_300x300_ssd_iter_140000.caffemodel`).
2. Detects faces in two images (`person1.jpg` and `unknown2.jpg`).
3. Extracts face embeddings by resizing and flattening the detected face.
4. Compares the embeddings using **cosine similarity**.
5. Classifies whether the two faces belong to the same person based on a similarity threshold.





