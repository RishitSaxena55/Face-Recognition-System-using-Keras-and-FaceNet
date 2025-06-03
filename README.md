# 🔍 Face Recognition System using Keras FaceNet

A robust deep learning-based face recognition system built using **Keras**, **FaceNet**, and **OpenCV**. This system detects faces in images and identifies individuals by comparing their facial embeddings to a known database.

---


## 📦 Dataset

This project uses the **Labeled Faces in the Wild (LFW)** dataset:

🔗 [LFW People Dataset on Kaggle](https://www.kaggle.com/datasets/atulanandjha/lfwpeople)

To use:
1. Download and extract the dataset in your working directory.
2. Use the `lfw_funneled/` folder as the database of known identities.

---


## 🚀 Features

- Face detection using Haar cascades
- Face embedding extraction using pre-trained Keras FaceNet model
- Face recognition by comparing embeddings with a database
- Automatic annotation with names and bounding boxes
- Supports adding new people easily

---

## 📂 Project Structure

- face-recognition/
- ├── face_recognition_system.py
- ├── notebook.ipynb
- ├── database/ # Folder of known persons (1 subfolder per person)
- │ └── Person_Name/
- │ ├── image1.jpg
- │ └── image2.jpg
- ├── test_images/ # Images for recognition/testing
- │ └── test.jpg
- └── README.md

---

## ⚙️ How It Works

1. **Load Pretrained FaceNet**: A pre-trained Keras model is used to extract 128-D embeddings from input faces.
2. **Build Database**: Traverse a folder of known people, extract face embeddings, and store them with names.
3. **Recognition**: For a given input image:
   - Detect all faces
   - Extract embedding for each face
   - Compare with database using L2 distance
   - Label face with the closest matching identity

## 🧪 Example Usage

```python
# Step 1: Build the database
database = build_face_database("/path/to/lfw_funneled", FRmodel)

# Step 2: Annotate and display a test image
output = annotate_faces_with_boxes_and_labels("/path/to/image.jpg", FRmodel, database)

# Step 3: Show image
plt.imshow(output)
plt.axis('off')
plt.show()

---

## 🛠️ Installation

Install all required packages using pip:

```bash
pip install numpy tensorflow opencv-python matplotlib

## 📚 Acknowledgements
- FaceNet: A Unified Embedding for Face Recognition and Clustering

- Pre-trained Keras FaceNet model

- Labeled Faces in the Wild (LFW) Dataset
