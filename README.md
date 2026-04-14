# Machine Learning Projects

This repository contains two machine learning projects completed as part of my learning and practice:

1. Fake News Detection using NLP
2. Chest X-Ray Image Classification using ResNet

---

## Project 1: Fake News Detection using NLP

Objective:
Build a model to classify news articles as Fake or Real using Natural Language Processing.

Dataset:
Fake and Real News Dataset from Kaggle: [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset]

Folder Structure:
- `notebook/fake_news_detection.ipynb` : Jupyter notebook with complete code
- `dataset/` : Local folder to store Fake.csv and True.csv (download from Kaggle)

Key Features:
- Combines news title and text for better accuracy
- Uses text preprocessing (lowercase, remove punctuation, remove numbers)
- TF-IDF vectorization
- Classification using SVM (Support Vector Machine)
- Accuracy: ~96–97%

How to Run:
1. Download dataset from Kaggle into `dataset/` folder.
2. Open notebook `notebook/fake_news_detection.ipynb` in Jupyter.
3. Run all cells.

---

 **Project 2: Chest X-Ray Image Classification using ResNet**

Objective:
Classify chest X-Ray images into normal or pneumonia using deep learning with ResNet.

Dataset:
Publicly available Chest X-Ray dataset (e.g., from Kaggle or other open sources).  
Include the dataset in a local folder `dataset_chest_xray/` (not uploaded due to size limits).



Key Features:
- Uses ResNet (pretrained model) from PyTorch/TensorFlow
- Image preprocessing and augmentation
- Trains model on labeled images
- Evaluates performance with accuracy, confusion matrix

How to Run:
1. Download chest X-Ray dataset and place in 'dataset_chest_xray/'.
2. Open notebook 'notebook/chest_xray_resnet.ipynb'.
3. Run all cells.



Install dependencies using:

