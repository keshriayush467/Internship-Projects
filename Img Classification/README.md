Pneumonia Detection from Chest X-rays using ResNet-50

This project uses deep learning to automatically detect Pneumonia from chest X-ray images. The model is trained using PyTorch and a ResNet-50 architecture.



 Project Overview

The goal of this project is to develop an AI system that can classify chest X-rays as Normal or Pneumonia. This can help radiologists in early detection and reduce manual effort in screening large numbers of X-ray images.



 Project Structure

 'pneumonia_model.pth': Trained model weights for ResNet-50.  
'predict.py': Script to predict a single chest X-ray image.  
'test_image.jpeg': Example X-ray image used for testing.  
 'README.md': This file with project details.


 Key Features

 Detects whether a chest X-ray is Normal or shows signs of Pneumonia.  
 Uses ResNet-50 pretrained model fine-tuned on chest X-ray data.  
 Handles imbalanced datasets using class weighting.  
 Supports single image prediction for quick evaluation.  
 Can be extended for batch testing, web apps, or clinical deployment.



  Dataset

The model is trained on the ChestX-ray14 dataset, which contains thousands of X-ray images labeled as Normal or Pneumonia.  

 Normal images: -1,300  
 Pneumonia images: -3,700  

 Link- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
