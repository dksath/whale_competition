# Whale & Dolphin Identification

## Introduction
Fingerprints and facial recognition are used to identify people in our world and we could use similar approaches with animals. Researchers today manually track marine life by the shape and markings on their tails, dorsal fins, heads and other body parts. Identification by natural markings via photographs, known as a Photo\_ID, is a powerful tool for marine mammal science. It allows individual animals to be tracked over time and enables assessments of population status and trends.

Currently, most research institutions rely on time-intensive and manual matching by the human eye, which can be inaccurate at times. Thousands of hours go into manual matching, which involves staring at photos to compare one individual to another, finding matches, and identifying new individuals.

In this Kaggle competition, our team will develop a model to match individual whales and dolphins by unique—but often subtle—characteristics of their natural markings. In this model we are to pay more attention to the dorsal fins and lateral body views in image sets from a multi-species dataset built by 28 research institutions. Our model will then suggest Photo\_ID solutions that are fast and accurate.

Whale and Dolphin Identification would be relevant to Computer Vision as the algorithm used in this model would fall under image classification. We are to use and experiment with various deep learning methods in order to achieve as accurate as possible results for this project.

## Instructions
1. Download required packages
```
pip install -r requirements.txt
```

2. Run `predict.py` to get prediction for image
```
python predict.py --image_path /sample1.jpg --model_path /Saved_models/model.h5
```
image_path: path to input image. 3 samples from the validation dataset have been provided.
model_path (optional): defaults to final model. 