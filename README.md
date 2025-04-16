# DIGIT DETECTION


In this homework, we implement the deep learning method to detect the digits images.First, we use detection package to construct Faster R-CNN model [3]. After the hyper-parameter tuning, we obtain a testing mAP of 0.389141.


##  Introduction of the Digit Recognition Dataset
The Digit Recognition Dataset is used for training and evaluating object detection models that identify digits in real-world RGB images. This dataset includes a total of 46,470 images, divided into 30,062 training images, 3,340 validation images, and 13,068 test images. The task involves detecting digits within each image using both classification and localization.

Task Description
Task 1: Detect and classify each digit in the image and output its bounding box. For example, identify digits “4” and “9” and their respective bounding boxes.

Task 2: Predict the number of digits detected in each image. For example, the image containing “49” should return the number 2.

Constraints
External data sources are not allowed; only the provided dataset may be used.

The detection model must be based on Faster R-CNN, which includes a backbone, Region Proposal Network (RPN), and a detection head. Modifications to these components are permitted and should be clearly explained in the report.

The use of pretrained weights is allowed but not mandatory.

The model should be optimized for both accuracy and inference speed.

## File Structure
faster_rcnn_digit_recognition/
│
├── config/                        # Configuration files
│   └── config.yaml               # Hyperparameters, paths, training settings
│
├── data/                          # Dataset-related code
│   ├── dataset.py                # Custom dataset class
│   └── transforms.py             # Data augmentations and preprocessing
│
├── models/                        # Model-related code
│   ├── faster_rcnn.py            # Faster R-CNN model builder
│   └── backbone.py               # Backbone network modifications (e.g., ResNet50)
│
├── utils/                         # Utility functions
│   ├── engine.py                 # Training and evaluation loop
│   ├── utils.py                  # Helper functions
│   └── visualization.py         # Tools for visualizing predictions
│
├── outputs/                       # Directory for saved results
│   ├── checkpoints/             # Model checkpoints
│   └── logs/                    # Training logs
│
├── test/                          # Testing and inference scripts
│   └── test.py                   # Run model on test set
│
├── train.py                       # Main training script
├── inference.py                   # Inference script
├── evaluate.py                    # Evaluation metrics script
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview

## Requirements
torch==1.6.0
torchvision==0.7.0          # Compatible with torch 1.6
numpy==1.17.0
opencv-python>=4.1.0
matplotlib
pandas
scipy
pycocotools
tqdm
pillow
tensorboard>=1.14
h5py                       

# Project Title: **Digit Detection with Faster R-CNN & YOLOv4**

This project focuses on object detection using **Faster R-CNN** and **YOLOv4** for digit detection from images. The model has been trained and evaluated on a custom dataset, achieving competitive performance across different metrics. This README outlines how to set up the environment, train the models, and run inference for both Faster R-CNN and YOLOv4.

## 🔧 Setup and Installation

### 1. Environment Setup
To set up the project environment, run the following commands in a **Google Colab** notebook or your local setup:

```bash
# Clone Torchvision to get access to the pretrained Faster R-CNN model
!git clone https://github.com/pytorch/vision.git
%cd vision
!pip install -e .
%cd ..

# Install necessary dependencies
!pip install pycocotools

## 2. Dataset Upload
You can upload your train, validation, and test images as well as COCO-format JSON annotations using Google Drive. Run the following command to mount your Google Drive:

from google.colab import drive
drive.mount('/content/drive')

## 3. Model Setup: Faster R-CNN
Use Faster R-CNN with a ResNet50 backbone to detect digits. Here's how to define the model:

import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

## 4. Custom Dataset (COCO Format)
For your dataset, create a custom DigitDataset class:

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class DigitDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for obj in annotations:
            xmin, ymin, w, h = obj['bbox']
            boxes.append([xmin, ymin, xmin+w, ymin+h])
            labels.append(obj['category_id'])

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.ids)

## 5. Training Loop
Train the Faster R-CNN model on your dataset:

from engine import train_one_epoch, evaluate
import utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_faster_rcnn_model(num_classes=11)  # Adjust number of classes accordingly
model.to(device)

# Create datasets and dataloaders
transform = T.Compose([T.ToTensor()])
train_dataset = DigitDataset('/path/to/train/images', '/path/to/train.json', transform)
val_dataset = DigitDataset('/path/to/val/images', '/path/to/val.json', transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=utils.collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=utils.collate_fn)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Train
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)
    evaluate(model, val_loader, device=device)

## 6. Inference
Run inference to get predictions on the test set:

import json

def make_predictions(model, test_loader, device):
    model.eval()
    predictions = []
    for images, targets in test_loader:
        images = list(img.to(device) for img in images)
        with torch.no_grad():
            outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"])
            for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
                box = box.tolist()
                w, h = box[2] - box[0], box[3] - box[1]
                predictions.append({
                    "image_id": image_id,
                    "bbox": [box[0], box[1], w, h],
                    "score": float(score),
                    "category_id": int(label)
                })

    with open("pred.json", "w") as f:
        json.dump(predictions, f)

## 7. Task 2: Full Digit Prediction

For each image, sort digits left to right by x_min and concatenate the digits into a sequence. The output is saved as a CSV file like:
image_id,pred_label
12345,49
12346,731
...

## 🚀 Results
Model Performance Comparison

Model	Faster-RCNN	YOLOv4
Test mAP@0.5:0.95	0.389141	0.41987
Speed on P100 GPU (img/s)	0.2	0.07364
Speed on K80 GPU (img/s)	X	0.13696

## 💡 References
Faster-RCNN
Digit Detector - GitHub

Torchvision Detection Examples

PyTorch Tutorial

YOLOv4
YOLOv4 PyTorch Implementation

