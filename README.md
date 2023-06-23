# Pneumonia detection using CNN

The idea for our network was to solve the medical problem of detecting pneumonia, which is a dangerous disease. The detection of this lung disease is based on an existing dataset which organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

### Used software and technologies
##### Transfer Learning

Transfer learning is a learning technique that allows data scientists to use what they've learned from a previous machine learning model that was used for a similar task. The features and weights can be used to train the new model, allowing it to be reused. When there is limited data, transfer learning works effectively for quickly training a model.

##### VGG16 Architecture 

In further work, the VGGNet architecture with 16 convolutional layers will be used. Its main advantage is the use of small filters (3x3) and deep structure, due to which it is able to extract complex features from images. Also, this architecture is quite easily reproduced and adapted for various computer vision tasks.

With the help of transfer learning, we will be able to adapt the pre-trained model to a specific task, which will allow reduce training time and achieve good results without having to train the model from scratch.

<img src ="https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png">

### Implementation stages

1. Loading the previously presented dataset from this <a href = "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"> link </a>

