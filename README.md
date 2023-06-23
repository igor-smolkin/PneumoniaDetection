# Pneumonia detection using CNN

The idea for our network was to solve the medical problem of detecting pneumonia, which is a dangerous disease. The detection of this lung disease is based on an existing dataset which organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

## Used software and technologies
#### Transfer Learning

Transfer learning is a learning technique that allows data scientists to use what they've learned from a previous machine learning model that was used for a similar task. The features and weights can be used to train the new model, allowing it to be reused. When there is limited data, transfer learning works effectively for quickly training a model.

#### VGG16 Architecture 

In further work, the VGGNet architecture with 16 convolutional layers will be used. Its main advantage is the use of small filters (3x3) and deep structure, due to which it is able to extract complex features from images. Also, this architecture is quite easily reproduced and adapted for various computer vision tasks.

With the help of transfer learning, we will be able to adapt the pre-trained model to a specific task, which will allow reduce training time and achieve good results without having to train the model from scratch.

<img src ="https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png">

## Implementation stages

1. Loading the previously presented dataset from this <a href = "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"> link </a> or from my directory

2. Loading required libraries
```bash
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
```
3. Loading a pre-trained VGG16 model without the last layers. Let's set a fixed image size of 224x224 for the architecture, as well as image depth, which means working with RGB images.
```bash
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))
```

4. Freezing Pretrained Model Weights
```bash
for layer in base_model.layers:
    layer.trainable = False
```
5. In the next listing, we will announce the addition of new layers, such as:

base_model.output – represents the output tensor of the model VGG16, on the basis of which the sequence is built additional layers to create a model.

x = Flatten()(x) – transforms the output from the previous
layer x into a one-dimensional vector and assigns the result to a variable x, which is then used to add further layers to the model.

Dense – performs the operation of transforming input features 'x' into output features using a fully connected layer with 128 neurons and the ReLU activation function.

Dropout – regularization method that helps deal with model overfitting.

```bash
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)  # Добавляем слой Dropout
predictions = Dense(2, activation='softmax')(x)
```

6. Create our model

```bash
model = Model(inputs=base_model.input, outputs=predictions)
```

7. Let's compile our model using metrics such as:

Cross entropy is a metric for evaluating the discrepancy between two probability distributions;

The Adam (Adaptive Moment Estimation) optimizer is a stochastic gradient descent optimization algorithm that is used to update the neural network weights during training.

Accuracy is the ratio of the number of correct predictions to the total number of predictions.

```bash
model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
```

8. Creation of data generators with augmentation to extend the existing data set. Consider some metrics for creating additional features

− Rescale. The metric measures the percentage change in scale for each object in the image and calculates the average value of the absolute scaling error.

− Shear_range. Specifies the maximum angle of rotation images in degrees at which stretching occurs images along one dimension and compression along another measurements.

− Zoom_range. Specifies the maximum zoom that can be applied during augmentation.

− Horizontal_flip. Allows you to reflect the image horizontally, which can help the model learn more varied data.

```bash
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
horizontal_flip = True
)
test_datagen = ImageDataGenerator(rescale=1. / 255)
```

9. Loading data and augmenting the training set:

```bash
train_generator = train_datagen.flow_from_directory(
    'chest_xray/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

10. Loading test sample data:

```bash
test_generator = test_datagen.flow_from_directory(
    'chest_xray/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```


11. Start training our model

```bash
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=5,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)
```

12. In conclusion, we will display accuracy graphs and save the resulting model for further use.

```bash
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпоха')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('model.h5')
```