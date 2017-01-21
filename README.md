# Behavioral Cloning

## Udacity Self-Driving Car Nanodegree Project 3

In this project, a Deep Neural Network will be trained and validated to clone a human driving behavior in order to control a self-driving car to steer in the Simulator. The only data we will use for training are steering data from human input and image data from front-facing cameras.

## Install

Download the simulator based on the Unity engine that uses real game physics to create a close approximation to real driving.
- Download it here: [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)

- Beta Simulators: [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/587525b2_udacity-sdc-udacity-self-driving-car-simulator-dominique-default-mac-desktop-universal-5/udacity-sdc-udacity-self-driving-car-simulator-dominique-default-mac-desktop-universal-5.zip)

This project requires **Python 3.5** with the following libraries/dependencies installed:

- [Numpy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](http://opencv.org/)
- [Eventlet](http://eventlet.net/)
- [SocketIO](https://pypi.python.org/pypi/python-socketio)
- [PIL](https://pillow.readthedocs.io/)
- [Flask](http://flask.pocoo.org/)

## Code

- `model.py` - The script used to create and train the model.
- `helper.py` - The script contained all helper functions.
- `drive.py` - The script to drive the car.
- `model.json` - The model architecture.
- `model.h5` - The model weights.
- `README.md` - explains the structure of the network and training approach.

## Run

Run the server
```
python drive.py model.json
```
Once the model is up and running in `drive.py`, you should see the car move around the track!

## Data

1. `IMG` folder -  this folder contains all the frames of your driving.
2. `driving_log.csv` - each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of your car. You'll mainly be using the steering angle.
3. `model.json` - model architecture.
4. `model.h5` - model weights.

# Data Augmentation

The following steps are implemented to randomly select and transform images. Here is the image augmentation pipeline:

| Raw Image |
| :-------: |
| Flip |
| Shear |
| Rotate |
| Adjust Gamma |
| Crop & Resize |
| RGB to YUV |
| Input Image to CNN |

1. Randomly selected images:

  64 images are randomly selected for each batch. The images are also randomly select from center, left, or right cameras.

  Here is a sample of raw image:

  ![Raw image](documents/raw.png)

2. Random Flip:

  Images are randomly flipped.

  Here is a sample of flipped image:

  ![Raw image](documents/flip.png)

3. Random Shear:

  Images are randomly sheared horizontally up to 50 pixels from the center point of the image.

  Here is a sample of flipped image:

  ![Raw image](documents/shear.png)

4. Random Rotation:

  Images are randomly rotated up to 5 degrees around the center of the image.

  Here is a sample of rotated image:

  ![Raw image](documents/rotate.png)

5. Random Gamma Adjustment:

  Images are randomly adjusted by gamma value up to ± 0.7.

  Here is a sample of gamma adjusted image:

  ![Raw image](documents/gamma.png)

6. Crop and Resize:

  Images are cropped 56 pixels from top and 16 pixels from bottom, and then resized to 80x80 pixels.

  Here is a sample of cropped and resized image:

  ![Raw image](documents/crop_resize.png)

7. Convert from RGB to YUV:

  Images are converted from RGB channel to YUV channel [1].

  ![Raw image](documents/rgb2yuv.png)

# Architecture

The architecture of model is strictly followed the model in this paper [1]. The only differences are:

1. the input size has been reduced from 66x200 to 80x80. The input size of images has been reduced more than a half, and I think that won't affect the results of training.

2. Two Dropout layers have been added between the Flatten layer, the first Fully-Connected layer, and the second Fully-Connected layer. In order to prevent overfitting and to yield a more generalized model which can adapt to new tracks, I added two Dropout layers there for regularization.

Here is the architecture diagram from the paper [1]:

![CNN Architecture](documents/CNN_architecture.png)

And here is the summary of model actually been implemented:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 80, 80, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 38, 38, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 17, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 7, 48)      43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 5, 64)      27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 3, 64)      36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 576)           0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 576)           0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          671628      dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================
Total params: 925,047
Trainable params: 925,047
Non-trainable params: 0
____________________________________________________________________________________________________
```

The Adam optimizer and MSE loss function are used for compilation.

A Fit Generator is used to fit the model on image data generated batch-by-batch. This will increase efficiency of model training.

# Training

The training process involves the following steps:

1. Fine tune steering angle adjustments for each step of image augmentation in order to make the car steer smoothly on tracks. These steering angle adjustments are empirically derived from triangular geometry, and tweaked by trial and error through the whole training process. These adjustments include:
  - Adjustment for images from the left and right cameras;
    We can assume that the width of car is 1.8m, camera sight distance is 20m, and `tan(\alpha) ~ \alpha` when `\alpha \to 0`, then the steering angle adjustment is `1.8 / 2 / 20 = 0.045`.
  - Adjustment for randomly sheared images;
    We can assume that the height of camera is 1.3m, the steering ratio is 15, then the steering angle adjustment is `S / (H / 2) / 2 * 1.3 * 15 / 20`, where `S` is sheared pixels, and `H` is the image height in pixels.
  - Adjustment for randomly rotated images.
    Based on previous assumptions, the steering angle adjustment is `\theta * \pi / 180 / 2 * 1.3 * 15 / 20`, where `\theta` is the rotation angle in degree.

2. The model was trained in the cloud on a Microsoft Azure Virtual Machine powered with 2x NVIDIA Tesla K80 GPUs. Each epoch took only 100s for training and validation. Because such computational capability is available, and regularization such as Dropout was implemented to reduce overfitting, I can brutally use a relatively large number of epochs without worrying about overfitting.

3. A callback of Early Stopping is implemented to stop training when validation loss has stopped improving. It can save processing time and will promise optimal model for the following analysis. Eventually the training stopped after 16 epochs.

# Results

The result performs surprisingly good, and the car can run on both training and testing tracks. On the training track, the car steers a little bit roughly, but on the testing track, it runs pretty smoothly.

![Train track](documents/train_track.gif)

![Test track](documents/test_track.gif)

## Reference

1. [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

## License

The contents of this repository are covered under the [MIT License](LICENSE).
