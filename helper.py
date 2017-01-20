import numpy as np
import pandas as pd
import cv2
from PIL import Image
import json


CSV_FILE = 'driving_log.csv'
TOP_CROP = 56
BOT_CROP = 16
STEERING_ADJ = 0.25
IMSIZE = 80


def load_img(image_name):
    '''
    Load and process image. image_name is a str of path and image name.
    Return cropped, resized, and normalized image of an np.array.
    '''
    im = Image.open(image_name)
    im = np.asarray(im)
    return im


def crop_resize_img(image):
    '''
    Crop and resize image.
    '''
    image = image.crop(box=(0, TOP_CROP, 320, 160-BOT_CROP))
    image = image.resize((IMSIZE, IMSIZE))
    return image


def get_batch_data(batch_size=64):
    '''
    Get batch data from data file.
    Return image file names and steering number.
    '''
    data = pd.read_csv(CSV_FILE, sep=',')
    data_size = len(data)
    random_idx = np.random.randint(0, data_size, batch_size)
    batch_data = []
    for index in random_idx:
        img_selector = np.random.randint(0, 3)
        if img_selector == 0:
            img_file = data.iloc[index, 0].strip()
            steer = data.iloc[index, 3]
            batch_data.append((img_file, steer))
        elif img_selector == 1:
            img_file = data.iloc[index, 1].strip()
            steer = data.iloc[index, 3] + STEERING_ADJ
            batch_data.append((img_file, steer))
        else:
            img_file = data.iloc[index, 2].strip()
            steer = data.iloc[index, 3] - STEERING_ADJ
            batch_data.append((img_file, steer))
    return batch_data


def batch_generator(batch_size=64):
    """
    Generate the training batch.
    """
    while True:
        X_batch = []
        y_batch = []
        data = get_batch_data(batch_size)
        for img_file, steer in data:
            im = load_img(img_file)
            im, steer = random_transform_img(im, steer)
            im = Image.fromarray(im)
            im = crop_resize_img(im)
            im = np.asarray(im)
            im = im / 255.0 - 0.5
            X_batch.append(im)
            y_batch.append(steer)
        yield (np.array(X_batch), np.array(y_batch))


def random_transform_img(image, steer):
    '''
    Randomly transform image in format of np.array.
    Return randomly flipped, sheared, rotated, gamma-adjusted image.
    '''
    image, steer = random_flip(image, steer)
    image, steer = random_shear(image, steer)
    image, steer = random_rotate(image, steer)
    image = random_gamma(image)
    return image, steer


def random_flip(image, steer):
    '''
    Randomly flip image.
    '''
    if np.random.binomial(1, .5, 1)[0]:
        image = np.fliplr(image)
        steer = - steer
    return image, steer


def random_shear(image, steer, shear_dist=50):
    '''
    Randomly shear image.
    '''
    rows, cols, _ = image.shape
    d = np.random.randint(-shear_dist, shear_dist+1)
    pt_1 = np.float32([[0, rows], [cols, rows], [cols/2, rows/2]])
    pt_2 = np.float32([[0, rows], [cols, rows], [cols/2+d, rows/2]])
    dsteer = d / (rows/2) * 360 / (2. * np.pi * 25) / 6.
    M = cv2.getAffineTransform(pt_1, pt_2)
    image = cv2.warpAffine(image, M, (cols,rows), borderMode=1)
    steer += dsteer
    return image, steer


def random_rotate(image, steer, angle_range=5):
    '''
    Randomly rotate image.
    '''
    angle = np.random.uniform(-angle_range, angle_range)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    image = cv2.warpAffine(image, M, (cols,rows), borderMode=1)
    dsteer = - (np.pi / 180) * angle
    steer += dsteer
    return image, steer


def random_gamma(image, gamma_range=0.7):
    '''
    Random adjust image gamma.
    '''
    gamma = np.random.uniform(1.-gamma_range, 1.+gamma_range)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def save_model(model):
    '''
    Save model files.
    '''
    json_string = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(json_string, outfile)
    model.save_weights('model.h5')
