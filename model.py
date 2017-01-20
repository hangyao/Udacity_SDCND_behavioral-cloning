import helper
import json
import os
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, model_from_json

IMSIZE = 80 # Input image size
SAMPLES_PER_EPOCH = 64#24576 # Number of samples per epoch
NB_EPOCH = 1 # Number of epoch
NB_VAL_SAMPLES = 64#2048 # Number of validation samples

try:
    # If model file exists in the folder, load the model and weights from files
    with open('model.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))
    model.summary()
    model.compile("adam", "mse")
    model.load_weights('model.h5', by_name=True)
    print("Model and weights are loaded.")

except:
    # If model file does not exist, construct a Xception model
    print("Construct a new Xception model.")
    base_model = Xception(input_shape=(IMSIZE, IMSIZE, 3),
                          weights='imagenet',
                          include_top=False)
    x = base_model.output
    # Add feature extraction layers for Xception model
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1)(x)
    model = Model(input=base_model.input, output=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.summary()
    model.compile("adam", "mse")

# Use Fit Generator for training and validation data epoches
train_gen = helper.batch_generator()
valid_gen = helper.batch_generator()
model.fit_generator(train_gen,
                    samples_per_epoch=SAMPLES_PER_EPOCH,
                    nb_epoch=NB_EPOCH,
                    validation_data=valid_gen,
                    nb_val_samples=NB_VAL_SAMPLES,
                    verbose=1)

print(model.ouput)
print(train_gen[1])

# Ask if overwrite the existing model files
if 'model.json' in os.listdir():
    print("Model exists in the folder. Overwrite (y) or Cancel (n)?")
    user_input = input()
    if user_input == "y":
        helper.save_model(model)
        print("Model and weights are overwritten.")
    else:
        print("Model and weights are cancelled.")
else:
    helper.save_model(model)
    print("Model and weights are saved.")
