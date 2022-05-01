# These library are for data manipulation 
import numpy as np
import pandas as pd

# These library are for working with directories
import os
from glob import glob
from tqdm import tqdm

# These library are for Visualization
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import argparse
# These library are for the Dataset
from sklearn.model_selection import train_test_split
from datetime import datetime
# These Library are for converting Label Encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# These library are for loading Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow import keras
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y = onehot_encoded
    return y, label_encoder

def main():

    parser = argparse.ArgumentParser(description="Cleaning preprocessed data.")
    parser.add_argument(
        "--image_path",
        type=str,
        help="The path to the image file.",
    )
    parser.add_argument(
        "--model_path",
        default="Saved_models/model.h5",
        type=str,
        help="The path to the model chosen. Default: `Saved_models/model.h5`.",
    )
    opt = parser.parse_args()
    START_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path=os.getcwd()     

    X_train = np.zeros((1, 32, 32, 3))
    img = image.load_img(opt.image_path,target_size=(32, 32, 3))
    X = image.img_to_array(img)
    X = preprocess_input(X)
    X_train[0] = X
    X_train /= 255
    model = keras.models.load_model(opt.model_path)

    
    pridictions=model.predict(np.array(X_train), verbose=1)


    path_traindata = path+'/csv/train.csv'
    train_df = pd.read_csv(path_traindata)
    train_df = train_df.drop_duplicates(subset=['individual_id'],keep='last')
    y, label_encoder = prepare_labels(train_df['individual_id'])
        
    for i, pred in enumerate(pridictions):
        p = pred.argsort()[-5:][::-1]
        for x in p:
            s = label_encoder.inverse_transform(p)[0]
        prediction_got = s

    y_result=train_df.loc[(train_df["individual_id"]==prediction_got)]

    print("The species that the model has pridicted is: ",y_result["species"].to_string(index=False))

    



if __name__ == '__main__':
    main()