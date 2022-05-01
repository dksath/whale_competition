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


def main():

    parser = argparse.ArgumentParser(description="Cleaning preprocessed data.")
    parser.add_argument(
        "--image_path",
        type=str,
        help="The path to the Image file.",
    )
    parser.add_argument(
        "--model_path",
        default="Saved_models/model.h5",
        type=str,
        help="The path to the model chosen. Default: `Saved_models/model.h5`.",
    )

    
    START_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    opt = parser.parse_args()


    img = image.load(opt.image_path,target_size=(32, 32, 3))
    X = image.img_to_array(img)
    X = preprocess_input(X)
    model = keras.models.load_model(opt.model_path)
    model.summary()
    
    pridictions=model.predict(np.array(X), verbose=1)
    print(pridictions)
    