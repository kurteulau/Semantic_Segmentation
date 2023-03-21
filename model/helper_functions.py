# from sklearn.utils import shuffle
# from pathlib import Path
# import matplotlib.pyplot as plt
# import pandas as pd
# import csv
# import numpy as np
# import rioxarray
# import xrspatial.multispectral as ms
# from PIL import Image


import shutil
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_path import path  # noqa
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
import torch
import rasterio
import pyproj
import rioxarray
import xrspatial.multispectral as ms
from my_preprocessing import train_val_test_split
from typing import Optional, List
import warnings
import csv


DATA_DIR = Path("/driven-data/cloud-cover")
TRAIN_FEATURES = DATA_DIR / "train_features"
TRAIN_LABELS = DATA_DIR / "train_labels"

assert TRAIN_FEATURES.exists()



def remove_chips(df, file):
    with open(file, 'r') as f:     # read csv file with bad chips
        bad_chips = csv.reader(f) 
        bad_chips_list = list(bad_chips)
    bad_chips_list = [item for sublist in bad_chips_list for item in sublist] # flatten the list
    df = df[~df['chip_id'].isin(bad_chips_list)] # filter out the chips using the flattened list
    return df
    
def train_dev_test_split(df, column='location', pct_train=0.6, pct_dev=0.2, pct_test=0.2, random_state=42):
    """
    Splits up a dataset using stratified random sampling to ensure equal proportion of observations by column
    in each of the train, development, and test sets.
    
    Params: 
     - df (DataFrame): pandas dataframe you want to split into train
     - col (str): column you want to stratify on. Default value is 'location'
     - pct_train (float): percent of dataset you want in the training set reprsented as float between 0 and 1. Default value is 0.6
     - pct_dev (float): percent of the dataset you want in the development set reprsented as float between 0 and 1. Default value is 0.2
     - pct_test (float): percent of the dataset you want in the test set reprsented as float between 0 and 1. Default value is 0.2
    Returns:
    - tuple with train, dev, and test datasets as pandas dataframes
    """
    

    train = pd.DataFrame()
    dev = pd.DataFrame()
    test = pd.DataFrame()

    for value in df[column].unique():
        # Create a dataframe for that column value and shuffle it
        col_df = df[df[column] == value]
        col_df_shuffled = shuffle(col_df, random_state=random_state)

        # Create splits for train, dev, and test sets
        split_1 = int(pct_train * col_df.shape[0])
        split_2 = int((pct_train + pct_dev) * col_df.shape[0])

        # Split up shuffled dataframe (for each col)
        col_df_train = col_df_shuffled.iloc[:split_1]
        col_df_dev = col_df_shuffled.iloc[split_1:split_2]
        col_df_test = col_df_shuffled.iloc[split_2:]

        # Add on the selections for train, dev, and test
        train = pd.concat(objs=[train, col_df_train])
        dev = pd.concat(objs=[dev, col_df_dev])
        test = pd.concat(objs=[test, col_df_test])

    return train, dev, test


# Exactly the same as above but uses validation/val instead of development/dev nomenclature
def train_val_test_split(df, column='location', pct_train=0.6, pct_val=0.2, pct_test=0.2, random_state=42):
    """
    Splits up a dataset using stratified random sampling to ensure equal proportion of observations by column
    in each of the train, development, and test sets.
    
    Params: 
     - df (DataFrame): pandas dataframe you want to split into train
     - col (str): column you want to stratify on. Default value is 'location'
     - pct_train (float): percent of dataset you want in the training set reprsented as float between 0 and 1. Default value is 0.6
     - pct_val (float): percent of the dataset you want in the validation set reprsented as float between 0 and 1. Default value is 0.2
     - pct_test (float): percent of the dataset you want in the test set reprsented as float between 0 and 1. Default value is 0.2
    Returns:
    - tuple with train, dev, and test datasets as pandas dataframes
    """
    

    train = pd.DataFrame()
    val = pd.DataFrame()
    test = pd.DataFrame()

    for value in df[column].unique():
        # Create a dataframe for that column value and shuffle it
        col_df = df[df[column] == value]
        col_df_shuffled = shuffle(col_df, random_state=random_state)

        # Create splits for train, val, and test sets
        split_1 = int(pct_train * col_df.shape[0])
        split_2 = int((pct_train + pct_val) * col_df.shape[0])

        # Split up shuffled dataframe (for each col)
        col_df_train = col_df_shuffled.iloc[:split_1]
        col_df_val = col_df_shuffled.iloc[split_1:split_2]
        col_df_test = col_df_shuffled.iloc[split_2:]

        # Add on the selections for train, val, and test
        train = pd.concat(objs=[train, col_df_train]).reset_index(drop=True)
        val = pd.concat(objs=[val, col_df_val]).reset_index(drop=True)
        test = pd.concat(objs=[test, col_df_test]).reset_index(drop=True)

    return train, val, test


def true_color_img(chip_id, data_dir=TRAIN_FEATURES):
    """Given the path to the directory of Sentinel-2 chip feature images,
    plots the true color image"""
    chip_dir = data_dir / chip_id
    red = rioxarray.open_rasterio(chip_dir / "B04.tif").squeeze()
    green = rioxarray.open_rasterio(chip_dir / "B03.tif").squeeze()
    blue = rioxarray.open_rasterio(chip_dir / "B02.tif").squeeze()

    return ms.true_color(r=red, g=green, b=blue)

# Copied directly from benchmark_tutorial
def display_random_chip(random_state):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    random_chip = train.sample(random_state=random_state).iloc[0]

    ax[0].imshow(true_color_img(random_chip.chip_id))
    ax[0].set_title(f"Chip {random_chip.chip_id}\n(Location: {random_chip.location})")
    label_im = Image.open(random_chip.label_path)
    ax[1].imshow(label_im)
    ax[1].set_title(f"Chip {random_chip.chip_id} label")

    plt.tight_layout()
    plt.show()
    
def display_true_color_and_label(chip_id, set_to_search, data_dir=TRAIN_FEATURES):
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    chip = set_to_search[set_to_search['chip_id'] == chip_id].iloc[0]

    ax[0].imshow(true_color_img(chip.chip_id))
    ax[0].set_title(f"Chip {chip.chip_id}\n(Location: {chip.location})")
    label_im = Image.open(chip.label_path)
    ax[1].imshow(label_im)
    ax[1].set_title(f"Chip {chip.chip_id} label")

    plt.tight_layout()
    plt.show()
    
def display_true_color_label_pixel_count(chip_id, set_to_search, data_dir=TRAIN_FEATURES):
    display_true_color_and_label(chip_id, set_to_search)
    chip = set_to_search[set_to_search['chip_id'] == chip_id].iloc[0]
    label_im = Image.open(chip.label_path)
    label_arr = np.array(label_im)
    num_cloud_pixels = label_arr.sum()
    proportion_cloud_pixels = round(label_arr.sum() / (512 * 512), 3)
    print(label_arr)
    print('Number of cloud pixels in label:', num_cloud_pixels)
    print('Proportion of cloud pixels in label:', proportion_cloud_pixels)
    
def count_cloud_pixels(train_x, BANDS, train_y):
    """
    Returns dataframe with training dataset chip_id's and the number of cloud pixels in the labels for those
    training set chips. Can be altered to check labels for other sets like validation as well but haven't
    had the time yet.
    """
    try:
        from cloud_dataset import CloudDataset
    except ImportError:
        from benchmark_src.cloud_dataset import CloudDataset

    train_dataset = CloudDataset(x_paths=train_x, bands=BANDS, y_paths=train_y)
    
    batch_size=200
    num_workers=2
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
    
    chip_ids = np.zeros(train_y.shape[0], dtype=object)
    num_cloud_pixels = np.zeros(train_y.shape[0], dtype=int)
    
    i = 0 
    for item in train_dataloader:
        item_chip_ids = item['chip_id']
        item_labels = item['label']
    
        for j, chip_id in enumerate(item_chip_ids):
            chip_ids[i] = chip_id
            num_cloud_pixels[i] = item_labels[j].sum().item()
            i += 1
    
    df = pd.DataFrame({'chip_id': chip_ids, 'num_cloud_pixels_in_label': num_cloud_pixels})
    
    return df