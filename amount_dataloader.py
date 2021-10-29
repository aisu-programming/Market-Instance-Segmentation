import cv2
import random
import numpy as np
import pandas as pd


def getDataset(data_dir, val_split=0.2, input_image_size=(640, 480), batch_size=4):
    
    csv = pd.read_csv(f"{data_dir}/annotation.csv").to_numpy()
    random.shuffle(csv)
    data_len = len(csv)

    val_image = list(range(data_len))
    random.shuffle(val_image)
    val_image = val_image[:int(data_len * val_split)]
    train_dataset, val_dataset = [], []

    train_img_batch, train_amount_batch = [], []
    val_img_batch  , val_amount_batch   = [], []
    for ri, row in enumerate(csv):

        filename, amount = row[0], row[1]
        img = cv2.imread(f"{data_dir}/images/{filename}")/255
        img = cv2.resize(img, input_image_size)

        if ri in val_image:
            val_img_batch.append(img)
            val_amount_batch.append(amount)
            if len(val_img_batch) == batch_size:
                val_dataset.append([ np.array(val_img_batch), np.array(val_amount_batch) ])
                val_img_batch, val_amount_batch = [], []
        else:
            train_img_batch.append(img)
            train_amount_batch.append(amount)
            if len(train_img_batch) == batch_size:
                train_dataset.append([ np.array(train_img_batch), np.array(train_amount_batch) ])
                train_img_batch, train_amount_batch = [], []
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    getDataset("dataset/3_192+204=396/396_spec_renamed_annotated")