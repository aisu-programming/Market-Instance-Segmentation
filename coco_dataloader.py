""" Libraries """
import cv2
import random
import numpy as np
import tensorflow as tf
import skimage.io as io
from pycocotools.coco import COCO
from tensorflow.keras.preprocessing.image import ImageDataGenerator


""" Functions """
def filterDataset(data_dir, classes=None, mode="train"):    
    # initialize COCO api for instance annotations
    annFile = f"{data_dir}/{mode}.json"
    coco = COCO(annFile)
    
    images = []
    if classes!=None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
    
    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])
            
    random.shuffle(unique_images)
    dataset_size = len(unique_images)
    return unique_images, dataset_size, coco


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"]==classID:
            return cats[i]["name"]
    return None


def getImage(imageObj, img_folder, input_image_size):
    iis = input_image_size
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj["file_name"]) / 255.0
    # Resize
    train_img = cv2.resize(train_img, (iis[1], iis[0]))
    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img


def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
    iis = input_image_size
    annIds = coco.getAnnIds(imageObj["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(iis)
    for a in range(len(anns)):
        className = getClassName(anns[a]["category_id"], cats)
        pixel_value = classes.index(className)+1
        new_mask = cv2.resize(coco.annToMask(anns[a])*pixel_value, (iis[1], iis[0]))
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(iis[0], iis[1], 1)
    return train_mask


def getBinaryMask(imageObj, coco, catIds, input_image_size):
    iis = input_image_size
    annIds = coco.getAnnIds(imageObj["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(iis)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), (iis[1], iis[0]))
        
        # Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(iis[0], iis[1], 1)
    return train_mask


def dataGeneratorCoco(images, classes, coco, data_dir, input_image_size, batch_size=4, mask_type="normal"):
    iis = input_image_size
    bs  = batch_size

    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)

    # for _ in range(2):
    c = 0
    random.shuffle(images)
    while(c+bs <= dataset_size):
        imgs        = np.zeros((bs, iis[0], iis[1], 3)).astype("float")
        img_statuses = np.zeros((bs, 1)).astype("float")
        img_masks   = np.zeros((bs, iis[0], iis[1], 1)).astype("float")

        for i in range(c, c+bs): # initially from 0 to batch_size, when c = 0
            imageObj = images[i]
            
            ### Retrieve Image ###
            img = getImage(imageObj, data_dir, iis)
            img_status = imageObj["status"]
            
            ### Create Mask ###
            if mask_type=="normal":
                img_mask = getNormalMask(imageObj, classes, coco, catIds, iis)
            elif mask_type=="binary":
                img_mask = getBinaryMask(imageObj, coco, catIds, iis)
            
            # Add to respective batch sized arrays
            imgs[i-c]         = img
            img_statuses[i-c] = img_status
            img_masks[i-c]    = img_mask

        c += bs
        yield imgs, img_statuses, img_masks


def getDataset(data_dir, classes, input_image_size=(480, 640), batch_size=4, mask_type="normal"):

    dataset = {}

    for mode in ["train", "validation"]:
        images, _, coco = filterDataset(data_dir, classes, mode)
        train_gen       = dataGeneratorCoco(images, classes, coco, data_dir, input_image_size, batch_size, mask_type)
        imgs, status, masks = [], [], []
        for batch_img, batch_status, batch_mask in train_gen:
            imgs.append(batch_img)
            status.append(batch_status)
            masks.append(batch_mask)
        dataset[mode] = tf.data.Dataset.from_tensor_slices((imgs, status, masks))

    return dataset["train"], dataset["validation"]