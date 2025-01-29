import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

from tqdm import tqdm
from skimage.morphology import skeletonize, thin
from skimage.util import invert
from PIL import Image
from pathlib import Path


def process_folder_with_images(folder_with_imgs, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for file in tqdm(os.scandir(folder_with_imgs), total=len(os.listdir(folder_with_imgs))):
        collage = make_collage_with_skeletonized_img(file.path)
        save_path = os.path.join(save_folder, Path(file.path).name)
        cv2.imwrite(save_path, collage)


def process_folder_and_save_skel_images_to_another_folder(folder_with_imgs, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for file in tqdm(os.scandir(folder_with_imgs), total=len(os.listdir(folder_with_imgs))):
        skel = get_skeletonized_image(file.path)[0]
        save_path = os.path.join(save_folder, Path(file.path).name)
        cv2.imwrite(save_path, skel)


def get_skeletonized_image(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    skel = thin(thresh).astype(np.uint8)*255
    return cv2.bitwise_not(skel), cv2.bitwise_not(thresh), gray


def make_collage_with_skeletonized_img(img, show=False):
    skel, thresh, gray = get_skeletonized_image(img)
    collage = np.concatenate([skel, gray])
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(8, 4))
        ax = axes.ravel()
        ax[0].imshow(gray, cmap='gray')
        ax[0].set_title("Original")
        ax[1].imshow(thresh, cmap='gray')
        ax[1].set_title("Thresh")
        ax[2].imshow(skel, cmap='gray')
        ax[2].set_title("Skeleton")
        fig.tight_layout()
        plt.show()

    return collage


if __name__ == "__main__":
    img = r'C:\Users\79777\Desktop\WORK\OCR\DrawingText\DrawingText\newText\gen0_4.jpg'
    folder = r'C:\Users\79777\Desktop\WORK\OCR\DrawingText\DrawingText\newText'
    save_folder = r'C:\Users\79777\Desktop\WORK\OCR\DrawingText\DrawingText\thinned'
    # process_folder_with_images(folder, save_folder)
    # res = make_collage_with_skeletonized_img(img, show=True)