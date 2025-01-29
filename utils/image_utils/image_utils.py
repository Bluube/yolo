import cv2
import os
import shutil
from tqdm import tqdm
from pathlib import Path
from roboflow import Roboflow


def resize_images(path_to_folder, path_to_new_folder, size=(640, 640)):
    x, y = size
    os.makedirs(path_to_new_folder, exist_ok=True)
    for file in tqdm(os.scandir(path_to_folder), total=len(os.listdir(path_to_folder))):
        img = cv2.imread(file.path)
        img_resized = cv2.resize(img, (x, y))
        cv2.imwrite(os.path.join(path_to_new_folder, Path(file.path).name), img_resized)


if __name__ == "__main__":
    pass