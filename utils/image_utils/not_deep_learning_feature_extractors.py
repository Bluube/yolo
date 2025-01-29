import cv2
import os

from tqdm import tqdm
from skimage.feature import hog
from pathlib import Path


class ImageFeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def get_hog_image(filepath):
        if isinstance(filepath, str):
            img = cv2.imread(filepath)
        else:
            img = filepath
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        features, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True)

        return hog_img

    def process_folder(self, folder):
        for file in tqdm(os.scandir(folder), total=len(os.listdir(folder))):
            hog = self.get_hog_image(file.path)
            cv2.imwrite(file.path, hog)


if __name__ == "__main__":
    # source = r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/data/OCR-BluePrints-13/test/images/im_jpg10_5_jpg.rf.37480e9cb5cde056063e4f9aa664ad3d.jpg'
    ife = ImageFeatureExtractor()
    for name in ['train', 'valid', 'test']:
        folder = rf'/root/171_data2/_share/Kirill/KirillOCR/YOLO/data/OCR-BluePrints-13_with_gen copy/{name}/images'
        ife.process_folder(folder)