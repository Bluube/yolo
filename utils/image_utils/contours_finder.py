import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from imutils import contours


def image_preprocessing(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (5, 5), 0)
    # thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    edged = cv2.Canny(blured, 30, 150)
    return img, edged


def process_folder_with_images(folder_with_imgs, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for file in tqdm(os.scandir(folder_with_imgs)):
        image_with_contours = select_contours(file.path)
        save_path = os.path.join(save_folder, Path(file.path).name)
        cv2.imwrite(save_path, image_with_contours)


def select_contours(img):
    initial, preprocessed = image_preprocessing(img)
    cnts, hierarchy = cv2.findContours(preprocessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # (cnts_processed, _) = contours.sort_contours(cnts, method="left-to-right")
    # print(contours[1])
    bboxes_xywh = [cv2.boundingRect(cnts[i]) for i in range(len(cnts))
                   if i != 0 and cv2.contourArea(cnts[i]) > 15]
    bboxes_xyxy = [(x, y, x+w, y+h) for x, y, w, h in bboxes_xywh]
    for bbox in bboxes_xyxy:
        cv2.rectangle(initial, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(initial, f'{len(bboxes_xyxy)}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    # print(bboxes_xyxy)
    # print(bboxes)

    # cv2.drawContours(initial, contours, -1, (0, 255, 0), 3)
    # plt.imshow(initial)
    # plt.show()
    return initial


if __name__ == '__main__':
    folder = r'C:\Users\79777\Desktop\WORK\OCR\DrawingText\DrawingText\newText'
    save_folder = r'C:\Users\79777\Desktop\WORK\OCR\DrawingText\DrawingText\newTextwithContours'
    process_folder_with_images(folder, save_folder)
    # select_contours(
    #     r'C:\Users\User1\Downloads\OCR BluePrints.v8i.yolov8\train\images\gen1_3_jpg.rf.1e1a03ea52ae9601feb16bb4bc39f0bd.jpg'
    # )