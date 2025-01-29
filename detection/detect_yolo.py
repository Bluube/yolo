from ultralytics import YOLO
from pathlib import Path
import collections
import csv
import cv2
import pandas as pd
import numpy as np
import os
import torch

print(torch.cuda.is_available())

from utils.image_utils.not_deep_learning_feature_extractors import ImageFeatureExtractor

index_to_char = {
    0: " ",
    1: '"',
    2: "(",
    3: ")",
    4: "*",
    5: ",",
    6: "-",
    7: ".",
    8: "0",
    9: "1",
    10: "2",
    11: "3",
    12: "4",
    13: "5",
    14: "6",
    15: "7",
    16: "8",
    17: "9",
    18: ":",
    19: "A",
    20: "E",
    21: "H",
    22: "I",
    23: "M",
    24: "R",
    25: "S",
    26: "T",
    27: "[",
    28: "]",
    29: "a",
    30: "e",
    31: "h",
    32: "i",
    33: "r",
    34: "s",
    35: "±",
    36: "×",
    37: "А",
    38: "В",
    39: "Г",
    40: "Д",
    41: "Е",
    42: "И",
    43: "К",
    44: "Л",
    45: "М",
    46: "О",
    47: "П",
    48: "Р",
    49: "С",
    50: "Т",
    51: "Ц",
    52: "Ь",
    53: "Э",
    54: "а",
    55: "б",
    56: "в",
    57: "г",
    58: "д",
    59: "е",
    60: "ж",
    61: "з",
    62: "и",
    63: "й",
    64: "к",
    65: "л",
    66: "м",
    67: "н",
    68: "о",
    69: "п",
    70: "р",
    71: "с",
    72: "т",
    73: "у",
    74: "ф",
    75: "х",
    76: "ц",
    77: "ч",
    78: "ш",
    79: "ы",
    80: "ь",
    81: "я",
    82: "№",
    83: "∅",
    84: "∘",
    85: "√",
}


def main(source):
    REPORT_COLUMNS = ['image_name', 'text']
    path_to_model = r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/runs/detect/tune/weights/best.pt'
    out_path = r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/stats.csv'
    model = YOLO(path_to_model)
    yolo_to_index = model.names
    results = model.predict(source, imgsz=512, verbose=False)
    stats = pd.DataFrame(columns=REPORT_COLUMNS)
    ind = 0
    for r in results:
        stats.loc[ind, 'image_name'] = Path(r.path).name
        boxes = r.boxes
        x1_to_class_name = dict()
        if not boxes:
            stats.loc[ind, 'text'] = ''
            ind += 1
            continue
        # template = np.copy(r.orig_img)
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            coords = list(map(int, b.tolist()))
            c = int(box.cls.item())
            class_name = index_to_char[int(yolo_to_index[c])]
            x1_to_class_name[coords[0]] = class_name

        sorted_x1_to_class_name = collections.OrderedDict(sorted(x1_to_class_name.items()))
        print(sorted_x1_to_class_name)
        stats.loc[ind, 'text'] = ''.join(list(sorted_x1_to_class_name.values()))
        ind += 1
        # break

    print(stats)
    stats.to_csv(out_path, index=False)


def get_visual_convenient_predict(path_to_weights, path_to_files, path_to_save, size=512, hog=False):
    model = YOLO(path_to_weights)
    os.makedirs(path_to_save, exist_ok=True)
    ife = ImageFeatureExtractor()
    for file in os.scandir(path_to_files):
        img = cv2.imread(file.path)
        if not hog:
            # if img.ndim > 2:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ret, im_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # img = cv2.fastNlMeansDenoising(img, h=10)
            if img.ndim < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_init = cv2.resize(img, (size, size))
            img = ife.get_hog_image(img_init)
            # print(img.shape)
            img = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)

        results = model(img, imgsz=size, conf=0.5, verbose=False)
        for r in results:
            boxes = r.boxes
            if not boxes:
                continue
            # template = np.copy(r.orig_img)
            i = 0
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                coords = list(map(int, b.tolist()))
                c = model.names[int(box.cls.item())]
                # c = int(box.cls.item())
                prob = round(box.conf.item(), ndigits=3)
                draw_results_on_image(img, coords, c, prob, i)
                i += 1

        img = cv2.resize(img, (1024, size))
        cv2.imwrite(os.path.join(path_to_save, Path(file.path).name), img)


def draw_results_on_image(img, pts, label, prob, i):
    if img.ndim < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color = (0, 255, 0)
    thickness = 1
    startpoint = pts[0], pts[1]
    endpoint = pts[2], pts[3]
    xavg = (pts[0] + pts[2]) // 2
    y = pts[3] if i % 2 == 0 else pts[1] + 20
    cv2.rectangle(img, startpoint, endpoint, color, thickness)
    cv2.putText(img, f'{index_to_char[int(label)]}', (xavg - 10, y), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                (255, 0, 0), thickness, bottomLeftOrigin=False)



def get_raw_prediction(path_to_weights, path_to_files):
    model = YOLO(path_to_weights)
    results = model(path_to_files, imgsz=512, save=True, conf=0.1, verbose=False)


if __name__ == "__main__":
    imgsz = 640
    hog = False
    a = r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/OCR-BluePrints-16/valid/images'
    source = r'/root/171_data2/_share/Kirill/KirillOCR/vectorization/runs/detect/predict2/crops/180_resized'
    path_to_weights = r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/runs_specified/v13_8l_batch_32_640_with_gen_new_classes_v3/tune/weights/best.pt'
    save_path = r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/results_visualization/watch_results_valid_16'
    get_visual_convenient_predict(path_to_weights, a, save_path, size=imgsz, hog=hog)