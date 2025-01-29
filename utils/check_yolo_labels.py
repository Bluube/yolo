import os
import cv2
from pathlib import Path
from tqdm import tqdm


class AnnotationChecker:
    def __init__(self):
        pass

    @staticmethod
    def xywh_to_ymin_ymax_xmin_xmax(x, y, w, h, initial_shape: tuple) -> list:
        width = initial_shape[1] * w
        height = initial_shape[0] * h
        xmin = initial_shape[1] * x - width / 2
        xmax = xmin + width
        ymin = initial_shape[0] * y - height / 2
        ymax = ymin + height

        return list(map(int, [ymin, ymax, xmin, xmax]))

    def check(self, folder, txt_folder, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        for file in tqdm(os.scandir(folder), total=len(os.listdir(folder))):
            img = cv2.imread(file.path)
            save_path = os.path.join(save_folder, Path(file.path).name)
            with open(os.path.join(txt_folder, Path(file.path).stem + '.txt'), 'r') as t:
                lines = [line.strip() for line in t.readlines()]
            for line in lines:
                cl, x, y, w, h = line.split(' ')
                x, y, w, h = list(map(float, [x, y, w, h]))
                ymin, ymax, xmin, xmax = self.xywh_to_ymin_ymax_xmin_xmax(x, y, w, h, img.shape[:2])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                cv2.putText(img, f'{cl}', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imwrite(save_path, img)


if __name__ == "__main__":
    name = 'train'
    folder_init = r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/data/OCR-BluePrints-17_with_gen_new_classes'
    # for name in ['train', 'valid', 'test']:
    folder = rf'{folder_init}/{name}/images'
    txt = rf'{folder_init}/{name}/labels'
    save = rf'{folder_init}/{name}/check'
    AnnotationChecker().check(folder, txt, save)