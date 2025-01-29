from PIL import ImageFont, ImageDraw, Image
# from indexes import index_to_char

import os
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from check_yolo_labels import AnnotationChecker

from settings import index_to_char, index_to_char_shorted, index_to_char_special

char_to_index = {v: k for k, v in index_to_char_shorted.items()}

roboflow_classes_sorted_asc = sorted(list(map(str, range(0, 86))))
indexes = list(range(0, 86))
roboflow_to_gen = dict(zip(indexes, roboflow_classes_sorted_asc))

font_size_to_char_width = {
    14: 7,
    15: 8,
    16: 8,
    17: 9,
    18: 9,
    19: 10,
    20: 10
}


class DataGenerator:
    def __init__(self, fonts_folder_path: str, output_dir=os.getcwd()):
        self.fonts_paths = [os.path.join(fonts_folder_path, file) for file in os.listdir(fonts_folder_path)]
        self.symbols_requiring_bbox_correction = ["*", "-", '"']

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train', "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train', "labels"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'valid', "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'valid', "labels"), exist_ok=True)

        self.fontsize_interval = [14, 18]
        self.num_symbols_interval = [6, 16]
        self.symbol_height = 28
        # self.symbol_width = {k: v for k in in range()}

        self.gen_annot_checker = AnnotationChecker()

    def set_settings(self, fontsize: list, num_symbols: list, symbol_height):
        self.fontsize_interval = fontsize
        self.num_symbols_interval = num_symbols
        self.symbol_height = symbol_height
        # self.symbol_width = symbol_width

    @staticmethod
    def xyxy_to_yolo_xyhw(xyxy: list, imgsz: list):
        left, top, right, bottom = xyxy
        x_center = (left + right) / 2 / imgsz[0]
        y_center = (top + bottom) / 2 / imgsz[1]
        width = (right - left + 1) / imgsz[0]
        height = (bottom - top + 2) / imgsz[1]
        return x_center, y_center, width, height

    def check_generated_annots(self):
        source_img = os.path.join(self.output_dir, 'valid', "images")
        source_txt = os.path.join(self.output_dir, 'valid', "labels")
        output = os.path.join(self.output_dir, 'valid', "check")
        os.makedirs(output, exist_ok=True)
        self.gen_annot_checker.check(source_img, source_txt, output)

    def generate_image_and_labels(self, image_id, final_yolo_size=None):
        ### FONTS ###
        font_path = random.choice(self.fonts_paths)  # шрифт для генерации символов
        font_size = random.randint(self.fontsize_interval[0], self.fontsize_interval[1])
        font = ImageFont.truetype(font_path, font_size)

        ### IMAGE ###
        symbol_width = font_size_to_char_width[font_size]
        num_symbols = random.randint(self.num_symbols_interval[0], self.num_symbols_interval[1])
        image_width = symbol_width * num_symbols
        image_size = (image_width, self.symbol_height)
        image = Image.new("RGB", image_size, "white")
        draw = ImageDraw.Draw(image)

        ### TEXT ###
        symbols = "".join(random.choice(list(index_to_char_shorted.values())) for _ in range(num_symbols))
        text_bbox = draw.textbbox((0, 0), symbols, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        current_x = 0
        y_start = (self.symbol_height - text_height) // 2
        yolo_labels = []
        for symbol in symbols:
            # if symbol not in index_to_char_shorted.values():
            #     continue
            bbox = list(draw.textbbox((current_x, y_start), symbol, font=font))
            if symbol in self.symbols_requiring_bbox_correction:
                bbox[3] = (bbox[3] + bbox[1]) // 2
            # left, top, right, bottom = bbox
            if symbol != " ":
                x_center, y_center, width, height = self.xyxy_to_yolo_xyhw(bbox, image_size)
                class_id = char_to_index[symbol]
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            draw.text((current_x, y_start), symbol, fill="black", font=font)
            current_x += symbol_width

        if final_yolo_size is not None:
            image = image.resize((final_yolo_size, final_yolo_size))

        if random.randint(0, 100) >= 10:
            image.save(os.path.join(self.output_dir, 'train', "images", f"{image_id}.jpg"))
            with open(os.path.join(self.output_dir, 'train', "labels", f"{image_id}.txt"), "w") as f:
                f.write("\n".join(yolo_labels))
        else:
            image.save(os.path.join(self.output_dir, 'valid', "images", f"{image_id}.jpg"))
            with open(os.path.join(self.output_dir, 'valid', "labels", f"{image_id}.txt"), "w") as f:
                f.write("\n".join(yolo_labels))


if __name__ == "__main__":
    path_to_fonts = r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/fonts'
    result_folder = r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/data/yolo_generated_dataset_v3'
    dg = DataGenerator(path_to_fonts, result_folder)
    NUM_IMAGES = 5000
    # Генерация датасета
    # try:
    for i in tqdm(range(NUM_IMAGES)):
        dg.generate_image_and_labels(i, final_yolo_size=512)

    dg.check_generated_annots()
    # except Exception as ex:
    #     print(ex)
    #     print("Something went wrong :()")
    # print(sorted(list(map(str, index_to_char.keys()))))

