import cv2
import os
import shutil
from tqdm import tqdm
from pathlib import Path


def get_name_filepath_dict(folder):
    names_dict = dict()
    # names = list(map(lambda x: x.split('.')[0], os.listdir(folder)))
    for filename in os.listdir(folder):
        full_path = os.path.join(folder, filename)
        names_dict[filename.split('.')[0]] = full_path
    return names_dict


def read_lines_from_file_to_list(filepath):
    with open(filepath, 'r') as t:
        lines = [line.strip() for line in t.readlines()]
    return lines


def copy_files_from_src_folder_to_dst_folder(src_folder, dst_folder):
    for file in tqdm(os.scandir(src_folder), total=len(os.listdir(src_folder))):
        file_src = file.path
        file_dst = os.path.join(dst_folder, Path(file_src).name)
        shutil.copy2(file_src, file_dst)


def main():
    sample_types = ['train', 'valid']
    data_types = ['images', 'labels']
    for sample_type in sample_types:
        for data_type in data_types:
            src_folder = rf'/root/171_data2/_share/Kirill/KirillOCR/YOLO/data/yolo_generated_dataset_v3/{sample_type}/{data_type}'
            dst_folder = rf'/root/171_data2/_share/Kirill/KirillOCR/YOLO/data/OCR-BluePrints-17_with_gen_new_classes/{sample_type}/{data_type}'
            copy_files_from_src_folder_to_dst_folder(src_folder, dst_folder)