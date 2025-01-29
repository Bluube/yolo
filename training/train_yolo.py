from ultralytics import YOLO
import torch
import os
print(torch.cuda.is_available())


def main():
    project = r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/runs_specified/v13_8l_batch_32_640_with_gen_new_classes_v3'
    os.makedirs(project, exist_ok=True)
    # experiment = r'v7_m'
    model = YOLO('yolov8l.pt')
    model.tune(data=r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/data/OCR-BluePrints-17_with_gen_new_classes/data.yaml',
            #    cfg=r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/runs_specified/v13_8l_batch_32_640_with_gen_new_classes/tune/best_hyperparameters.yaml',
               iterations=50, epochs=100, imgsz=640, batch=32, patience=10, mosaic=0.0, fliplr=0.0, close_mosaic=0, project=project, augment=True)

    # model.tune(
    #     #    cfg=r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/runs_specified/v13_8l_batch_32_640_with_gen_new_classes/tune/best_hyperparameters.yaml',
    #             iterations=50,
    #             cfg=r'/root/171_data2/_share/Kirill/KirillOCR/YOLO/runs_specified/v13_8l_batch_32_640_with_gen_new_classes_v2/train23/args.yaml',
    #             project=project, augment=True)


if __name__ == "__main__":
    # print(list(range(86)))
    main()
