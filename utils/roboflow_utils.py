from roboflow import Roboflow


def generate_roboflow_to_gen_classes_dict():
    roboflow_classes_sorted_asc = sorted(list(map(str, range(0, 86))))
    indexes = list(range(0, 86))
    roboflow_to_gen = dict(zip(indexes, roboflow_classes_sorted_asc))
    return roboflow_to_gen


def get_roboflow_data(vers: int):
    api_key = '2NvFXrvKZR2ePwjgUrNH'
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("ocrblueprints").project("ocr-blueprints")
    version = project.version(vers)
    dataset = version.download("yolov8")


def rename_classes_from_roboflow_in_text_files(source_folder, save_folder=None):
    # os.makedirs(save_folder, exist_ok=True)
    for file in os.scandir(source_folder):
        # save_path = os.path.join(save_folder, Path(file.path).name)
        if 'rf' not in file.path:
            continue
        with open(file.path, 'r') as t:
            lines = [line.strip() for line in t.readlines()]
        t.close()
        with open(file.path, 'w') as k:
            for line in lines:
                cl, x, y, w, h = line.split(' ')
                cl_new = roboflow_to_gen[int(cl)]
                k.write(f'{cl_new} {x} {y} {w} {h}\n')


def make_file_with_all_classes(file_path):
    x, y, w, h = [0.5]*4
    with open(file_path, 'w') as f:
        for i in range(86):
            f.write(f'{i} {x} {y} {w} {h}\n')


def leave_files_only_with_specified_classes(txt_folder, images_folder=None, result_folder=None, classes=None):
    new_txt_folder = result_folder / 'labels'
    new_images_folder = result_folder / 'images'
    os.makedirs(new_txt_folder, exist_ok=True)
    os.makedirs(new_images_folder, exist_ok=True)
    for file in tqdm(os.scandir(txt_folder), total=len(os.listdir(txt_folder))):
        file_name = Path(file.path).name
        image_path = images_folder / f'{Path(file.path).stem}.jpg'
        new_image_path = new_images_folder / f'{Path(file.path).stem}.jpg'
        # print(file.path)
        lines = read_lines_from_file_to_list(file.path)
        classes_in_file = list(map(lambda x: int(x.split(' ')[0]), lines))
        # print(classes_in_file)
        intersect = set(classes) & set(classes_in_file)
        if not intersect:
            continue
        shutil.copy2(image_path, new_image_path)
        with open(new_txt_folder / f'{file_name}', 'w') as k:
            for line in lines:
                cl, x, y, w, h = line.split(' ')
                if int(cl) not in classes:
                    continue
                if cl == '23' or cl == '66':
                    k.write(f'45 {x} {y} {w} {h}\n')
                else:
                    k.write(f'{cl} {x} {y} {w} {h}\n')