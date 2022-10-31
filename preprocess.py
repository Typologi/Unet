import os


def convert_image_to_png(path):
    for filename in os.listdir(path):
        source = os.path.join(path, filename)
        print(f'ffmpeg -i {source} {filename[:-4]}.png')
        os.system(f'ffmpeg -i {source} ' + os.path.join(path, filename[:-4] + ".png"))


def process_dataset(path, train=True):
    if train:
        images = os.path.join(path, 'train/images')
        labels = os.path.join(path, 'train/1st_manual')
    else:
        images = os.path.join(path, 'test/images')
        labels = os.path.join(path, 'test/1st_manual')
    images_list = []
    labels_list = []
    for image_name in os.listdir(images):
        if not image_name.endswith('.png'):
            continue
        images_list.append(os.path.join(images, image_name))
        labels_list.append(os.path.join(labels, image_name))
    if train:
        with open('./datasets/datasets.txt', 'a+') as f:
            for (i, image_path) in enumerate(images_list):
                print(f'{image_path}|{labels_list[i]}'.replace('\\', '/'), file=f)
    else:
        with open('./datasets/testsets.txt', 'a+') as f:
            for (i, image_path) in enumerate(images_list):
                print(f'{image_path}|{labels_list[i]}'.replace('\\', '/'), file=f)


if __name__ == '__main__':
    if os.path.exists('./datasets/datasets.txt'):
        os.remove('./datasets/datasets.txt')
    process_dataset('./datasets/CHASE')
    process_dataset('./datasets/DRIVE')
    if os.path.exists('./datasets/testsets.txt'):
        os.remove('./datasets/testsets.txt')
    process_dataset('./datasets/CHASE', train=False)
    process_dataset('./datasets/DRIVE', train=False)
