import os
import shutil

data_dir = './CUB_200_2011/'
images_dir = os.path.join(data_dir, 'images')

train_dir = os.path.join(data_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

test_dir = os.path.join(data_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

classes = os.listdir(images_dir)

with open('./CUB_200_2011/images.txt') as f:
    image_names = f.readlines()

with open('./CUB_200_2011/train_test_split.txt') as f:
    train_test_split = f.readlines()

for train_idx, image_name in zip(train_test_split, image_names):
    image_idx, train_flag = train_idx.split()
    num, class_image_name = image_name.split()
    
    image_path = os.path.join(images_dir, class_image_name)
    class_name = class_image_name.split('/')[0]
    
    train_class_dir = os.path.join(train_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(test_class_dir, exist_ok=True)
    
    if train_flag == '0':
        shutil.copy(image_path, train_class_dir)
    else:
        shutil.copy(image_path, test_class_dir)
