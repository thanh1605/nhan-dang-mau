import os
import random
import shutil

# Đường dẫn thư mục gốc chứa toàn bộ dữ liệu ảnh
SOURCE_DIR = "plantvillage dataset\color"     # chỉnh lại theo máy bạn
DEST_DIR = "dataset_split"     # nơi sẽ lưu tập train/val/test
SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

os.makedirs(DEST_DIR, exist_ok=True)

for cls_name in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls_name)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total * SPLIT_RATIOS['train'])
    n_val = int(n_total * SPLIT_RATIOS['val'])

    split_sets = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }

    for split_name, file_list in split_sets.items():
        split_cls_dir = os.path.join(DEST_DIR, split_name, cls_name)
        os.makedirs(split_cls_dir, exist_ok=True)
        for fname in file_list:
            shutil.copy(os.path.join(cls_path, fname),
                        os.path.join(split_cls_dir, fname))

print("✅ Đã chia xong dữ liệu theo tỷ lệ 80/10/10!")
