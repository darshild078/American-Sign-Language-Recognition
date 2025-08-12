import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def get_data_generators(train_dir='dataset/train/', img_size=(64, 64), batch_size=64, test_split=0.2, seed=42, color_mode='grayscale'):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=test_split
    )
    train_gen = datagen.flow_from_directory(
        train_dir, target_size=img_size, color_mode=color_mode,
        batch_size=batch_size, class_mode='categorical', subset='training', seed=seed
    )
    val_gen = datagen.flow_from_directory(
        train_dir, target_size=img_size, color_mode=color_mode,
        batch_size=batch_size, class_mode='categorical', subset='validation', seed=seed
    )
    print("Train class indices:", train_gen.class_indices)
    return train_gen, val_gen

def load_test_data(test_dir='dataset/test/', img_size=(64, 64), class_indices=None):
    files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
    if class_indices is None:
        expected_classes = sorted([cls.lower() for cls in ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                                   'del', 'nothing', 'space']])
    else:
        expected_classes = [k.lower() for k,v in sorted(class_indices.items(), key=lambda x:x[1])]
    class_to_idx = {cls: idx for idx, cls in enumerate(expected_classes)}
    data = []
    labels = []
    class_counts = {cls:0 for cls in expected_classes}
    for img_name in files:
        cls = img_name.split('_')[0].lower()
        if cls not in class_to_idx:
            continue
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        data.append(img)
        labels.append(class_to_idx[cls])
        class_counts[cls] += 1
    if not data:
        raise RuntimeError("No test data loaded! Check your test directory and file names.")
    data = np.array(data).reshape(-1, img_size[0], img_size[1], 3)
    labels = to_categorical(labels, num_classes=len(expected_classes))
    print("Test samples per class:")
    for cls in expected_classes:
        print(f"  {cls}: {class_counts[cls]}")
    return data, labels, expected_classes
