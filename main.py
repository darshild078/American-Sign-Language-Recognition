import sys
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np
from utils.data_prep import get_data_generators, load_test_data
from utils.evaluation import evaluate_model, generate_confusion_matrix
from models.cnn_model import build_cnn
from models.transfer_mobilenet import build_mobilenet_model, fine_tune_mobilenet

def main(eval_only=False, model_type='cnn'):
    if model_type == 'cnn':
        color_mode = 'grayscale'
        img_size = (64, 64)
    elif model_type == 'mobilenet':
        color_mode = 'rgb'
        img_size = (128, 128)
    else:
        raise ValueError("Model type must be 'cnn' or 'mobilenet'")

    train_gen, val_gen = get_data_generators(color_mode=color_mode, img_size=img_size)
    class_indices = train_gen.class_indices

    # Ensure checkpoints dir exists and set paths
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'best_{model_type}.h5')

    if model_type == 'cnn':
        print("Using CNN model architecture")
        if eval_only:
            model = load_model(checkpoint_path)
        else:
            model = build_cnn(input_shape=img_size+(1,))
    else:
        print("Using MobileNetV2 transfer learning architecture")
        if eval_only:
            model = load_model(checkpoint_path)
        else:
            model = build_mobilenet_model(input_shape=img_size+(3,), num_classes=len(class_indices))

    if not eval_only:
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
        history = model.fit(
            train_gen,
            epochs=15,
            validation_data=val_gen,
            callbacks=[early_stop, lr_reduce, checkpoint],
        )
        model.save(checkpoint_path)

    X_test, y_test, test_classes = load_test_data(class_indices=class_indices, test_dir='dataset/test/', img_size=img_size)
    evaluate_model(model, X_test, y_test, test_classes)
    generate_confusion_matrix(model, X_test, y_test, test_classes)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train/Evaluate ASL Alphabet Model")
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mobilenet'], help="Choose model architecture")
    parser.add_argument('--eval_only', action='store_true', help="Only evaluate a saved model")
    args = parser.parse_args()
    main(eval_only=args.eval_only, model_type=args.model)