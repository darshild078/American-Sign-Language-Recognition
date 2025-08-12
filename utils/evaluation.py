import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os

def evaluate_model(model, X_test, y_test, test_classes):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")
    return test_loss, test_acc

def generate_confusion_matrix(model, X_test, y_test, test_classes, save_path='results/confusion_matrix.png'):
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    labels = list(range(len(test_classes)))
    cm = confusion_matrix(y_true_classes, y_pred_classes, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_classes)
    plt.figure(figsize=(12, 12))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
    plt.title('Test Set Confusion Matrix')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    print("Per-class label counts (order matches confusion matrix):")
    for idx, cls in enumerate(test_classes):
        y_true_count = np.sum(y_true_classes == idx)
        y_pred_count = np.sum(y_pred_classes == idx)
        print(f"{cls}: in y_true={y_true_count}, in y_pred={y_pred_count}")
    report = classification_report(
        y_true_classes, y_pred_classes, labels=labels, target_names=test_classes, zero_division=0
    )
    print("Classification Report:")
    print(report)
