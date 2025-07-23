# %%
import os
import cv2
import shutil
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
import glob
from PIL import Image

# %%
# 1. Видалення порожніх .txt файлів з мітками
def remove_empty_label_files(folder_path):
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                full_path = os.path.join(root, file)
                if os.stat(full_path).st_size == 0:
                    os.remove(full_path)
                    count += 1
    print(f"[INFO] Removed {count} empty label files from {folder_path}")

remove_empty_label_files('./data/train/labels')
remove_empty_label_files('./data/val/labels')

# %%
# 2. Видалення зіпсованих або надто малих зображень
def remove_bad_images(image_dir):
    removed = 0
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[BAD] Can't read: {img_path} — deleting")
                os.remove(img_path)
                removed += 1
            else:
                h, w = img.shape[:2]
                if h < 10 or w < 10:
                    print(f"[BAD] Too small ({w}x{h}): {img_path} — deleting")
                    os.remove(img_path)
                    removed += 1
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
    print(f"[INFO] Removed {removed} bad images from {image_dir}")

remove_bad_images('data/train/images')
remove_bad_images('data/val/images')

# %%
# 3. Основна логіка
def main():
    # 3.1 Створення YAML конфігурації
    data_yaml = {
        'train': './data/train',
        'val': './data/val',
        'nc': 10,
        'names': [
            'door', 'open_door', 'cabinet_door', 'fridge_door',
            'window', 'chair', 'table', 'cabinet', 'sofa', 'pillar'
        ]
    }

    if not os.path.exists('indoor.yaml'):
        with open('indoor.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
        print("[INFO] Created indoor.yaml")
    else:
        print("[INFO] indoor.yaml already exists — skipping write.")

    # %%
    # 3.2 Завантаження моделі
    print("[INFO] Loading YOLOv9s model...")
    model = YOLO('yolov9s.pt')

    # %%
    # 3.3 Тренування
    print("[INFO] Starting training...")
    results = model.train(
        data='indoor.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='indoor_yolov9s6',
        device=0,      # GPU (або 'cpu')
        workers=0      # важливо для Windows!
    )

    # %%
    # 3.4 Побудова графіків метрик
    csv_path = 'runs/detect/indoor_yolov9s6/results.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        df[['metrics/mAP_0.5', 'metrics/precision', 'metrics/recall']].plot(
            figsize=(10, 6), title='Training Metrics'
        )
        plt.grid(True)
        plt.savefig('training_metrics.png')
        plt.show()
    else:
        print("[WARNING] results.csv not found. Skipping metric plots.")

    # %%
    # 3.5 Оцінка моделі
    print("[INFO] Running validation...")
    metrics = model.val()

    print("[INFO] Confusion matrix:")
    try:
        print(metrics.box.confusion_matrix.matrix)
    except Exception as e:
        print(f"[WARNING] Couldn't print confusion matrix: {e}")

    # %%
    # 3.6 Візуалізація помилок по класах
    try:
        metrics.box.plot()
    except Exception as e:
        print(f"[WARNING] Couldn't plot class-wise metrics: {e}")

    # %%
    # 3.7 Збереження фінальної моделі
    best_weights = getattr(model, 'ckpt_path', 'runs/detect/indoor_yolov9s6/weights/best.pt')
    print(f"[INFO] Model saved at: {best_weights}")

    # %%
    # 3.8 Тестування на нових зображеннях
    if os.path.exists('data/test/images/'):
        print("[INFO] Running predictions on test images...")
        model.predict('data/test/images/', save=True, conf=0.4)
        print("[INFO] Predictions saved to runs/detect/predict/")
    else:
        print("[WARNING] Test image folder not found. Skipping prediction.")

    # %%
    # 3.9 Відкриття першого передбаченого зображення (опціонально)
    pred_images = sorted(glob.glob('runs/detect/predict/*.jpg'))
    if pred_images:
        print(f"[INFO] Showing prediction preview: {pred_images[0]}")
        img = Image.open(pred_images[0])
        img.show()
    else:
        print("[INFO] No predicted images found to display.")

# %%
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Для Windows
    main()
