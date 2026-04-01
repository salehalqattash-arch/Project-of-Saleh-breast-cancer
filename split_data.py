# تقسيم الصور عشوائياً إلى تدريب واختبار

import os
import shutil
import random
from pathlib import Path


# الإعدادات
SOURCE_DIR = "images_dataset"              # اسم مجلد الصور الأصلي
TARGET_DIR = "breast_cancer_images"     # المجلد النهائي
TRAIN_RATE = 0.8                       # 80% تدريب، 20% اختبار
RANDOM_SEED = 42                        # نفس الرقم = نفس التقسيم

# فئات البيانات من المصدر
CLASSES = ['malignant', 'benign']

# إنشاء المجلدات للتدريب ولاختبار بشكل عشوائي
for split in ['train', 'test']:
    for cls in ['malignant', 'benign']:
        folder_path = os.path.join(TARGET_DIR, split, cls)
        os.makedirs(folder_path, exist_ok=True)
        print(f" create : {folder_path}")

#  ضبط العشوائية لتكرار النتائج بشكل صحيح
random.seed(RANDOM_SEED)

#  تقسيم كل فئة لتدريب واختبار  
for cls in ['malignant', 'benign']:
    source_path = os.path.join(SOURCE_DIR, cls)
    
    # التحقق من وجود المجلد
    if not os.path.exists(source_path):
        print(f" folder {cls} not exists.")
        continue
    
    # جلب قائمة الصور
    images = [f for f in os.listdir(source_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if len(images) == 0:
        print(f"   folder {cls} not have photos!")
        continue
    
    print(f" total images : {len(images)}")
    
    # خلط عشوائي
    random.shuffle(images)
    
    # تقسيم
    split_id = int(len(images) * TRAIN_RATE)
    train_images = images[:split_id]
    test_images = images[split_id:]
    
    # نسخ للتدريب
    for img in train_images:
        src = os.path.join(source_path, img)
        dst = os.path.join(TARGET_DIR, 'train', cls, img)
        shutil.copy2(src, dst)
    
    # نسخ للاختبار
    for img in test_images:
        src = os.path.join(source_path, img)
        dst = os.path.join(TARGET_DIR, 'test', cls, img)
        shutil.copy2(src, dst)
    
    print(f" {cls}: training is: {len(train_images)}  | testing is: {len(test_images)} ")
