# تدريب نموذج CNN للكشف عن الأورام من الصور

import os
import sys
import tensorflow as tf
from models.cnn_model import TumorImageDetector

#  الإعدادات 
DATA_DIR = "breast_cancer_images"     # مجلد الصور (بعد التقسيم)
MODEL_SAVE_PATH = "models/tumor_image_model.h5"
EPOCHS = 20                           # عدد مرات التدريب
# نستخدم VGG16 لأنه نموذج جاهز تم تدريبه على ملايين الصور، فيعرف كيف يستخرج الملامح من أي صورة،
#  ونحن فقط نعلمه التمييز بين الخبيث والحميد بدلاً من تعليمه كل شيء من الصفر.
USE_TRANSFER = True                   # استخدام VGG16 

print("=" * 50)
print(" تدريب نموذج CNN للكشف عن الأورام من الصور")
print("=" * 50)

print("\nالتحقق من مجلد البيانات...")
if not os.path.exists(DATA_DIR):
    print(f" مجلد {DATA_DIR} غير موجود!")
    print(" قم بتشغيل split_data.py أولاً لتقسيم الصور")
    sys.exit(1)

#  التحقق من وجود مجلدات train و test
train_dir = os.path.join(DATA_DIR, 'train')
test_dir = os.path.join(DATA_DIR, 'test')
if not os.path.exists(train_dir):
    print(f" مجلد {train_dir} غير موجود!")
    sys.exit(1)
if not os.path.exists(test_dir):
    print(f" مجلد {test_dir} غير موجود!")
    sys.exit(1)

malignant_train = os.path.join(train_dir, 'malignant')
benign_train = os.path.join(train_dir, 'benign')

# عرض إحصائيات الصور
if os.path.exists(malignant_train) and os.path.exists(benign_train):
    malignant_count = len(os.listdir(malignant_train))
    benign_count = len(os.listdir(benign_train))
    print(f" صور التدريب:")
    print(f"- malignant: {malignant_count} صورة")
    print(f"- benign: {benign_count} صورة")
    print(f"- المجموع: {malignant_count + benign_count} صورة")
else:
    print(f" مجلدات الصور غير مكتملة!")
    sys.exit(1)

print("\n بناء النموذج...")
detector = TumorImageDetector()

# تدريب النموذج
history = detector.train(
    data_dir=DATA_DIR,
    epochs=EPOCHS,
    use_transfer=USE_TRANSFER
)

print("\n حفظ النموذج...")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
detector.model.save(MODEL_SAVE_PATH)
print(f" تم حفظ النموذج في: {MODEL_SAVE_PATH}")

print("\n" + "=" * 50)
print(" اكتمل التدريب بنجاح!")
print("=" * 50)

final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\n نتائج التدريب النهائية:")
# هذا هو الرقم المهم! يعبر عن قدرة النموذج الحقيقية
print(f"   دقة التدريب:   {final_train_acc:.2f}%")
print(f"   دقة الاختبار:  {final_val_acc:.2f}%")

# كلما كانت أقل، كان النموذج أفضل
print(f"   خسارة التدريب: {final_train_loss:.4f}")
print(f"   خسارة الاختبار: {final_val_loss:.4f}")

print(f"\n النموذج المحفوظ: {MODEL_SAVE_PATH}")
