import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import subprocess
import sys

# =========================
# تحميل النموذج من Kaggle
# =========================
def download_model_from_kaggle():
    try:
        import kaggle

        dataset_name = "salehalqattash/saleh-project"

        print("جاري تحميل النموذج من Kaggle...")

        kaggle.api.dataset_download_files(
            dataset_name,
            path='models',
            unzip=True,
            quiet=False
        )

        print("تم تحميل النموذج بنجاح!")
        return True

    except ImportError:
        print("تثبيت Kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        return download_model_from_kaggle()

    except Exception as e:
        print(f"فشل التحميل: {e}")
        return False


def check_and_download_model(model_path):
    if os.path.exists(model_path):
        print("النموذج موجود محلياً")
        return True

    print("النموذج غير موجود، سيتم التحميل...")
    return download_model_from_kaggle()


# =========================
# الكلاس الرئيسي
# =========================
class TumorImageDetector:

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.model = None
        self.is_trained = False
        self.model_path = "models/tumor_image_model.h5"

        if check_and_download_model(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path, compile=False)
                self.is_trained = True
                print("تم تحميل النموذج المدرب")
            except Exception as e:
                print(f"خطأ في تحميل النموذج: {e}")
        else:
            print("سيتم تدريب نموذج جديد")

    # =========================
    # VGG16
    # =========================
    def build_transfer_model(self):
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )

        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        return model

    # =========================
    # CNN from scratch
    # =========================
    def build_model_from_scratch(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        return model

    # =========================
    # التدريب
    # =========================
    def train(self, data_dir, epochs=20, use_transfer=True):

        if use_transfer:
            self.model = self.build_transfer_model()
            print("Using VGG16")
        else:
            self.model = self.build_model_from_scratch()
            print("Training from scratch")

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=self.image_size,
            batch_size=32,
            class_mode='binary'
        )

        test_generator = test_datagen.flow_from_directory(
            os.path.join(data_dir, 'test'),
            target_size=self.image_size,
            batch_size=32,
            class_mode='binary'
        )

        history = self.model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=epochs
        )

        self.is_trained = True

        os.makedirs("models", exist_ok=True)
        self.model.save(self.model_path)

        print("تم حفظ النموذج")

        return history

    # =========================
    # prediction
    # =========================
    def predict(self, image_path):

        if not self.is_trained:
            raise Exception("النموذج غير مدرب")

        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize(self.image_size)

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prob = float(self.model.predict(img_array)[0][0])

        label = 1 if prob >= 0.5 else 0

        return {
            "label": label,
            "probability": round(prob * 100, 2),
            "diagnosis": "خبيث" if label == 1 else "حميد"
        }