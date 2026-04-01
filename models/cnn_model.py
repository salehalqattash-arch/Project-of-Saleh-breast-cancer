# نموذج CNN للكشف عن الأورام من الصور

import os
import numpy as np
# المكتبة الرئيسية لبناء الشبكة العصبية
import tensorflow as tf
from tensorflow import keras
# طبقات الشبكة (Conv2D, MaxPooling, Dense, ...)
from tensorflow.keras import layers
# لقراءة الصور من المجلدات وتحسينها
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# لفتح ومعالجة الصور عند التنبؤ
from PIL import Image 

#نموذج للكشف عن الأورام من الصور
class TumorImageDetector:
    #تهيئة النموذج
    def __init__(self, image_size=(224, 224)):
        """
        image_size: حجم الصورة بعد التحجيم (العرض, الارتفاع)
        224×224 هو الحجم القياسي الذي يتوقعه VGG16
        """
        self.image_size = image_size
        self.model = None
        self.is_trained = False
        self.model_path = "models/tumor_image_model.h5"
        
        # محاولة تحميل نموذج مدرب مسبقاً إذا كان موجوداً
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                self.is_trained = True
                print(" تم تحميل النموذج المدرب مسبقاً")
            except:
                print(" فشل تحميل النموذج، سيتم تدريب نموذج جديد")

    def build_transfer_model(self):
        """
        استخدام نموذج VGG16 المدرب مسبقاً (Transfer Learning)
        هذه الطريقة أفضل إذا كانت البيانات محدودة
        """
        # تحميل VGG16  Visual Geometry Group
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',           # أوزان مدربة على ملايين الصور
            include_top=False,            # نستبعد الطبقات الأخيرة (التصنيف)
            input_shape=(*self.image_size, 3)  # 224x224x3 (RGB)
        )
        
        #   لا نعيد تدريب base_model 
        base_model.trainable = False
        
        #  إضافة طبقات جديدة مخصصة لسرطان الثدي
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),  # تحويل الخريطة إلى مصفوفة
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),               # يمنع الإفراط في التخصيص
            layers.Dense(1, activation='sigmoid')  # طبقة الإخراج (0=حميد، 1=خبيث)
        ])
        
        return model
    
    def build_model_from_scratch(self):
        """
        بناء نموذج CNN من الصفر
        هذه الطريقة تحتاج بيانات أكثر
        """
        model = keras.Sequential([
            # الطبقة 1: تكتشف الحواف والزوايا
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            layers.MaxPooling2D(2, 2),
            
            # الطبقة 2: تكتشف أشكال أكثر تعقيداً
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # الطبقة 3: تكتشف ملامح عالية المستوى
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # تحويل الخريطة إلى مصفوفة
            layers.Flatten(),
            
            # طبقات متصلة بالكامل (مثل النماذج التقليدية)
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # طبقة الإخراج
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    #تدريب النموذج على الصور
    def train(self, data_dir, epochs=20, use_transfer=True):
        """
        data_dir: المجلد الذي يحتوي على train/ و test/ داخله
        epochs: عدد مرات التدريب
        use_transfer: True = استخدام VGG16، False = بناء من الصفر
        """
        # اختيار طريقة بناء النموذج
        if use_transfer:
            self.model = self.build_transfer_model()
            print(" باستخدام VGG16 (Transfer Learning)")
        else:
            self.model = self.build_model_from_scratch()
            print(" بناء نموذج من الصفر")
        
        # تجميع النموذج (تحديد المحسن ودالة الخسارة)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )

        # مولد البيانات للتدريب (مع تحسين الصور)
        train_datagen = ImageDataGenerator(
            rescale=1./255,           # تطبيع الألوان (0-255 → 0-1)
            rotation_range=20,        # تدوير عشوائي
            width_shift_range=0.2,    # إزاحة أفقية
            height_shift_range=0.2,   # إزاحة رأسية
            zoom_range=0.2,           # تكبير/تصغير
            horizontal_flip=True      # قلب أفقي
        )
        
        # مولد البيانات للاختبار 
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # قراءة صور التدريب
        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=self.image_size,
            batch_size=32,
            class_mode='binary'
        )
        
        # قراءة صور الاختبار
        test_generator = test_datagen.flow_from_directory(
            os.path.join(data_dir, 'test'),
            target_size=self.image_size,
            batch_size=32,
            class_mode='binary'
        )
        
        print(f"\n صور التدريب: {train_generator.samples}")
        print(f" صور الاختبار: {test_generator.samples}")

        # تدريب النموذج
        history = self.model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=epochs,
            verbose=1
        )
        
        self.is_trained = True
        
        # حفظ النموذج للاستخدام لاحقاً
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"\n تم حفظ النموذج في: {self.model_path}")
        
        # عرض النتائج النهائية
        final_train_acc = history.history['accuracy'][-1] * 100
        final_test_acc = history.history['val_accuracy'][-1] * 100
        final_train_loss = history.history['loss'][-1]
        final_test_loss = history.history['val_loss'][-1]
        
        print(f"\n نتائج التدريب النهائية:")
        print(f"   دقة التدريب:   {final_train_acc:.2f}%")
        print(f"   دقة الاختبار:  {final_test_acc:.2f}%")
        print(f"   خسارة التدريب: {final_train_loss:.4f}")
        print(f"   خسارة الاختبار: {final_test_loss:.4f}")
        
        return history
    
    #تشخيص صورة جديدة
    def predict(self, image_path):
        if not self.is_trained:
            raise Exception(" النموذج لم يدرب بعد! قم بتشغيل train() أولاً")
        
        # فتح الصورة
        img = Image.open(image_path)
        
        # تحويل إلى RGB (إذا كانت بتدرج رمادي)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # تحجيم الصورة
        img = img.resize(self.image_size)
        
        # تحويل إلى مصفوفة وتطبيع
        img_array = np.array(img) / 255.0
        
        # إضافة بُعد الدفعة (batch dimension)
        img_array = np.expand_dims(img_array, axis=0)
        
        #التنبؤ
        prob = float(self.model.predict(img_array, verbose=0)[0][0])
        
        # تحديد التصنيف
        label = 1 if prob >= 0.5 else 0
        
        return {
            "label": label,
            "probability": round(prob * 100, 2),
            "diagnosis": "خبيث" if label == 1 else "حميد"
        }