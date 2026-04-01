# 🔬 نظام الكشف عن الأورام الخبيثة | Tumor Detection AI System

## نظرة عامة | Overview

نظام ذكاء اصطناعي متكامل للكشف عن الأورام الخبيثة يستخدم تقنيات Machine Learning و Deep Learning.

> AI-powered system for malignant tumor detection using 4 ML models ensemble.

---

## 🚀 تشغيل المشروع | Quick Start

### 1. تثبيت المتطلبات | Install Requirements
```bash
pip install -r requirements.txt
```

### 2. تشغيل التطبيق | Run Application
```bash
python app.py
```

### 3. فتح المتصفح | Open Browser
```
http://localhost:5000
```

---

## 🧠 النماذج المستخدمة | Models Used

| النموذج | Model | الدقة | Accuracy |
|---------|-------|-------|----------|
| Random Forest | Random Forest | ~96.5% | ~96.5% |
| Gradient Boosting | Gradient Boosting | ~95.0% | ~95.0% |
| Support Vector Machine | SVM | ~94.0% | ~94.0% |
| Logistic Regression | Logistic Regression | ~92.5% | ~92.5% |

---

## 📊 الخصائص المستخدمة | Features (7 Inputs)

1. **متوسط نصف القطر** | Radius Mean (6.0 – 30.0)
2. **متوسط الملمس** | Texture Mean (9.0 – 40.0)
3. **متوسط النعومة** | Smoothness Mean (0.05 – 0.17)
4. **متوسط التراص** | Compactness Mean (0.02 – 0.35)
5. **متوسط التقعر** | Concavity Mean (0.0 – 0.45)
6. **متوسط التناسق** | Symmetry Mean (0.10 – 0.30)
7. **متوسط البعد الكسوري** | Fractal Dimension Mean (0.05 – 0.10)

---

## 🗂 هيكل المشروع | Project Structure

```
tumor_detection/
├── app.py                  # Flask API
├── requirements.txt        # Dependencies
├── README.md
├── models/
│   ├── __init__.py
│   └── tumor_model.py      # ML/DL Models
└── templates/
    └── index.html          # Frontend UI
```

---

## ⚠️ تنبيه | Disclaimer

هذا النظام للأغراض التعليمية والبحثية فقط ولا يُغني عن التشخيص الطبي المتخصص.

> For educational & research purposes only. Does not replace professional medical diagnosis.
