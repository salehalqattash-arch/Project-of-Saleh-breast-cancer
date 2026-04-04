"""
Flask API - Tumor Detection System
نظام الكشف عن سرطان الثدي - واجهة برمجية
مصادر الداتا: Breast Cancer Wisconsin + Synthetic + CSV Upload
"""

from flask import Flask, request, jsonify, render_template, Response
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from models.tumor_model import (get_system, reset_system,
                                FEATURE_NAMES, FEATURE_LABELS_AR, FEATURE_RANGES,
                                load_csv_dataset, TumorDetectionSystem)
# لفتح ومعالجة الصور
from PIL import Image
# لتحويل الصور الى ارقام
import numpy as np
# لتشغيل نموذج CNN
import tensorflow as tf


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# إعدادات رفع الصور 
UPLOAD_FOLDER = 'uploads'

ich = 12

ALLOWED_EXTENSIONS = {'png', 'jpg'}



ALLOWED_EXTENSIONS = {'png', 'jpg'}


# نخبر Flask بمكان حفظ الملفات
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# إنشاء مجلد uploads إذا لم يكن موجوداً
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f" uploading folder  : {UPLOAD_FOLDER}")

# التحقق من امتداد الملف
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html',
        features=list(zip(FEATURE_NAMES, FEATURE_LABELS_AR,
                        [FEATURE_RANGES[f] for f in FEATURE_NAMES])))

# متغير عام لتخزين النموذج
_image_detector = None

# تحميل نموذج الصور 
def get_image_detector():
    global _image_detector
    if _image_detector is None:
        try:
            # the main of cnn model
            from models.cnn_model import TumorImageDetector
            _image_detector = TumorImageDetector()
            # مسار ملف النموذج المدرب
            model_path = "models/tumor_image_model.h5"
            
            if os.path.exists(model_path):
                # نحمل النموذج من الملف
                _image_detector.model = tf.keras.models.load_model(model_path)
                _image_detector.is_trained = True
                print(" تم تحميل نموذج الصور المدرب")
            else:
                print(" لم يتم العثور على نموذج الصور المدرب")
                print("   قم بتشغيل train_cnn.py أولاً")
        except ImportError:
            print(" لم يتم استيراد cnn_model.py")
            _image_detector = None
        except Exception as e:
            print(f" خطأ في تحميل نموذج الصور: {e}")
            _image_detector = None
    return _image_detector

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        values = [float(data[f]) for f in FEATURE_NAMES]
        result = get_system().predict(values)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/metrics', methods=['GET'])
def metrics():
    try:
        sys_ = get_system()
        return jsonify({"success": True, **sys_.get_metrics()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/data_info', methods=['GET'])
def data_info():
    return jsonify({"success": True, "data_stats": get_system().data_stats})

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        f = request.files['file']
        if not f.filename.endswith('.csv'):
            return jsonify({"success": False, "error": "Only CSV files accepted"}), 400
        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        f.save(tmp.name); tmp.close()
        try:
            X_csv, y_csv, _ = load_csv_dataset(tmp.name)
            n_rows = len(X_csv)
        except Exception as e:
            os.unlink(tmp.name)
            return jsonify({"success": False, "error": f"CSV format error: {str(e)}"}), 400
        reset_system()
        import models.tumor_model as tm
        tm._system = TumorDetectionSystem()
        tm._system.train(csv_path=tmp.name)
        os.unlink(tmp.name)
        s = tm._system
        return jsonify({
            "success": True,
            "message": "تم إعادة التدريب بنجاح | Retrained successfully",
            "csv_rows": n_rows,
            "data_stats": s.data_stats,
            "best_model": s.best_model_name,
            "metrics": {k: {"accuracy": v["accuracy"], "auc": v["auc"]} for k,v in s.metrics.items()}
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/sample_csv')
def sample_csv():
    header = ','.join(FEATURE_NAMES + ['diagnosis'])
    rows = ["14.5,19.2,0.095,0.100,0.085,0.180,0.062,1",
            "11.2,15.8,0.082,0.065,0.028,0.162,0.058,0",
            "17.8,22.1,0.108,0.155,0.175,0.210,0.065,1",
            "12.1,17.4,0.088,0.078,0.042,0.170,0.061,0"]
    content = header + '\n' + '\n'.join(rows) + '\n'
    return Response(content, mimetype='text/csv',
                    headers={"Content-Disposition":"attachment;filename=tumor_sample_template.csv"})

@app.route('/api/feature_info')
def feature_info():
    info = [{"name":n,"label_ar":a,"min":FEATURE_RANGES[n][0],"max":FEATURE_RANGES[n][1]}
            for n,a in zip(FEATURE_NAMES, FEATURE_LABELS_AR)]
    return jsonify(info)

# تشخيص صورة مرفوعة من المستخدم
@app.route('/api/predict_image', methods=['POST'])
def predict_image():
    filepath = None
    try:
        # التحقق من وجود صورة
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400
        file = request.files['image']
        
        # التحقق من اسم الملف
        if file.filename == '':
            return jsonify({"success": False, "error": "No image selected"}), 400
        
        # التحقق من امتداد الملف
        if not allowed_file(file.filename):
            return jsonify({
                "success": False, 
                "error": "Only image files accepted (png, jpg, jpeg, bmp, tiff)"
            }), 400
        
        # حفظ الصورة في مجلد uploads
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f" تم حفظ الصورة: {filepath}")
        
        # تحميل نموذج الصور
        detector = get_image_detector()
        if detector is None:
            os.remove(filepath)
            return jsonify({
                "success": False,
                "error": "Image model not available. Please train the model first."
            }), 500
        if not detector.is_trained:
            os.remove(filepath)
            return jsonify({
                "success": False,
                "error": "Image model not trained yet. Run train_cnn.py first."
            }), 500
        
        # تشخيص الصورة
        result = detector.predict(filepath)
        
        #حذف الصورة المؤقتة
        os.remove(filepath)
        print(f" تم حذف الصورة المؤقتة: {filepath}")
        # إرجاع النتيجة
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        # حذف الملف المؤقت في حالة الخطأ
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f" تم حذف الصورة المؤقتة : {filepath}")
            except:
                pass
        return jsonify({
            "success": False, 
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 50)
    print(" نظام الكشف عن سرطان الثدي ")
    print("=" * 50)
    
    print("\ تحميل وتدريب النماذج الرقمية...")
    s = get_system()
    print(f"{s.data_stats}")
    print(f"أفضل نموذج: {s.best_model_name}")
    print("نموذج الصور جاهز عند وجود ملف التدريب")
    
    print("\nتشغيل الخادم...")
    print("   http://127.0.0.1:5000")
    print("=" * 50)
    
    app.run(debug=True, port=5000)
