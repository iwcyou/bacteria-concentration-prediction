from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

# 引入你之前的推理函数
from predict_single_image import predict_single_image, ask_deepseek

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 加载模型
weights_path = "weights/best_model_epoch_7_val_acc_1.0000.pth"

# 检查文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    print("收到一张图片！")  # 这里会在终端输出
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选中文件'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # 调用预测逻辑
        prompt, suggestion = predict_single_image(filepath, weights_path)

        # 返回结果
        result = {
            "concentration": prompt.split("为【")[1].split("】")[0],
            "suggestion": suggestion
        }
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
