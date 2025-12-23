from flask import Flask, jsonify, redirect, request, render_template, url_for
import os  
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications.vgg16 import preprocess_input,VGG16 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Flatten
import numpy as np 

app = Flask(__name__)   
app.config['UPLOAD_FOLDER'] = 'static/uploads'  
app.config["template_folder"] = "templates"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) 


#download the model from drive
import gdown 
list_of_file =  os.listdir('model')

if "model_vgg16.h5" in list_of_file:
    print("File exists")
else:
    file_id = "1xtFkEqCABwxjDpgpmKXeAA1lE8fGNZgG"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "model/model_vgg16.h5"  
    gdown.download(url, output, quiet=False)


# model path 
MODEL_PATH = 'model/model_vgg16.h5'  




def build_model(): 
    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))  
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(2, activation='softmax')
    ])
    return model

model = build_model() 
model.load_weights(MODEL_PATH) 
print("Model loaded successfully.") 

classes_names = [
    "Oblique Fracture",
    "Spiral Fracture"
] 

def allowed_file(filename): 
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 


# now base route for get and post both 
@app.route('/', methods=['GET', 'POST']) 
def index():
    if request.method == 'POST': 
        if 'file' not in request.files: 
            return jsonify({'error': 'No file part in the request'}), 400 
        file = request.files['file'] 
        if file.filename == '': 
            return jsonify({'error': 'No selected file'}), 400 
        if file and allowed_file(file.filename): 
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename) 
            file.save(filepath) 
            
            # preprocess
            img = load_img(filepath, target_size=(224, 224)) 
            img_array = img_to_array(img) 
            img_array = np.expand_dims(img_array, axis=0) 
            img_array = preprocess_input(img_array) 
            
            # prediction
            preds = model.predict(img_array)
            idx = np.argmax(preds[0])
            label = classes_names[idx]
            confidence = preds[0][idx]


            # return on result with filename,label,confidence 
            return render_template('index.html', filename=file.filename, label=label,
                                   confidence=f"{confidence*100:.2f}%") 
        else: 
            return redirect(request.url) 
    return render_template('index.html') 


# upload route 
@app.route('/uploads/<filename>') 
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))


if __name__ == '__main__': 
    app.run(host="0.0.0.0", port=5000, debug=True)