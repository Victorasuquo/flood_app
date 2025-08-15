from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
import cv2

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/VGG16_Transfer_best.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

def build_model():
    """Build the VGG16 transfer learning model architecture"""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    
    return model

def load_model_with_fallback(model_path):
    """Attempt to load model with fallback to architecture rebuild"""
    try:
        # First try loading the complete model
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully with architecture and weights")
        return model
    except Exception as e:
        print(f"Full model loading failed: {str(e)}")
        print("Attempting to rebuild architecture and load weights...")
        
        try:
            # Build the model architecture
            model = build_model()
            
            # Try loading just the weights
            model.load_weights(model_path)
            print("Successfully rebuilt architecture and loaded weights")
            return model
        except Exception as e:
            print(f"Weight loading failed: {str(e)}")
            raise RuntimeError("Failed to load model both as complete model and as weights")

# Load the model
try:
    model = load_model_with_fallback(MODEL_PATH)
    # Compile the model (important if loading weights only)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    print("Model is ready for predictions")
except Exception as e:
    print(f"Critical error loading model: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess image for VGG16 model"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(img_path):
    """Make prediction on the input image"""
    try:
        processed_img = preprocess_image(img_path)
        prediction = model.predict(processed_img)
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][class_idx])
        
        if class_idx == 0:
            label = "Flooded"
        else:
            label = "Normal"
        
        return label, confidence
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error", 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                label, confidence = predict_image(filepath)
                
                return render_template('results.html', 
                                     filename=filename, 
                                     label=label, 
                                     confidence=f"{confidence*100:.2f}%")
            except Exception as e:
                print(f"File handling error: {str(e)}")
                return render_template('index.html', error="Error processing image")
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=8000, debug=True)