from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import random
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def simulate_prediction(filename):
    """Simulate AI prediction based on filename or random choice"""
    # Add some processing delay to simulate model inference
    time.sleep(1)
    
    # Simple simulation logic based on filename keywords
    filename_lower = filename.lower()
    
    if any(keyword in filename_lower for keyword in ['flood', 'water', 'river', 'lake', 'ocean']):
        # Bias towards flood detection for water-related filenames
        is_flood = random.choices([True, False], weights=[0.8, 0.2])[0]
    elif any(keyword in filename_lower for keyword in ['dry', 'desert', 'mountain', 'forest']):
        # Bias towards normal for dry/land-related filenames
        is_flood = random.choices([True, False], weights=[0.2, 0.8])[0]
    else:
        # Random prediction for other images
        is_flood = random.choice([True, False])
    
    if is_flood:
        label = "Flooded"
        confidence = random.uniform(0.75, 0.98)  # High confidence for flood detection
        risk_level = "High" if confidence > 0.9 else "Medium"
        color_class = "danger" if confidence > 0.9 else "warning"
    else:
        label = "Normal"
        confidence = random.uniform(0.65, 0.95)  # Varying confidence for normal
        risk_level = "Low"
        color_class = "success"
    
    return {
        'label': label,
        'confidence': confidence,
        'risk_level': risk_level,
        'color_class': color_class
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                # Ensure upload directory exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)
                
                # Simulate prediction
                prediction_result = simulate_prediction(filename)
                
                return render_template('results.html', 
                                     filename=filename,
                                     filepath=filepath,
                                     **prediction_result)
            except Exception as e:
                print(f"File handling error: {str(e)}")
                return render_template('index.html', error="Error processing image. Please try again.")
        else:
            return render_template('index.html', error="Invalid file type. Please upload an image file.")
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Flood Detection Simulation Server...")
    print("Upload any image to see simulated AI flood detection results!")
    app.run(host='0.0.0.0', port=8000, debug=True)