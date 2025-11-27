"""
Flask REST API for fruit classification
Handles predictions, metrics, and retraining
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
import threading
import pickle
import sys
sys.path.append(os.path.dirname(__file__))

from prediction import Predictor
from retraining import Retrainer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize predictor and retrainer
try:
    predictor = Predictor(
        'models/fruit_classifier_light.h5',
        'models/class_names.pkl'
    )
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

try:
    retrainer = Retrainer(
        'models/fruit_classifier_light.h5',
        'models/class_names.pkl'
    )
except Exception as e:
    print(f"Error loading retrainer: {e}")
    retrainer = None

# Metrics tracking
class MetricsTracker:
    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0
        self.start_time = time.time()
        self.retraining = False
    
    def add_request(self, response_time):
        self.request_count += 1
        self.total_response_time += response_time
    
    def get_metrics(self):
        uptime = time.time() - self.start_time
        avg_response_time = (self.total_response_time / self.request_count * 1000) if self.request_count > 0 else 0
        requests_per_minute = (self.request_count / (uptime / 60)) if uptime > 0 else 0
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'average_response_time_ms': avg_response_time,
            'requests_per_minute': requests_per_minute,
            'retraining': self.retraining
        }

metrics = MetricsTracker()

def allowed_file(filename, types={'jpg', 'jpeg', 'png', 'zip'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in types

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': time.time()
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Get API status"""
    return jsonify({
        'status': 'running',
        'model': 'fruit_classifier',
        'classes': predictor.get_classes() if predictor else [],
        'timestamp': time.time()
    }), 200

# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction on uploaded image
    """
    start_time = time.time()
    
    try:
        print("🔍 Starting prediction request...")
        
        if predictor is None:
            print("❌ Predictor is None - model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'image' not in request.files:
            print("❌ No 'image' key in request.files")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        print(f"📁 Received file: {file.filename}")
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename, {'jpg', 'jpeg', 'png'}):
            print(f"❌ Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Use jpg, jpeg, or png'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"💾 Saving file to: {filepath}")
        
        file.save(filepath)
        print("✅ File saved successfully")
        
        # Make prediction
        print("🤖 Making prediction...")
        result = predictor.predict(filepath)
        print(f"✅ Prediction result: {result}")
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            print("✅ Temporary file cleaned up")
        
        # Track metrics
        response_time = time.time() - start_time
        metrics.add_request(response_time)
        
        print(f"✅ Prediction completed in {response_time:.2f}s")
        
        return jsonify({
            'prediction': result,
            'response_time_ms': response_time * 1000,
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        print(f"❌ ERROR in /predict: {str(e)}")
        import traceback
        print("🔍 Full traceback:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Make predictions on multiple images
    
    Expected: Multiple files with key 'images'
    Returns: List of predictions
    """
    start_time = time.time()
    
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    if len(files) == 0:
        return jsonify({'error': 'No images selected'}), 400
    
    try:
        results = []
        filepaths = []
        
        for file in files:
            if file and allowed_file(file.filename, {'jpg', 'jpeg', 'png'}):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                filepaths.append(filepath)
                
                result = predictor.predict(filepath)
                results.append({
                    'filename': filename,
                    'prediction': result
                })
        
        # Clean up
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        response_time = time.time() - start_time
        metrics.add_request(response_time)
        
        return jsonify({
            'predictions': results,
            'count': len(results),
            'response_time_ms': response_time * 1000,
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get API metrics"""
    return jsonify(metrics.get_metrics()), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'classes': predictor.get_classes(),
        'num_classes': predictor.get_num_classes(),
        'model_file': 'models/fruit_classifier_light.h5',
        'input_size': [150, 150, 3]
    }), 200

# ============================================================================
# RETRAINING ENDPOINTS
# ============================================================================

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Trigger model retraining
    
    Expected: Zip file with key 'file' containing class subdirectories
    Returns: Success/failure status
    """
    if retrainer is None:
        return jsonify({'error': 'Retrainer not initialized'}), 500
    
    if metrics.retraining:
        return jsonify({'error': 'Retraining already in progress'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename, {'zip'}):
        return jsonify({'error': 'Invalid file type. Use zip'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Mark as retraining
        metrics.retraining = True
        
        # Run retraining in background thread
        def retrain_background():
            try:
                success = retrainer.retrain_from_zip(
                    filepath,
                    epochs=20,
                    batch_size=32,
                    cleanup=True
                )
                
                if success:
                    # Reload predictor with new model
                    global predictor
                    predictor = Predictor(
                        'models/fruit_classifier_light.h5',
                        'models/class_names.pkl'
                    )
                    print("✅ Predictor reloaded with new model")
                
                # Clean up zip file
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
            finally:
                metrics.retraining = False
        
        thread = threading.Thread(target=retrain_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Retraining started in background',
            'status': 'processing'
        }), 202
        
    except Exception as e:
        metrics.retraining = False
        return jsonify({'error': str(e)}), 500

@app.route('/retrain-status', methods=['GET'])
def retrain_status():
    """Get retraining status"""
    return jsonify({
        'retraining': metrics.retraining,
        'timestamp': time.time()
    }), 200

# ============================================================================
# DATA ENDPOINTS
# ============================================================================

@app.route('/upload-data', methods=['POST'])
def upload_data():
    """
    Upload training data (images or zip)
    
    Expected: File or zip with images
    Returns: Upload status
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename, {'jpg', 'jpeg', 'png', 'zip'}):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'filepath': filepath,
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# INFO ENDPOINTS
# ============================================================================

@app.route('/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        'name': 'Fruit Classification API',
        'version': '1.0',
        'description': 'REST API for fruit image classification',
        'endpoints': {
            'POST /predict': 'Make prediction on single image',
            'POST /predict-batch': 'Make predictions on multiple images',
            'POST /retrain': 'Trigger model retraining',
            'POST /upload-data': 'Upload training data',
            'GET /metrics': 'Get API metrics',
            'GET /model-info': 'Get model information',
            'GET /health': 'Health check',
            'GET /status': 'API status'
        }
    }), 200

@app.route('/', methods=['GET'])
def index():
    """API home page"""
    return jsonify({
        'message': 'Fruit Classification API',
        'status': 'running',
        'visit': '/info for endpoints'
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Fruit Classification API")
    print("=" * 60)
    print(f"Model loaded: {predictor is not None}")
    print(f"Retrainer ready: {retrainer is not None}")
    print("=" * 60)
    print("\n📊 Starting server on http://0.0.0.0:5000\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )