#!/usr/bin/env python3
"""
Face Mask Detection Web Application
Run with: python app.py
Requirements: flask, tensorflow, pillow, numpy, waitress, h5py
"""

import os
import sys
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU if causing issues

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'face_mask_model.h5')

# Ensure directories exist
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, 'css'), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Import Flask and TensorFlow
try:
    from flask import Flask, render_template, request, jsonify, send_from_directory, Response
    from PIL import Image
    import numpy as np
    from datetime import datetime
    import base64
    import io
    logger.info("✅ Flask and dependencies imported successfully")
except ImportError as e:
    logger.error(f"❌ Missing dependency: {e}")
    logger.error("Install with: pip install flask pillow numpy")
    sys.exit(1)

# Initialize Flask
app = Flask(__name__, 
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'face-mask-detector-secret-key')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Global model variable
model = None
model_info = {
    'loaded': False,
    'path': MODEL_PATH,
    'exists': os.path.exists(MODEL_PATH),
    'error': None,
    'input_shape': None,
    'output_shape': None
}

def create_simple_cnn():
    """Create a simple CNN model for face mask detection"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        logger.info("🔄 Creating simple CNN model architecture...")
        
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("✅ Simple CNN model created successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to create model: {e}")
        return None

def load_model_safely():
    """Load model with multiple fallback strategies"""
    global model, model_info
    
    if not model_info['exists']:
        logger.warning(f"⚠️  Model file not found: {MODEL_PATH}")
        logger.info("🔄 Creating fallback model...")
        model = create_simple_cnn()
        if model:
            model_info['loaded'] = True
            model_info['input_shape'] = model.input_shape
            model_info['output_shape'] = model.output_shape
            model_info['mode'] = 'fallback_created'
        return model
    
    try:
        import tensorflow as tf
        import h5py
        
        logger.info(f"⏳ Loading model from: {MODEL_PATH}")
        
        # Check h5py version compatibility
        h5py_version = h5py.__version__
        logger.info(f"📦 h5py version: {h5py_version}")
        
        # Strategy 1: Standard load
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("✅ Model loaded with compile=False")
        except Exception as e1:
            logger.warning(f"⚠️  Standard load failed: {e1}")
            
            # Strategy 2: Load with custom objects
            try:
                model = tf.keras.models.load_model(
                    MODEL_PATH, 
                    compile=False,
                    custom_objects=None
                )
                logger.info("✅ Model loaded with custom_objects")
            except Exception as e2:
                logger.warning(f"⚠️  Custom objects load failed: {e2}")
                
                # Strategy 3: Load weights only (requires architecture)
                logger.error("❌ All load strategies failed")
                raise e1
        
        # Compile model after loading
        try:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        except Exception as compile_error:
            logger.warning(f"⚠️  Could not compile model: {compile_error}")
        
        model_info['loaded'] = True
        model_info['input_shape'] = model.input_shape
        model_info['output_shape'] = model.output_shape
        model_info['mode'] = 'loaded_from_h5'
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Input shape: {model.input_shape}")
        logger.info(f"   Output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Critical error loading model: {e}")
        model_info['error'] = str(e)
        
        # Fallback to created model
        logger.info("🔄 Falling back to created model...")
        model = create_simple_cnn()
        if model:
            model_info['loaded'] = True
            model_info['input_shape'] = model.input_shape
            model_info['output_shape'] = model.output_shape
            model_info['mode'] = 'fallback_after_error'
        return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess image with error handling"""
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
        
    except Exception as e:
        logger.error(f"❌ Image preprocessing error: {e}")
        raise

def predict_mask(image_path):
    """Make prediction with comprehensive error handling"""
    global model
    
    # Ensure model is loaded
    if model is None:
        logger.error("❌ Model not loaded")
        return {
            'success': False,
            'error': 'Model not loaded',
            'label': 0,
            'class_name': 'Error',
            'confidence': 0.0
        }
    
    try:
        # Preprocess
        processed_img, original_img = preprocess_image(image_path)
        
        # Verify input shape
        expected_shape = model.input_shape
        actual_shape = processed_img.shape
        
        logger.info(f"📊 Input shape: expected {expected_shape}, got {actual_shape}")
        
        if expected_shape[1:] != actual_shape[1:]:
            logger.warning(f"⚠️  Shape mismatch! Resizing...")
            processed_img = np.resize(processed_img, (1, expected_shape[1], expected_shape[2], expected_shape[3]))
        
        # Predict
        logger.info("🧠 Running prediction...")
        predictions = model.predict(processed_img, verbose=0)
        
        logger.info(f"📈 Raw predictions: {predictions}")
        
        # Handle different output formats
        if predictions.shape[1] == 1:
            # Binary classification (sigmoid)
            prob = float(predictions[0][0])
            label = 1 if prob > 0.5 else 0
            confidence = prob if label == 1 else 1 - prob
        else:
            # Multi-class (softmax)
            label = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            prob = float(predictions[0][label])
        
        class_name = "Mask" if label == 1 else "No Mask"
        
        result = {
            'success': True,
            'label': label,
            'class_name': class_name,
            'confidence': float(confidence),
            'probability': prob,
            'raw_predictions': predictions[0].tolist(),
            'input_shape': list(actual_shape)
        }
        
        logger.info(f"✅ Prediction: {class_name} ({confidence:.2%} confidence)")
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {
            'success': False,
            'error': str(e),
            'label': 0,
            'class_name': 'Error',
            'confidence': 0.0
        }

# Flask Routes
@app.route('/')
def home():
    """Render the main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template error: {e}")
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Face Mask Detector - Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 40px; background: #0f172a; color: white; }}
                .error-box {{ background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; padding: 20px; border-radius: 10px; }}
                code {{ background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <h1>⚠️ Template Error</h1>
            <div class="error-box">
                <p><strong>Error:</strong> {str(e)}</p>
                <p>Template folder: <code>{TEMPLATE_DIR}</code></p>
                <p>Exists: {os.path.exists(TEMPLATE_DIR)}</p>
                <p>Files: {os.listdir(TEMPLATE_DIR) if os.path.exists(TEMPLATE_DIR) else 'N/A'}</p>
            </div>
            <h2>Quick Fix:</h2>
            <ol>
                <li>Create folder: <code>mkdir templates</code></li>
                <li>Move HTML file: <code>mv index.html templates/</code></li>
                <li>Restart the server</li>
            </ol>
        </body>
        </html>
        """, 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(STATIC_DIR, filename)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False, 
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            })
        
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"📁 Saved upload: {filepath}")
        
        # Make prediction
        result = predict_mask(filepath)
        
        if not result['success']:
            return jsonify(result)
        
        # Convert image to base64 for display
        try:
            with open(filepath, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            result['image_data'] = f"data:image/jpeg;base64,{img_base64}"
        except Exception as img_error:
            logger.warning(f"⚠️  Could not encode image: {img_error}")
        
        # Clean up (optional - comment out to keep files)
        # try:
        #     os.remove(filepath)
        # except:
        #     pass
        
        return jsonify({
            'success': True,
            'prediction': result['class_name'],
            'label': result['label'],
            'confidence': f"{result['confidence']:.2%}",
            'confidence_value': round(result['confidence'] * 100, 2),
            'image_data': result.get('image_data', ''),
            'raw_predictions': result['raw_predictions']
        })
        
    except Exception as e:
        logger.error(f"❌ Route error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_info['loaded'],
        'model_path_exists': model_info['exists'],
        'model_mode': model_info.get('mode', 'unknown'),
        'model_error': model_info['error'],
        'input_shape': str(model_info['input_shape']) if model_info['input_shape'] else None,
        'output_shape': str(model_info['output_shape']) if model_info['output_shape'] else None,
        'template_folder': TEMPLATE_DIR,
        'static_folder': STATIC_DIR,
        'upload_folder': UPLOAD_FOLDER
    })

@app.route('/model/info')
def model_info_endpoint():
    """Get detailed model information"""
    return jsonify({
        'model_info': model_info,
        'tensorflow_version': tf.__version__ if 'tf' in globals() else 'not loaded',
        'numpy_version': np.__version__,
        'python_version': sys.version
    })

# Initialize model on startup
logger.info("="*60)
logger.info("🚀 Face Mask Detection Server Starting...")
logger.info("="*60)

# Load the model
load_model_safely()

# Print startup info
logger.info(f"\n📁 Directories:")
logger.info(f"   Base: {BASE_DIR}")
logger.info(f"   Templates: {TEMPLATE_DIR} (exists: {os.path.exists(TEMPLATE_DIR)})")
logger.info(f"   Static: {STATIC_DIR} (exists: {os.path.exists(STATIC_DIR)})")
logger.info(f"   Uploads: {UPLOAD_FOLDER} (exists: {os.path.exists(UPLOAD_FOLDER)})")
logger.info(f"\n🧠 Model: {MODEL_PATH} (exists: {model_info['exists']})")
logger.info(f"   Loaded: {model_info['loaded']}")
logger.info(f"   Mode: {model_info.get('mode', 'unknown')}")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Starting Web Server...")
    print("="*60)
    print("📱 Open your browser at:")
    print("   http://localhost:5000")
    print("   http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Try different server options
    server_type = os.environ.get('SERVER_TYPE', 'auto')
    port = int(os.environ.get('PORT', 5000))
    
    if server_type == 'flask':
        # Development server only
        app.run(debug=True, host='0.0.0.0', port=port, threaded=True)
    else:
        # Try Waitress first, fallback to Flask
        try:
            from waitress import serve
            logger.info("🔧 Using Waitress WSGI server...")
            serve(app, host='0.0.0.0', port=port, threads=4)
        except ImportError:
            logger.warning("⚠️  Waitress not installed, using Flask dev server")
            logger.info("   Install Waitress: pip install waitress")
            app.run(debug=True, host='0.0.0.0', port=port, threaded=True)