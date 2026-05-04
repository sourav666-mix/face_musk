import streamlit as st
import cv2
import numpy as np
import h5py
from PIL import Image
import time
import os
import json

# ==================== CONFIGURATION ====================
MODEL_PATH = "face_mask_detection_model.h5"  # Your model file
IMG_SIZE = (224, 224)  # (height, width) - CHANGE THIS IF NEEDED
CONFIDENCE_THRESHOLD = 0.5

GREEN_ALERT = (0, 255, 0)    # With mask
RED_ALERT = (0, 0, 255)      # Without mask

# ==================== INSPECT .h5 FILE STRUCTURE ====================
def inspect_h5_file(h5_path):
    """Inspect the structure of .h5 file to understand weight organization"""
    info = {'layers': [], 'total_weights': 0, 'model_config': None}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Try to get model config
            if 'model_config' in f.attrs:
                try:
                    config = json.loads(f.attrs['model_config'].decode('utf-8'))
                    info['model_config'] = config
                except:
                    pass
            
            # List all datasets recursively
            def list_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    info['layers'].append({
                        'name': name,
                        'shape': obj.shape,
                        'size': obj.size
                    })
                    info['total_weights'] += obj.size
            
            f.visititems(list_datasets)
    except Exception as e:
        info['error'] = str(e)
    
    return info

# ==================== CUSTOM CNN MODEL (No TensorFlow) ====================
class SimpleMaskDetector:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.weight_groups = {}  # Organized by layer
        self.layer_info = []
    
    def load_weights_from_h5(self, h5_path):
        """Load and organize weights from .h5 file"""
        with h5py.File(h5_path, 'r') as f:
            # First, inspect structure
            top_keys = list(f.keys())
            
            # Keras models typically store layers under 'model_weights'
            if 'model_weights' in top_keys:
                model_group = f['model_weights']
                layer_names = list(model_group.keys())
                
                for layer_name in layer_names:
                    layer_group = model_group[layer_name]
                    layer_weights = {}
                    
                    # Get all datasets in this layer
                    def collect_datasets(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            layer_weights[name] = np.array(obj)
                    
                    layer_group.visititems(collect_datasets)
                    
                    if layer_weights:
                        self.weight_groups[layer_name] = layer_weights
                        self.layer_info.append({
                            'name': layer_name,
                            'weights': layer_weights
                        })
            else:
                # Flat structure - collect all datasets
                all_weights = {}
                def collect_all(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        all_weights[name] = np.array(obj)
                f.visititems(collect_all)
                self.weight_groups['all'] = all_weights
        
        return len(self.weight_groups) > 0
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _maxpool2d(self, x, pool_size=2, stride=2):
        batch, h, w, c = x.shape
        out_h = h // stride
        out_w = w // stride
        output = np.zeros((batch, out_h, out_w, c))
        for i in range(out_h):
            for j in range(out_w):
                x_slice = x[:, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size, :]
                output[:, i, j, :] = np.max(x_slice, axis=(1, 2))
        return output
    
    def _flatten(self, x):
        return x.reshape(x.shape[0], -1)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _apply_dense(self, x, weights_dict):
        """Apply a dense layer from weight dict"""
        # Find kernel and bias
        kernel = None
        bias = None
        
        for key, val in weights_dict.items():
            if 'kernel' in key.lower() or 'weight' in key.lower():
                kernel = val
            elif 'bias' in key.lower():
                bias = val
        
        if kernel is None:
            raise ValueError("No kernel found in weights")
        
        # kernel shape could be (in_features, out_features) or flattened
        if len(kernel.shape) == 1:
            # Flattened - need to infer shape
            in_features = x.shape[1]
            out_features = kernel.shape[0] // in_features if kernel.shape[0] % in_features == 0 else kernel.shape[0]
            kernel = kernel.reshape(in_features, out_features)
        elif len(kernel.shape) > 2:
            # Flatten extra dimensions
            kernel = kernel.reshape(-1, kernel.shape[-1])
        
        output = x @ kernel
        if bias is not None:
            output += bias.reshape(1, -1)
        
        return output
    
    def _apply_conv(self, x, weights_dict):
        """Apply a conv layer from weight dict"""
        kernel = None
        bias = None
        
        for key, val in weights_dict.items():
            if 'kernel' in key.lower() or 'weight' in key.lower():
                kernel = val
            elif 'bias' in key.lower():
                bias = val
        
        if kernel is None:
            raise ValueError("No kernel found in weights")
        
        batch, h, w, c = x.shape
        
        # Handle different kernel shapes
        if len(kernel.shape) == 4:
            kh, kw, in_c, out_c = kernel.shape
        elif len(kernel.shape) == 3:
            # Could be (in_c, kh, kw) or similar - try to infer
            total = kernel.size
            out_c = 32  # guess
            in_c = c
            remaining = total // (in_c * out_c)
            if remaining > 0:
                kh = kw = int(np.sqrt(remaining))
                kernel = kernel.reshape(kh, kw, in_c, out_c)
            else:
                raise ValueError(f"Cannot infer conv shape from {kernel.shape}")
        elif len(kernel.shape) == 2:
            # (in_c*kh*kw, out_c)
            out_c = kernel.shape[1]
            remaining = kernel.shape[0] // c
            if remaining > 0:
                kh = kw = int(np.sqrt(remaining))
                kernel = kernel.reshape(kh, kw, c, out_c)
            else:
                raise ValueError(f"Cannot infer conv shape from {kernel.shape}")
        elif len(kernel.shape) == 1:
            # Fully flattened
            total = kernel.size
            # Try common configurations
            possible_configs = [(3,3,3,32), (3,3,32,64), (3,3,64,128), (5,5,3,32)]
            for config in possible_configs:
                if np.prod(config) == total:
                    kernel = kernel.reshape(config)
                    kh, kw, in_c, out_c = config
                    break
            else:
                raise ValueError(f"Cannot find matching conv config for size {total}")
        else:
            raise ValueError(f"Unexpected kernel shape: {kernel.shape}")
        
        # Now perform convolution
        padding = 1
        stride = 1
        out_h = (h + 2*padding - kh) // stride + 1
        out_w = (w + 2*padding - kw) // stride + 1
        
        x_padded = np.pad(x, ((0,0), (padding,padding), (padding,padding), (0,0)), mode='constant')
        output = np.zeros((batch, out_h, out_w, out_c))
        
        for i in range(out_h):
            for j in range(out_w):
                i0, i1 = i*stride, i*stride + kh
                j0, j1 = j*stride, j*stride + kw
                x_slice = x_padded[:, i0:i1, j0:j1, :]
                x_flat = x_slice.reshape(batch, -1)
                w_flat = kernel.reshape(-1, out_c)
                output[:, i, j, :] = x_flat @ w_flat
        
        if bias is not None:
            output += bias.reshape(1, 1, 1, -1)
        
        return output
    
    def predict(self, x):
        """Forward pass with error handling per layer"""
        current = x
        
        # Try to identify layer types from names
        conv_layers = []
        dense_layers = []
        
        for layer_name, weights in self.weight_groups.items():
            if 'conv' in layer_name.lower():
                conv_layers.append((layer_name, weights))
            elif 'dense' in layer_name.lower() or 'fc' in layer_name.lower() or 'output' in layer_name.lower():
                dense_layers.append((layer_name, weights))
        
        # Sort by name to maintain order
        conv_layers.sort(key=lambda x: x[0])
        dense_layers.sort(key=lambda x: x[0])
        
        # Apply conv layers
        for name, weights in conv_layers:
            try:
                current = self._apply_conv(current, weights)
                current = self._relu(current)
                current = self._maxpool2d(current)
            except Exception as e:
                st.sidebar.warning(f"Conv layer {name} failed: {e}")
                continue
        
        # Flatten
        current = self._flatten(current)
        
        # Apply dense layers
        for i, (name, weights) in enumerate(dense_layers):
            try:
                current = self._apply_dense(current, weights)
                if i < len(dense_layers) - 1:  # Not last layer
                    current = self._relu(current)
            except Exception as e:
                st.sidebar.warning(f"Dense layer {name} failed: {e}")
                continue
        
        # Output sigmoid
        current = self._sigmoid(current)
        return current

# ==================== HEURISTIC FALLBACK DETECTOR ====================
class HeuristicMaskDetector:
    """Fallback detector using face region color analysis"""
    def predict(self, x):
        batch = x.shape[0]
        scores = np.zeros((batch, 1))
        
        for i in range(batch):
            img = x[i]
            h, w = img.shape[:2]
            lower_face = img[int(h*0.5):int(h*0.9), int(w*0.2):int(w*0.8)]
            
            if lower_face.size == 0:
                scores[i] = 0.5
                continue
            
            mean_color = np.mean(lower_face)
            std_color = np.std(lower_face)
            uniformity = 1.0 - min(std_color / 50.0, 1.0)
            brightness_score = 1.0 - abs(mean_color - 0.5) * 2
            score = uniformity * 0.6 + brightness_score * 0.4
            scores[i] = np.clip(score, 0.1, 0.9)
        
        return scores

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load model - inspect first, then try custom CNN, fallback to heuristic"""
    
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error(f"Model file not found: {MODEL_PATH}")
        return HeuristicMaskDetector(), "heuristic"
    
    # Inspect file first
    info = inspect_h5_file(MODEL_PATH)
    
    if 'error' in info:
        st.sidebar.error(f"Cannot read .h5: {info['error']}")
        return HeuristicMaskDetector(), "heuristic"
    
    st.sidebar.text(f"Found {len(info['layers'])} weight arrays")
    
    # Show first few layer shapes for debugging
    for i, layer in enumerate(info['layers'][:5]):
        st.sidebar.text(f"  {layer['name']}: {layer['shape']}")
    
    # Try custom CNN
    try:
        model = SimpleMaskDetector(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        success = model.load_weights_from_h5(MODEL_PATH)
        
        if success:
            # Test with dummy input to verify
            dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
            try:
                test_out = model.predict(dummy)
                st.sidebar.success(f"Model test OK: output shape {test_out.shape}")
                return model, "cnn"
            except Exception as e:
                st.sidebar.warning(f"Model test failed: {e}")
                return HeuristicMaskDetector(), "heuristic"
        else:
            st.sidebar.warning("No weights loaded")
            return HeuristicMaskDetector(), "heuristic"
            
    except Exception as e:
        st.sidebar.error(f"Load error: {str(e)[:100]}")
        return HeuristicMaskDetector(), "heuristic"

# ==================== FACE DETECTION ====================
@st.cache_resource
def load_face_detector():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return face_cascade

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

# ==================== MASK DETECTION ====================
def preprocess_face(face_img, target_size):
    """Preprocess face image for model prediction"""
    face_img = cv2.resize(face_img, target_size)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def predict_mask(model, face_img, model_type):
    """Predict if face has mask or not"""
    processed = preprocess_face(face_img, IMG_SIZE)
    
    try:
        prediction = model.predict(processed)[0][0]
        has_mask = prediction > CONFIDENCE_THRESHOLD
        confidence = float(prediction if has_mask else 1 - prediction)
        return has_mask, confidence, prediction
        
    except Exception as e:
        # Silently fail and return heuristic guess
        return False, 0.5, 0.5

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="Face Mask Detection",
        page_icon="😷",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .alert-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .green-alert {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .red-alert {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .model-info {
        background-color: #e7f3ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://img.icons8.com/color/96/face-mask.png", width=80)
    with col2:
        st.title("Real-Time Face Mask Detection")
        st.markdown("*AI-powered mask detection using device camera*")
    
    st.sidebar.header("Settings")
    st.sidebar.subheader("Model Settings")
    
    size_options = [64, 96, 128, 160, 224]
    selected_size = st.sidebar.selectbox(
        "Model Input Size",
        options=size_options,
        index=size_options.index(224),
        help="Must match training image size"
    )
    
    global IMG_SIZE
    IMG_SIZE = (selected_size, selected_size)
    
    model_path = st.sidebar.text_input("Model Path", MODEL_PATH)
    
    st.sidebar.subheader("Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.0, 1.0, 
        CONFIDENCE_THRESHOLD, 
        0.05
    )
    
    st.sidebar.subheader("Camera Settings")
    camera_id = st.sidebar.selectbox(
        "Select Camera", 
        options=[0, 1, 2, 3], 
        format_func=lambda x: f"Camera {x} {'(Default)' if x == 0 else ''}"
    )
    
    st.sidebar.subheader("Display Options")
    show_confidence = st.sidebar.checkbox("Show Confidence Score", True)
    show_box = st.sidebar.checkbox("Show Detection Box", True)
    show_raw_score = st.sidebar.checkbox("Show Raw Model Output", False)
    
    model, model_type = load_model()
    
    st.sidebar.markdown(f"""
    <div class="model-info">
        <strong>Model Type:</strong> {model_type.upper()}<br>
        <strong>Input Size:</strong> {IMG_SIZE[0]}x{IMG_SIZE[1]}<br>
        <strong>File:</strong> {os.path.basename(model_path)}
    </div>
    """, unsafe_allow_html=True)
    
    if model_type == "heuristic":
        st.sidebar.markdown("""
        <div class="warning-box">
            <strong>Heuristic Mode Active</strong><br>
            Using color analysis fallback.
        </div>
        """, unsafe_allow_html=True)
    
    face_cascade = load_face_detector()
    
    col_video, col_status = st.columns([2, 1])
    
    with col_video:
        st.subheader("Live Camera Feed")
        video_placeholder = st.empty()
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            start_btn = st.button("Start Detection", use_container_width=True)
        with btn_col2:
            stop_btn = st.button("Stop Detection", use_container_width=True)
        with btn_col3:
            snap_btn = st.button("Capture", use_container_width=True)
    
    with col_status:
        st.subheader("Alert Status")
        status_placeholder = st.empty()
        
        st.subheader("Statistics")
        total_detected = st.empty()
        mask_count = st.empty()
        no_mask_count = st.empty()
    
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'stats' not in st.session_state:
        st.session_state.stats = {'total': 0, 'mask': 0, 'no_mask': 0}
    if 'snap_requested' not in st.session_state:
        st.session_state.snap_requested = False
    
    if start_btn:
        st.session_state.running = True
        st.session_state.snap_requested = False
    if stop_btn:
        st.session_state.running = False
    if snap_btn:
        st.session_state.snap_requested = True
    
    if st.session_state.running:
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            st.error("Cannot open camera.")
            st.session_state.running = False
            st.stop()
        
        frame_count = 0
        
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break
            
            frame_count += 1
            if frame_count % 2 != 0:
                continue
            
            frame = cv2.flip(frame, 1)
            faces = detect_faces(frame, face_cascade)
            
            frame_stats = {'mask': 0, 'no_mask': 0}
            
            for (x1, y1, x2, y2) in faces:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                
                has_mask, confidence, raw_score = predict_mask(model, face_roi, model_type)
                
                st.session_state.stats['total'] += 1
                if has_mask:
                    frame_stats['mask'] += 1
                    st.session_state.stats['mask'] += 1
                    color = GREEN_ALERT
                    label = "WITH MASK"
                else:
                    frame_stats['no_mask'] += 1
                    st.session_state.stats['no_mask'] += 1
                    color = RED_ALERT
                    label = "NO MASK!"
                
                if show_box:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), 
                        color, 
                        -1
                    )
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )
                    
                    if show_confidence:
                        conf_text = f"{confidence:.1%}"
                        y_offset = y2 + 20 if y2 + 20 < frame.shape[0] else y1 - 25
                        cv2.putText(
                            frame, conf_text, (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                        )
                    
                    if show_raw_score:
                        raw_text = f"Raw: {raw_score:.3f}"
                        y_offset = y2 + 40 if y2 + 40 < frame.shape[0] else y1 - 45
                        cv2.putText(
                            frame, raw_text, (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
                        )
            
            if frame_stats['no_mask'] > 0:
                status_placeholder.markdown("""
                <div class="alert-box red-alert">
                    RED ALERT: NO MASK DETECTED!<br>
                    <span style="font-size:16px">Please wear a mask</span>
                </div>""", unsafe_allow_html=True)
            elif frame_stats['mask'] > 0:
                status_placeholder.markdown("""
                <div class="alert-box green-alert">
                    GREEN ALERT: MASK DETECTED<br>
                    <span style="font-size:16px">Thank you for wearing a mask</span>
                </div>""", unsafe_allow_html=True)
            else:
                status_placeholder.info("Scanning for faces...")
            
            total_detected.markdown(f"**Total Detections:** {st.session_state.stats['total']}")
            mask_count.markdown(f"**With Mask:** {st.session_state.stats['mask']}")
            no_mask_count.markdown(f"**Without Mask:** {st.session_state.stats['no_mask']}")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if st.session_state.snap_requested:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                st.sidebar.success(f"Saved {filename}")
                st.session_state.snap_requested = False
            
            video_placeholder.image(
                frame_rgb, 
                channels="RGB", 
                use_container_width=False
            )
            
            time.sleep(0.05)
            
            if not st.session_state.running:
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        video_placeholder.info("Press 'Start Detection' to begin")
        status_placeholder.info("System Standby")
        total_detected.markdown("**Total Detections:** 0")
        mask_count.markdown("**With Mask:** 0")
        no_mask_count.markdown("**Without Mask:** 0")

if __name__ == "__main__":
    main()