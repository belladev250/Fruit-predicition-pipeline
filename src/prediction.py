import numpy as np
import tensorflow as tf
from PIL import Image
import pickle


class Predictor:
    """Make predictions with trained model"""
    
    def __init__(self, model_path, classes_path):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model (.h5 file)
            classes_path: Path to saved classes (.pkl file)
        """
        self.model = tf.keras.models.load_model(model_path)
        
        # Load classes
        with open(classes_path, 'rb') as f:
            self.classes = pickle.load(f)
        
        # Auto-detect input size from model
        self.img_size = self._detect_input_size()
        
        print(f"✅ Model loaded: {model_path}")
        print(f"✅ Classes loaded: {list(self.classes)}")
        print(f"✅ Detected input size: {self.img_size}")
    
    def _detect_input_size(self):
        """Detect the correct input size from model architecture"""
        input_shape = self.model.input_shape
        
        if len(input_shape) == 4:  # (batch, height, width, channels)
            height, width = input_shape[1], input_shape[2]
            return (height, width)
        else:
            # Default to common sizes
            print("⚠️  Could not detect input size, trying common sizes...")
            return (224, 224)  # Try this first
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.array: Preprocessed image with batch dimension
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        """
        Make prediction on single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Prediction results with class and confidence
        """
        img_array = self.preprocess_image(image_path)
        
        if img_array is None:
            return {'error': 'Failed to process image'}
        
        try:
            prediction = self.model.predict(img_array, verbose=0)
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
            
            return {
                'class': str(self.classes[class_idx]),
                'confidence': confidence,
                'probabilities': {
                    str(self.classes[i]): float(prediction[0][i]) 
                    for i in range(len(self.classes))
                }
            }
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    # ... keep the rest of your methods the same ...
    def predict_batch(self, image_paths):
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results
    
    def predict_from_array(self, img_array):
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        prediction = self.model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        return {
            'class': str(self.classes[class_idx]),
            'confidence': confidence,
            'probabilities': {
                str(self.classes[i]): float(prediction[0][i]) 
                for i in range(len(self.classes))
            }
        }
    
    def get_classes(self):
        return list(self.classes)
    
    def get_num_classes(self):
        return len(self.classes)