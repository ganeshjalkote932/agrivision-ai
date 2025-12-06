"""
Model Service for Crop Disease Detection System

This module provides the ModelService class for loading and managing
the PyTorch disease detection model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from PIL import Image
import torchvision.transforms as transforms


class BetterCNN(nn.Module):
    """
    CNN model for hyperspectral crop disease detection.
    
    Architecture:
    - 2 Conv blocks with 64 filters
    - MaxPool
    - 2 Conv blocks with 128 filters
    - AdaptiveAvgPool
    - Fully connected classifier with dropout
    """
    
    def __init__(self, in_ch, n_classes):
        """
        Initialize the BetterCNN model.
        
        Args:
            in_ch: Number of input channels (spectral bands after PCA)
            n_classes: Number of output classes
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Output tensor of shape (batch, n_classes)
        """
        x = self.net(x)
        return self.classifier(x)


class ModelService:
    """
    Service class for managing the PyTorch disease detection model.
    
    Handles model loading, caching, GPU detection, and provides
    inference capabilities for hyperspectral image analysis.
    """
    
    def __init__(self, model_path: str = "best_model (2).pth"):
        """
        Initialize the ModelService.
        
        Args:
            model_path: Path to the PyTorch model file
        """
        self.model_path = model_path
        self.model: Optional[torch.nn.Module] = None
        self.device: Optional[torch.device] = None
        self._model_info: Optional[Dict[str, Any]] = None
        self.n_classes: Optional[int] = None
        self.in_channels: Optional[int] = None
        
    def load_model(self) -> bool:
        """
        Load the PyTorch model from disk and cache it in memory.
        
        Detects and uses GPU if available, otherwise falls back to CPU.
        Implements error handling for model loading failures.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found at: {self.model_path}"
                )
            
            # Detect device (GPU if available, otherwise CPU)
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"Using device: {self.device}")
            
            # Load the model
            # Note: Using weights_only=False to load the full model
            loaded_data = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False
            )
            
            # Check if it's a state_dict or a full model
            if isinstance(loaded_data, dict) and not hasattr(loaded_data, 'eval'):
                # It's a state_dict (OrderedDict)
                state_dict = loaded_data
                
                # Infer architecture from state dict
                if 'net.0.weight' in state_dict:
                    self.in_channels = state_dict['net.0.weight'].shape[1]
                else:
                    self.in_channels = 40  # default
                
                if 'classifier.5.weight' in state_dict:
                    self.n_classes = state_dict['classifier.5.weight'].shape[0]
                else:
                    self.n_classes = 2  # default
                
                # Create model with correct architecture
                self.model = BetterCNN(in_ch=self.in_channels, n_classes=self.n_classes)
                self.model.load_state_dict(state_dict)
                print(f"Loaded BetterCNN model:")
                print(f"  - Input channels: {self.in_channels}")
                print(f"  - Output classes: {self.n_classes}")
            else:
                # It's a full model object
                self.model = loaded_data
                # Try to infer parameters
                try:
                    self.in_channels = self.model.net[0].weight.shape[1]
                    self.n_classes = self.model.classifier[-1].weight.shape[0]
                except:
                    self.in_channels = None
                    self.n_classes = None
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Cache model info
            self._cache_model_info()
            
            print(f"Model loaded successfully from {self.model_path}")
            return True
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(f"Error: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _cache_model_info(self) -> None:
        """
        Cache model information for quick retrieval.
        
        Extracts and stores model architecture details, parameters,
        and configuration information.
        """
        if self.model is None:
            return
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        # Get model architecture as string
        model_architecture = str(self.model)
        
        # Get model file size
        model_size_bytes = os.path.getsize(self.model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Store cached info
        self._model_info = {
            'model_path': self.model_path,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(model_size_mb, 2),
            'model_size_bytes': model_size_bytes,
            'architecture': model_architecture,
            'model_type': type(self.model).__name__
        }
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready for inference.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.model is not None and self.device is not None
    
    def get_device(self) -> Optional[str]:
        """
        Get the device being used for inference.
        
        Returns:
            Optional[str]: Device name ('cuda' or 'cpu'), or None if not loaded
        """
        return str(self.device) if self.device else None
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Perform disease detection inference on an image.
        
        Takes an image path, preprocesses the image, runs inference,
        and returns a binary result (diseased/healthy) with confidence score.
        
        Args:
            image_path: Path to the hyperspectral image file
            
        Returns:
            dict: Dictionary containing:
                - 'result': str - 'diseased' or 'healthy'
                - 'confidence': float - Confidence score between 0 and 1
                - 'raw_output': float - Raw model output (optional)
                
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If image validation fails
            Exception: If inference fails
        """
        # Check if model is loaded
        if not self.is_loaded():
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )
        
        try:
            # Create preprocessor
            preprocessor = ImagePreprocessor()
            
            # Preprocess image
            image_tensor = preprocessor.preprocess(image_path)
            
            # Move tensor to device
            image_tensor = image_tensor.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                output = self.model(image_tensor)
            
            # Process output
            # Model outputs logits for each class
            print(f"Raw model output: {output}")
            
            if output.shape[-1] == 1:
                # Single output with sigmoid (binary classification)
                probability = torch.sigmoid(output).item()
                is_diseased = probability > 0.5
                confidence = probability if is_diseased else (1 - probability)
            else:
                # Multiple outputs with softmax (multi-class)
                probabilities = torch.softmax(output, dim=1)
                print(f"Probabilities after softmax: {probabilities}")
                
                # Get predicted class and its probability
                predicted_class = probabilities.argmax(dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                # Assuming class 0 is healthy, class 1 is diseased
                # Adjust based on your model's class mapping
                is_diseased = (predicted_class == 1)
                
                print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
            
            # Format result
            result = {
                'result': 'diseased' if is_diseased else 'healthy',
                'confidence': round(confidence, 4),
                'raw_output': output.cpu().numpy().tolist()
            }
            
            return result
            
        except ValueError as e:
            # Re-raise validation errors
            raise
            
        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            print(f"Error: {error_msg}")
            raise Exception(error_msg) from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns model architecture details, parameters, layers,
        and configuration information.
        
        Returns:
            dict: Dictionary containing:
                - 'model_path': str - Path to model file
                - 'device': str - Device being used (cuda/cpu)
                - 'total_parameters': int - Total number of parameters
                - 'trainable_parameters': int - Number of trainable parameters
                - 'model_size_mb': float - Model file size in MB
                - 'model_size_bytes': int - Model file size in bytes
                - 'architecture': str - Model architecture description
                - 'model_type': str - Model class name
                - 'layers': list - List of layer information (if available)
                
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )
        
        # Return cached info with additional layer details
        info = self._model_info.copy()
        
        # Add layer information
        layers = []
        try:
            for name, module in self.model.named_modules():
                if name:  # Skip the root module
                    layer_info = {
                        'name': name,
                        'type': type(module).__name__,
                    }
                    
                    # Add parameter count for this layer
                    layer_params = sum(p.numel() for p in module.parameters())
                    if layer_params > 0:
                        layer_info['parameters'] = layer_params
                    
                    layers.append(layer_info)
            
            info['layers'] = layers
            info['num_layers'] = len(layers)
            
        except Exception as e:
            print(f"Warning: Could not extract layer details: {e}")
            info['layers'] = []
            info['num_layers'] = 0
        
        return info


class ImagePreprocessor:
    """
    Preprocessor for hyperspectral images before model inference.
    
    Handles image validation, format checking, and transformation
    to prepare images for the disease detection model.
    Supports both standard image formats and .npy files.
    
    Preprocessing pipeline matches training:
    1. Resize to 32x32
    2. Apply PCA to reduce to 40 bands (if needed)
    3. Extract center 24x24 patch
    """
    
    # Allowed image formats - ONLY .npy files
    ALLOWED_FORMATS = set()  # No standard image formats
    ALLOWED_EXTENSIONS = {'.npy'}  # Only numpy arrays
    
    # Size constraints (in bytes)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_FILE_SIZE = 1024  # 1KB
    
    # Dimension constraints (in pixels)
    MIN_WIDTH = 32
    MIN_HEIGHT = 32
    MAX_WIDTH = 4096
    MAX_HEIGHT = 4096
    
    def __init__(self, resize_to: tuple = (32, 32), patch_size: int = 24, target_bands: int = 40):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            resize_to: Resize images to this size first (H, W) - matches training
            patch_size: Extract center patch of this size - matches training
            target_bands: Target number of spectral bands after PCA
        """
        self.resize_to = resize_to
        self.patch_size = patch_size
        self.target_bands = target_bands
        
        # For standard RGB images, define transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.ToTensor(),
        ])
    
    def validate_file(self, file_path: str) -> tuple[bool, Optional[str]]:
        """
        Validate image file before processing.
        
        Checks file existence, size, format, and dimensions.
        Supports both standard image formats and .npy files.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            tuple: (is_valid, error_message)
                - is_valid: True if file is valid, False otherwise
                - error_message: Description of validation error, or None if valid
        """
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        # Check file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in self.ALLOWED_EXTENSIONS:
            return False, f"Invalid file extension: {ext}. Allowed: {self.ALLOWED_EXTENSIONS}"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size < self.MIN_FILE_SIZE:
            return False, f"File too small: {file_size} bytes (minimum: {self.MIN_FILE_SIZE})"
        if file_size > self.MAX_FILE_SIZE:
            return False, f"File too large: {file_size} bytes (maximum: {self.MAX_FILE_SIZE})"
        
        # Only .npy files are supported
        if ext.lower() == '.npy':
            try:
                # Try to load the numpy array
                data = np.load(file_path)
                # Check if it's a valid array
                if not isinstance(data, np.ndarray):
                    return False, "Invalid .npy file: not a numpy array"
                # Check dimensions (should be at least 2D)
                if data.ndim < 2:
                    return False, f"Invalid .npy file: array must be at least 2D, got {data.ndim}D"
                return True, None
            except Exception as e:
                return False, f"Invalid or corrupted .npy file: {str(e)}"
        else:
            return False, f"Only .npy files are supported. Got: {ext}"
    
    def preprocess(self, file_path: str) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Loads image, applies transformations, and returns tensor.
        Supports both standard image formats and .npy files.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
            
        Raises:
            ValueError: If image validation fails
            RuntimeError: If preprocessing fails
        """
        # Validate file first
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        try:
            # Check if it's a .npy file (hyperspectral data)
            _, ext = os.path.splitext(file_path)
            if ext.lower() == '.npy':
                # Load numpy array - expected shape: (H, W, Bands)
                data = np.load(file_path)
                
                # Step 1: Resize to 32x32 (matches training)
                if data.ndim == 3:
                    H, W, B = data.shape
                    if H != self.resize_to[0] or W != self.resize_to[1]:
                        # Simple nearest neighbor resize
                        y_indices = np.linspace(0, H-1, self.resize_to[0]).astype(int)
                        x_indices = np.linspace(0, W-1, self.resize_to[1]).astype(int)
                        data = data[np.ix_(y_indices, x_indices)]
                        print(f"Resized from {H}x{W} to {self.resize_to[0]}x{self.resize_to[1]}")
                
                # Step 2: Apply PCA if bands > target_bands
                if data.ndim == 3 and data.shape[2] > self.target_bands:
                    H, W, B = data.shape
                    print(f"Applying PCA: {B} bands → {self.target_bands} bands")
                    
                    # Flatten spatial dimensions
                    flat = data.reshape(-1, B)
                    
                    # Simple PCA: standardize and take first N components
                    # Standardize
                    mean = flat.mean(axis=0)
                    std = flat.std(axis=0) + 1e-8
                    flat_scaled = (flat - mean) / std
                    
                    # For simplicity, just take first N bands (pseudo-PCA)
                    # In production, use sklearn.decomposition.PCA
                    flat_reduced = flat_scaled[:, :self.target_bands]
                    
                    # Reshape back
                    data = flat_reduced.reshape(H, W, self.target_bands).astype(np.float32)
                    print(f"PCA complete: shape now {data.shape}")
                
                # Step 3: Extract center patch (24x24 from 32x32)
                if data.ndim == 3:
                    H, W, B = data.shape
                    if H >= self.patch_size and W >= self.patch_size:
                        # Extract center patch
                        start_h = (H - self.patch_size) // 2
                        start_w = (W - self.patch_size) // 2
                        data = data[start_h:start_h+self.patch_size, 
                                  start_w:start_w+self.patch_size, :]
                        print(f"Extracted center {self.patch_size}x{self.patch_size} patch")
                
                # Convert to tensor and transpose to (C, H, W)
                image_tensor = torch.from_numpy(data).float()
                
                # Ensure it has the right shape
                if image_tensor.ndim == 2:
                    # 2D image, add channel dimension
                    image_tensor = image_tensor.unsqueeze(0)
                elif image_tensor.ndim == 3:
                    # 3D image (H, W, C) -> (C, H, W)
                    image_tensor = image_tensor.permute(2, 0, 1)
                
                # Add batch dimension
                if image_tensor.ndim == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                
                print(f"Final tensor shape: {image_tensor.shape} (batch, channels, height, width)")
                return image_tensor
            
            # If we reach here, file is not .npy
            raise ValueError(f"Only .npy files are supported. Please upload hyperspectral data as .npy format.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess image: {str(e)}") from e
