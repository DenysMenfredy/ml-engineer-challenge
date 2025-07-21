"""Pre
This module contains the preprocessor for the OCR pipeline.
"""

import cv2
import numpy as np
from scipy import ndimage
from ml_pipeline.ocr.base import OCRResult

class OCRPreprocessor:
    """Abstract base class for OCR preprocessors."""
    def __init__(self) -> None:
        pass

    def binarize_image(self, image: np.ndarray, method: str = "adaptive", block_size: int = 11, c: int = 2):
        """Binarize the image using the specified method.
        Args:
            image: Input grayscale image
            method: 'otsu', 'adaptive', 'global', 'sauvola'
            block_size: Size of neighborhood for adaptive thresholding (must be odd)
            c: Constant subtracted from mean in adaptive thresholding
        Returns:
            The binarized image.
        """
        binary = None
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if method == "otsu":
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                image, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                block_size, 
                c
            )
        elif method == "global":
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

        elif method == "sauvola":
            window_size = 25
            k = 0.5
            binary = self._sauvola_threshold(image, window_size, k)
        
        return binary

    def _sauvola_threshold(self, image: np.ndarray, window_size, k):
        """Sauvola thresholding for binarization."""
        mean = cv2.boxFilter(image.astype(np.float32), -1, (window_size, window_size))
        sqmean = cv2.boxFilter((image.astype(np.float32)) ** 2, -1, (window_size, window_size))
        std = np.sqrt(sqmean - mean ** 2)

        threshold = mean * (1 + k * (std / 128 - 1))
        binary = np.where(image > threshold, 255, 0).astype(np.uint8)

        return binary

    def remove_noise(self, image: np.ndarray, method: str = "bilateral", **kwargs):
        """Remove noise from the image using the specified method.
        
        Args:
            image: Input grayscale image
            method: 'bilateral', 'median', 'gaussian' 'morphological'
            **kwargs: Additional parameters for the method
        Returns:
            The denoised image.
        """
        if method == "gaussian":
            kernel_size = kwargs.get("kernel_size", 5)
            sigma = kwargs.get("sigma", 1.0)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        elif method == "median":
            kernel_size = kwargs.get("kernel_size", 5)
            return cv2.medianBlur(image, kernel_size)

        elif method == "bilateral":
            d = kwargs.get("d", 9)
            sigma_color = kwargs.get("sigma_color", 75)
            sigma_space = kwargs.get("sigma_space", 75)
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        elif method == "morphological":
            kernel_size = kwargs.get("kernel_size", 3)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            return closed

        
    def correct_skew(self, image: np.ndarray, method: str = "hough"):
        """Correct the skew of the image using the specified method.
        
        Args:
            image: Input grayscale image
            method: 'hough', 'contour', 'projection'
        Returns:
            The corrected image.
        """
        if method == "hough":
            return self._correct_skew_hough(image)
        elif method == "contour":
            return self._correct_skew_contour(image)
        elif method == "projection":
            return self._correct_skew_projection(image)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        
    def _correct_skew_hough(self, image: np.ndarray):
        """Correct the skew of the image using the Hough transform."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi - 90
                if -45 < angle < 45: # Only consider horizontal lines
                    angles.append(angle)

            if angles:
                skew_angle = np.mean(angles)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                corrected = cv2.warpAffine(image, rotation_matrix, (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
                return corrected, skew_angle
            
        return image, 0
    
    def _correct_skew_contour(self, image: np.ndarray):
        """Skew correction using text line contours"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Binarize
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (likely text lines)
        text_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 10 and h < 100:  # Reasonable text line dimensions
                text_contours.append(contour)
        
        if len(text_contours) > 3:
            # Calculate angles of text lines
            angles = []
            for contour in text_contours:
                # Fit line to contour
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = np.arctan2(vy, vx) * 180 / np.pi
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                skew_angle = np.median(angles)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                corrected = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
                return corrected, skew_angle
        
        return image, 0

    def _correct_skew_projection(self, image: np.ndarray):
        """Skew correction using horizontal projection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarize if not already binary
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        def calculate_score(angle):
            # Rotate image
            rotated = ndimage.rotate(binary, angle, reshape=False, order=0)
            # Calculate horizontal projection
            projection = np.sum(rotated, axis=1)
            # Score is variance of projection (higher = better alignment)
            return np.var(projection)
        
        # Test angles from -10 to 10 degrees
        angles = np.arange(-10, 10.1, 0.1)
        scores = [calculate_score(angle) for angle in angles]
        
        # Find angle with maximum score
        best_angle = angles[np.argmax(scores)]
        
        # Apply correction
        corrected = ndimage.rotate(image, best_angle, reshape=False)
        return corrected, best_angle

    def scale_image(self, image, target_dpi=300, current_dpi=72, method='cubic'):
        """
        Scale image to optimal resolution for OCR
        
        Args:
            image: Input image
            target_dpi: Target DPI (typically 300 for OCR)
            current_dpi: Current image DPI
            method: Interpolation method ('nearest', 'linear', 'cubic', 'lanczos')
        """
        # Calculate scaling factor
        scale_factor = target_dpi / current_dpi
        
        if abs(scale_factor - 1.0) < 0.1:  # No significant scaling needed
            return image
        
        # Get new dimensions
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Choose interpolation method
        interpolation_methods = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        interp = interpolation_methods.get(method, cv2.INTER_CUBIC)
        
        # Apply scaling
        scaled = cv2.resize(image, (new_w, new_h), interpolation=interp)
        
        return scaled

    def preprocess_pipeline(self, image: np.ndarray, pipeline_config=None):
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image
            pipeline_config: Dict with processing parameters
        """
        if pipeline_config is None:
            pipeline_config = {
                'scale': {'target_dpi': 300, 'current_dpi': 72},
                'noise_removal': {'method': 'bilateral'},
                'skew_correction': {'method': 'hough'},
                'binarization': {'method': 'adaptive', 'block_size': 11}
            }
        
        result = image.copy()
        processing_info = {}
        
        # 1. Scaling
        if 'scale' in pipeline_config:
            result = self.scale_image(result, **pipeline_config['scale'])
            processing_info['scaling'] = 'applied'
        
        # 2. Noise removal
        if 'noise_removal' in pipeline_config:
            result = self.remove_noise(result, **pipeline_config['noise_removal'])
            processing_info['noise_removal'] = pipeline_config['noise_removal']['method']
        
        # 3. Skew correction
        if 'skew_correction' in pipeline_config:
            result, skew_angle = self.correct_skew(result, **pipeline_config['skew_correction'])
            processing_info['skew_angle'] = skew_angle
        
        # 4. Binarization
        if 'binarization' in pipeline_config:
            result = self.binarize_image(result, **pipeline_config['binarization'])
            processing_info['binarization'] = pipeline_config['binarization']['method']
        
        return result, processing_info




