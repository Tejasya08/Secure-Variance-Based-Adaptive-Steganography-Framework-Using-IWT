"""
Attack module for robustness testing in steganography
"""
import os
import cv2
import numpy as np
from collections import OrderedDict


class SteganoAttacks:
    """Collection of attacks for robustness testing"""
    
    @staticmethod
    def jpeg_compression(image, quality_factor):
        """
        Apply JPEG compression attack
        
        Args:
            image: Input image
            quality_factor: JPEG quality factor (1-100)
            
        Returns:
            Compressed image
        """
        tmp = '_temp_jpeg.jpg'
        cv2.imwrite(tmp, image, [cv2.IMWRITE_JPEG_QUALITY, quality_factor])
        attacked = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
        if os.path.exists(tmp):
            os.remove(tmp)
        
        # Resize if dimensions changed
        if attacked is not None and attacked.shape != image.shape:
            attacked = cv2.resize(attacked, (image.shape[1], image.shape[0]))
        
        return attacked if attacked is not None else image
    
    @staticmethod
    def gaussian_noise(image, sigma):
        """
        Add Gaussian noise attack
        
        Args:
            image: Input image
            sigma: Standard deviation of noise
            
        Returns:
            Image with added noise
        """
        noise = np.random.normal(0, sigma, image.shape)
        attacked = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        return attacked
    
    @staticmethod
    def salt_pepper_noise(image, amount=0.05):
        """
        Add salt and pepper noise attack
        
        Args:
            image: Input image
            amount: Noise proportion
            
        Returns:
            Image with salt and pepper noise
        """
        attacked = image.copy()
        num_salt = int(amount * image.size * 0.5)
        num_pepper = int(amount * image.size * 0.5)
        
        # Salt (white) noise
        coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
        attacked[coords[0], coords[1]] = 255
        
        # Pepper (black) noise
        coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
        attacked[coords[0], coords[1]] = 0
        
        return attacked
    
    @staticmethod
    def median_filter(image, kernel_size=3):
        """
        Apply median filtering attack
        
        Args:
            image: Input image
            kernel_size: Size of median filter kernel
            
        Returns:
            Filtered image
        """
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def gaussian_blur(image, kernel_size=(3, 3), sigma=0):
        """
        Apply Gaussian blur attack
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation (0 = auto)
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    @staticmethod
    def average_filter(image, kernel_size=(3, 3)):
        """
        Apply average filtering attack
        
        Args:
            image: Input image
            kernel_size: Size of averaging kernel
            
        Returns:
            Filtered image
        """
        return cv2.blur(image, kernel_size)
    
    @staticmethod
    def histogram_equalization(image):
        """
        Apply histogram equalization attack
        
        Args:
            image: Input image
            
        Returns:
            Equalized image
        """
        return cv2.equalizeHist(image)
    
    @staticmethod
    def brightness_adjustment(image, delta):
        """
        Adjust brightness attack
        
        Args:
            image: Input image
            delta: Brightness adjustment value (±)
            
        Returns:
            Brightness-adjusted image
        """
        return np.clip(image.astype(float) + delta, 0, 255).astype(np.uint8)
    
    @staticmethod
    def contrast_adjustment(image, alpha):
        """
        Adjust contrast attack
        
        Args:
            image: Input image
            alpha: Contrast multiplier
            
        Returns:
            Contrast-adjusted image
        """
        return np.clip(image.astype(float) * alpha, 0, 255).astype(np.uint8)
    
    @staticmethod
    def rotation(image, angle):
        """
        Apply rotation attack
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image (cropped to original size)
        """
        h, w = image.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated
    
    @staticmethod
    def scaling(image, scale_factor):
        """
        Apply scaling attack (resize)
        
        Args:
            image: Input image
            scale_factor: Scaling factor (<1 = downscale, >1 = upscale)
            
        Returns:
            Scaled image (resized back to original)
        """
        h, w = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled = cv2.resize(image, (new_w, new_h))
        scaled_back = cv2.resize(scaled, (w, h))
        return scaled_back
    
    @staticmethod
    def cropping(image, crop_percent):
        """
        Apply cropping attack
        
        Args:
            image: Input image
            crop_percent: Percentage to crop from each side
            
        Returns:
            Cropped and resized image
        """
        h, w = image.shape
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        
        cropped = image[crop_h:h-crop_h, crop_w:w-crop_w]
        resized = cv2.resize(cropped, (w, h))
        return resized


class AttackSuite:
    """Complete attack suite for robustness evaluation"""
    
    def __init__(self):
        """Initialize attack suite with predefined attacks"""
        self.attacks = OrderedDict()
        self._setup_attacks()
    
    def _setup_attacks(self):
        """Setup all attack configurations"""
        
        # JPEG compression attacks
        for qf in [90, 70, 50, 30]:
            self.attacks[f'JPEG QF={qf}'] = lambda img, q=qf: SteganoAttacks.jpeg_compression(img, q)
        
        # Gaussian noise attacks
        for sigma in [5, 10, 20]:
            self.attacks[f'Noise σ={sigma}'] = lambda img, s=sigma: SteganoAttacks.gaussian_noise(img, s)
        
        # Salt & pepper noise
        for amount in [0.01, 0.05, 0.1]:
            self.attacks[f'SaltPepper {int(amount*100)}%'] = lambda img, a=amount: SteganoAttacks.salt_pepper_noise(img, a)
        
        # Filtering attacks
        self.attacks['Median 3×3'] = lambda img: SteganoAttacks.median_filter(img, 3)
        self.attacks['Median 5×5'] = lambda img: SteganoAttacks.median_filter(img, 5)
        self.attacks['GaussBlur 3×3'] = lambda img: SteganoAttacks.gaussian_blur(img, (3, 3), 0)
        self.attacks['GaussBlur 5×5'] = lambda img: SteganoAttacks.gaussian_blur(img, (5, 5), 0)
        self.attacks['Average 3×3'] = lambda img: SteganoAttacks.average_filter(img, (3, 3))
        
        # Geometric attacks
        for angle in [5, 15, 30]:
            self.attacks[f'Rotation {angle}°'] = lambda img, a=angle: SteganoAttacks.rotation(img, a)
        
        # Scaling attacks
        for scale in [0.9, 0.8, 0.7]:
            self.attacks[f'Scale {int(scale*100)}%'] = lambda img, s=scale: SteganoAttacks.scaling(img, s)
        
        # Photometric attacks
        self.attacks['Histogram EQ'] = lambda img: SteganoAttacks.histogram_equalization(img)
        
        for delta in [20, 50, -20, -50]:
            sign = '+' if delta > 0 else ''
            self.attacks[f'Brightness {sign}{delta}'] = lambda img, d=delta: SteganoAttacks.brightness_adjustment(img, d)
        
        for alpha in [0.8, 1.2, 1.5]:
            self.attacks[f'Contrast {alpha}x'] = lambda img, a=alpha: SteganoAttacks.contrast_adjustment(img, a)
    
    def get_attack_names(self):
        """Get list of attack names"""
        return list(self.attacks.keys())
    
    def apply_attack(self, image, attack_name):
        """
        Apply specific attack to image
        
        Args:
            image: Input image
            attack_name: Name of attack to apply
            
        Returns:
            Attacked image
        """
        if attack_name in self.attacks:
            return self.attacks[attack_name](image)
        else:
            raise ValueError(f"Unknown attack: {attack_name}")
    
    def apply_all_attacks(self, image):
        """
        Apply all attacks to image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of attacked images
        """
        results = {}
        for name, attack_func in self.attacks.items():
            try:
                results[name] = attack_func(image)
            except Exception as e:
                print(f"  ⚠️ Attack '{name}' failed: {e}")
                results[name] = image.copy()
        return results
    
    def compute_attack_robustness(self, stego_image, extraction_func, secret_bits):
        """
        Compute robustness against all attacks
        
        Args:
            stego_image: Stego image
            extraction_func: Function to extract bits from image
            secret_bits: Original secret bits
            
        Returns:
            Dictionary of BER values for each attack
        """
        results = {}
        
        for attack_name in self.attacks.keys():
            try:
                attacked = self.apply_attack(stego_image, attack_name)
                extracted = extraction_func(attacked, len(secret_bits))
                
                # Compute BER
                mn = min(len(secret_bits), len(extracted))
                ber = np.sum(secret_bits[:mn] != extracted[:mn]) / mn if mn > 0 else 1.0
                
                results[attack_name] = ber
            except Exception as e:
                print(f"  ⚠️ Error with {attack_name}: {e}")
                results[attack_name] = 1.0
        
        return results


# Predefined attack sets for common evaluations
QUICK_ATTACKS = OrderedDict([
    ('JPEG QF=90', lambda i: SteganoAttacks.jpeg_compression(i, 90)),
    ('JPEG QF=70', lambda i: SteganoAttacks.jpeg_compression(i, 70)),
    ('JPEG QF=50', lambda i: SteganoAttacks.jpeg_compression(i, 50)),
    ('JPEG QF=30', lambda i: SteganoAttacks.jpeg_compression(i, 30)),
    ('Noise σ=5', lambda i: SteganoAttacks.gaussian_noise(i, 5)),
    ('Noise σ=10', lambda i: SteganoAttacks.gaussian_noise(i, 10)),
    ('Noise σ=20', lambda i: SteganoAttacks.gaussian_noise(i, 20)),
    ('Median 3×3', lambda i: SteganoAttacks.median_filter(i, 3)),
    ('GaussBlur', lambda i: SteganoAttacks.gaussian_blur(i, (3, 3), 0)),
])

STANDARD_ATTACKS = OrderedDict([
    # JPEG Attacks
    ('JPEG QF=90', lambda i: SteganoAttacks.jpeg_compression(i, 90)),
    ('JPEG QF=70', lambda i: SteganoAttacks.jpeg_compression(i, 70)),
    ('JPEG QF=50', lambda i: SteganoAttacks.jpeg_compression(i, 50)),
    ('JPEG QF=30', lambda i: SteganoAttacks.jpeg_compression(i, 30)),
    
    # Noise Attacks
    ('Noise σ=5', lambda i: SteganoAttacks.gaussian_noise(i, 5)),
    ('Noise σ=10', lambda i: SteganoAttacks.gaussian_noise(i, 10)),
    ('Noise σ=20', lambda i: SteganoAttacks.gaussian_noise(i, 20)),
    ('SaltPepper 5%', lambda i: SteganoAttacks.salt_pepper_noise(i, 0.05)),
    
    # Filtering Attacks
    ('Median 3×3', lambda i: SteganoAttacks.median_filter(i, 3)),
    ('GaussBlur', lambda i: SteganoAttacks.gaussian_blur(i, (3, 3), 0)),
    ('Average 3×3', lambda i: SteganoAttacks.average_filter(i, (3, 3))),
    
    # Photometric Attacks
    ('Brightness +50', lambda i: SteganoAttacks.brightness_adjustment(i, 50)),
    ('Brightness -50', lambda i: SteganoAttacks.brightness_adjustment(i, -50)),
    ('Contrast 0.8x', lambda i: SteganoAttacks.contrast_adjustment(i, 0.8)),
    ('Contrast 1.2x', lambda i: SteganoAttacks.contrast_adjustment(i, 1.2)),
    ('Histogram EQ', lambda i: SteganoAttacks.histogram_equalization(i)),
    
    # Geometric Attacks
    ('Rotation 10°', lambda i: SteganoAttacks.rotation(i, 10)),
    ('Scale 90%', lambda i: SteganoAttacks.scaling(i, 0.9)),
    ('Crop 10%', lambda i: SteganoAttacks.cropping(i, 0.1)),
])