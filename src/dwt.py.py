"""
Discrete Wavelet Transform (DWT) module for steganography
"""
import pywt
import numpy as np

WAVELET = 'haar'


class DWT:
    """Discrete Wavelet Transform operations"""
    
    @staticmethod
    def forward(image):
        """
        Perform forward DWT on image
        
        Args:
            image: Input grayscale image
            
        Returns:
            LL, LH, HL, HH: Four wavelet subbands
        """
        LL, (LH, HL, HH) = pywt.dwt2(image.astype(np.float64), WAVELET)
        return LL, LH, HL, HH

    @staticmethod
    def inverse(LL, LH, HL, HH):
        """
        Perform inverse DWT to reconstruct image
        
        Args:
            LL, LH, HL, HH: Wavelet subbands
            
        Returns:
            Reconstructed image
        """
        return np.clip(pywt.idwt2((LL, (LH, HL, HH)), WAVELET), 0, 255).astype(np.uint8)