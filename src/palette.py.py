"""
Palette-based steganography module
"""
import numpy as np
from embedding import KeyBasedPositionScrambler, LSBMethods


class PaletteSteganography:
    """
    Palette-based steganography with key-based position scrambling
    """
    
    def __init__(self, wavelet_transform, key=12345):
        """
        Initialize palette steganography
        
        Args:
            wavelet_transform: Wavelet transform class (DWT or IWT)
            key: Secret key for scrambling
        """
        self.wavelet = wavelet_transform
        self.key = key
    
    def embed(self, cover, secret, secret_ratio=0.25):
        """
        Embed secret using HH subband with scrambling
        
        Args:
            cover: Cover image
            secret: Secret bits
            secret_ratio: Ratio of capacity to use
            
        Returns:
            Stego image, extracted bits, bits embedded
        """
        LL, LH, HL, HH = self.wavelet.forward(cover)
        
        max_capacity = HH.size
        n_to_embed = min(len(secret), int(max_capacity * secret_ratio))
        
        scrambler = KeyBasedPositionScrambler(HH.size, key=self.key)
        positions = scrambler.get_embedding_positions(n_to_embed)
        
        HH_flat = HH.flatten().copy()
        
        # Convert to appropriate type based on wavelet
        if isinstance(self.wavelet.__name__, str) and 'IWT' in str(self.wavelet):
            HH_flat = HH_flat.astype(np.int64)
        
        for i, pos in enumerate(positions):
            if i >= n_to_embed:
                break
            val = int(HH_flat[pos])
            HH_flat[pos] = (val & ~1) | int(secret[i])
        
        if isinstance(self.wavelet.__name__, str) and 'IWT' in str(self.wavelet):
            HH_stego = HH_flat.reshape(HH.shape)
            stego = self.wavelet.inverse(LL, LH, HL, HH_stego.astype(np.int64))
        else:
            HH_stego = HH_flat.reshape(HH.shape)
            stego = self.wavelet.inverse(LL, LH, HL, HH_stego.astype(np.float64))
        
        # Extract for verification
        _, _, _, HH_ext = self.wavelet.forward(stego)
        HH_ext_flat = HH_ext.flatten()
        
        extracted = np.zeros(n_to_embed, dtype=np.uint8)
        for i, pos in enumerate(positions):
            if i >= n_to_embed:
                break
            extracted[i] = int(HH_ext_flat[pos]) & 1
        
        return stego, extracted, n_to_embed
    
    def extract(self, stego, n_bits):
        """
        Extract secret from stego image
        
        Args:
            stego: Stego image
            n_bits: Number of bits to extract
            
        Returns:
            Extracted bits
        """
        _, _, _, HH = self.wavelet.forward(stego)
        scrambler = KeyBasedPositionScrambler(HH.size, key=self.key)
        positions = scrambler.get_extraction_order(n_bits)
        
        flat = HH.flatten()
        extracted = np.zeros(n_bits, dtype=np.uint8)
        
        for i, pos in enumerate(positions):
            if i >= n_bits:
                break
            extracted[i] = int(flat[pos]) & 1
        
        return extracted