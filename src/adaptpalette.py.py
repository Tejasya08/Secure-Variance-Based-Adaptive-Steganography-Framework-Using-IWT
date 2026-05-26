"""
Adaptive palette-based steganography using all subbands
"""
import hashlib
import numpy as np
from embedding import KeyBasedPositionScrambler


class AdaptiveEmbedding:
    """Adaptive embedding across multiple subbands"""
    
    def __init__(self, key=12345):
        """Initialize adaptive embedding"""
        self.key = key
        self.allocations = {}
        self.scramblers = {}

    def analyze_subbands(self, LH, HL, HH):
        """
        Analyze subband variances for adaptive allocation
        
        Args:
            LH, HL, HH: Wavelet subbands
            
        Returns:
            Variance dictionary
        """
        variances = {
            'LH': np.var(LH.astype(np.float64)) + 1e-10,
            'HL': np.var(HL.astype(np.float64)) + 1e-10,
            'HH': np.var(HH.astype(np.float64)) + 1e-10,
        }
        self.variances = variances
        return variances

    def allocate_bits(self, total_bits, LH, HL, HH):
        """
        Allocate bits adaptively based on variance
        
        Args:
            total_bits: Total bits to allocate
            LH, HL, HH: Wavelet subbands
            
        Returns:
            Allocation dictionary
        """
        variances = self.analyze_subbands(LH, HL, HH)
        total_var = sum(variances.values())

        allocation = {}
        used = 0
        subs = ['LH', 'HL', 'HH']
        sizes = {'LH': LH.size, 'HL': HL.size, 'HH': HH.size}

        for i, name in enumerate(subs):
            if i == len(subs) - 1:
                bits = total_bits - used
            else:
                bits = int(total_bits * variances[name] / total_var)
            
            bits = min(bits, sizes[name])
            allocation[name] = bits
            used += bits
            
            sub_key = int(hashlib.md5(f"{self.key}_{name}".encode()).hexdigest(), 16) % (2**31)
            self.scramblers[name] = KeyBasedPositionScrambler(sizes[name], sub_key)

        self.allocations = allocation
        return allocation


class AdaptivePaletteSteganography:
    """Adaptive steganography using multiple subbands"""
    
    def __init__(self, wavelet_transform, key=12345):
        """
        Initialize adaptive palette steganography
        
        Args:
            wavelet_transform: Wavelet transform class (DWT or IWT)
            key: Secret key for scrambling
        """
        self.wavelet = wavelet_transform
        self.key = key
        self.adaptive = AdaptiveEmbedding(key=key)
    
    def embed(self, cover, secret, secret_ratio=0.25):
        """
        Embed secret adaptively across all subbands
        
        Args:
            cover: Cover image
            secret: Secret bits
            secret_ratio: Ratio of capacity to use
            
        Returns:
            Stego image, extracted bits, bits embedded
        """
        LL, LH, HL, HH = self.wavelet.forward(cover)
        
        total_capacity = LH.size + HL.size + HH.size
        n_to_embed = min(len(secret), int(total_capacity * secret_ratio))
        
        allocation = self.adaptive.allocate_bits(n_to_embed, LH, HL, HH)
        
        subbands = {'LH': LH.copy(), 'HL': HL.copy(), 'HH': HH.copy()}
        offset = 0
        total_embedded = 0
        
        # Convert to appropriate type
        is_iwt = 'IWT' in str(self.wavelet)
        
        for sub_name in ['LH', 'HL', 'HH']:
            target = subbands[sub_name]
            n_bits = min(allocation[sub_name], len(secret) - offset)
            
            if n_bits <= 0:
                continue
            
            if is_iwt:
                target = target.astype(np.int64)
            
            scrambler = self.adaptive.scramblers[sub_name]
            positions = scrambler.get_embedding_positions(n_bits)
            
            flat = target.flatten().copy()
            for i, pos in enumerate(positions):
                if i >= n_bits or offset + i >= len(secret):
                    break
                val = int(flat[pos])
                flat[pos] = (val & ~1) | int(secret[offset + i])
            
            subbands[sub_name] = flat.reshape(target.shape)
            offset += n_bits
            total_embedded += n_bits
        
        # Reconstruct stego image
        if is_iwt:
            stego = self.wavelet.inverse(
                LL, 
                subbands['LH'].astype(np.int64),
                subbands['HL'].astype(np.int64),
                subbands['HH'].astype(np.int64)
            )
        else:
            stego = self.wavelet.inverse(
                LL, 
                subbands['LH'].astype(np.float64),
                subbands['HL'].astype(np.float64),
                subbands['HH'].astype(np.float64)
            )
        
        # Extract for verification
        _, LH_e, HL_e, HH_e = self.wavelet.forward(stego)
        subs_ext = {'LH': LH_e, 'HL': HL_e, 'HH': HH_e}
        
        extracted = []
        for sub_name in ['LH', 'HL', 'HH']:
            n_bits = allocation[sub_name]
            if n_bits <= 0:
                continue
            
            scrambler = self.adaptive.scramblers[sub_name]
            positions = scrambler.get_embedding_positions(n_bits)
            flat = subs_ext[sub_name].flatten()
            
            for i, pos in enumerate(positions):
                if len(extracted) >= total_embedded:
                    break
                if is_iwt:
                    extracted.append(int(flat[pos]) & 1)
                else:
                    extracted.append(int(np.round(flat[pos])) & 1)
        
        return stego, np.array(extracted[:total_embedded], dtype=np.uint8), total_embedded
    
    def extract(self, stego, allocation):
        """
        Extract secret from stego image
        
        Args:
            stego: Stego image
            allocation: Bit allocation dictionary
            
        Returns:
            Extracted bits
        """
        _, LH, HL, HH = self.wavelet.forward(stego)
        subs = {'LH': LH, 'HL': HL, 'HH': HH}
        
        extracted = []
        is_iwt = 'IWT' in str(self.wavelet)
        
        for sub_name in ['LH', 'HL', 'HH']:
            n_bits = allocation[sub_name]
            if n_bits <= 0:
                continue
            
            scrambler = self.adaptive.scramblers[sub_name]
            positions = scrambler.get_embedding_positions(n_bits)
            flat = subs[sub_name].flatten()
            
            for i, pos in enumerate(positions):
                if len(extracted) >= sum(allocation.values()):
                    break
                if is_iwt:
                    extracted.append(int(flat[pos]) & 1)
                else:
                    extracted.append(int(np.round(flat[pos])) & 1)
        
        return np.array(extracted, dtype=np.uint8)