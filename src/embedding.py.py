"""
LSB embedding and extraction utilities
"""
import numpy as np


class LSBMethods:
    """LSB (Least Significant Bit) embedding methods"""
    
    @staticmethod
    def embed(coefficients, secret_bits):
        """
        Embed secret bits into coefficients using LSB
        
        Args:
            coefficients: Coefficient array (numpy array)
            secret_bits: Secret bits to embed
            
        Returns:
            Modified coefficients and number of bits embedded
        """
        flat = coefficients.flatten().copy().astype(np.float64)
        n = min(len(secret_bits), len(flat))

        for i in range(n):
            val = int(np.round(flat[i]))
            new_val = (val & ~1) | int(secret_bits[i])
            flat[i] = float(new_val)

        return flat.reshape(coefficients.shape), n

    @staticmethod
    def extract(coefficients, n_bits):
        """
        Extract secret bits from coefficients
        
        Args:
            coefficients: Coefficient array
            n_bits: Number of bits to extract
            
        Returns:
            Extracted secret bits
        """
        flat = coefficients.flatten()
        n = min(n_bits, len(flat))
        return np.array([int(np.round(flat[i])) & 1 for i in range(n)], dtype=np.uint8)

    @staticmethod
    def embed_at_positions(coefficients, secret_bits, positions):
        """
        Embed bits at specific positions
        
        Args:
            coefficients: Coefficient array
            secret_bits: Secret bits to embed
            positions: Positions to embed at
            
        Returns:
            Modified coefficients and number embedded
        """
        flat = coefficients.flatten().copy().astype(np.float64)
        n = min(len(secret_bits), len(positions))
        
        for i in range(n):
            pos = positions[i]
            val = int(np.round(flat[pos]))
            flat[pos] = float((val & ~1) | int(secret_bits[i]))
        
        return flat.reshape(coefficients.shape), n
    
    @staticmethod
    def extract_from_positions(coefficients, positions, n_bits):
        """
        Extract bits from specific positions
        
        Args:
            coefficients: Coefficient array
            positions: Positions to extract from
            n_bits: Number of bits to extract
            
        Returns:
            Extracted bits
        """
        flat = coefficients.flatten()
        n = min(n_bits, len(positions))
        extracted = np.zeros(n, dtype=np.uint8)
        
        for i in range(n):
            pos = positions[i]
            extracted[i] = int(np.round(flat[pos])) & 1
        
        return extracted


class KeyBasedPositionScrambler:
    """Position scrambler using cryptographic key"""
    
    def __init__(self, size, key=12345):
        """
        Initialize scrambler
        
        Args:
            size: Size of coefficient array
            key: Secret key for permutation
        """
        self.size = size
        self.key = key
        self._generate_permutation()

    def _generate_permutation(self):
        """Generate permutation based on key"""
        if isinstance(self.key, int):
            seed = self.key
        else:
            import hashlib
            seed = int(hashlib.md5(str(self.key).encode()).hexdigest(), 16) % (2 ** 31)

        rng = np.random.RandomState(seed)
        self.positions = rng.permutation(self.size)
        
        self.inverse_positions = np.zeros(self.size, dtype=int)
        for i, pos in enumerate(self.positions):
            self.inverse_positions[pos] = i

    def get_embedding_positions(self, n_bits):
        """Get positions for embedding"""
        return self.positions[:n_bits]

    def get_extraction_order(self, n_bits):
        """Get order for extraction"""
        return self.positions[:n_bits]