"""
Metrics computation for steganography evaluation
"""
import numpy as np
import cv2
from collections import OrderedDict
from skimage.metrics import structural_similarity


class SteganographyMetrics:
    """Comprehensive metrics for steganography evaluation"""
    
    @staticmethod
    def compute_psnr(cover, stego):
        """
        Compute Peak Signal-to-Noise Ratio
        
        Args:
            cover: Cover image
            stego: Stego image
            
        Returns:
            PSNR in dB
        """
        c = cover.astype(np.float64)
        s = stego.astype(np.float64)
        mse = np.mean((c - s) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
        return psnr
    
    @staticmethod
    def compute_ssim(cover, stego):
        """
        Compute Structural Similarity Index
        
        Args:
            cover: Cover image
            stego: Stego image
            
        Returns:
            SSIM value
        """
        return structural_similarity(cover, stego.astype(np.uint8), data_range=255)
    
    @staticmethod
    def compute_mse(cover, stego):
        """
        Compute Mean Squared Error
        
        Args:
            cover: Cover image
            stego: Stego image
            
        Returns:
            MSE value
        """
        c = cover.astype(np.float64)
        s = stego.astype(np.float64)
        return np.mean((c - s) ** 2)
    
    @staticmethod
    def compute_ncc(cover, stego):
        """
        Compute Normalized Cross-Correlation
        
        Args:
            cover: Cover image
            stego: Stego image
            
        Returns:
            NCC value
        """
        c = cover.astype(np.float64)
        s = stego.astype(np.float64)
        return np.sum(c * s) / (np.sqrt(np.sum(c ** 2) * np.sum(s ** 2)) + 1e-10)
    
    @staticmethod
    def compute_uiqi(cover, stego):
        """
        Compute Universal Image Quality Index
        
        Args:
            cover: Cover image
            stego: Stego image
            
        Returns:
            UIQI value
        """
        c = cover.astype(np.float64)
        s = stego.astype(np.float64)
        mc, ms = np.mean(c), np.mean(s)
        vc, vs = np.var(c), np.var(s)
        cov = np.mean((c - mc) * (s - ms))
        uiqi = (4 * cov * mc * ms) / ((vc + vs) * (mc ** 2 + ms ** 2) + 1e-10)
        return uiqi
    
    @staticmethod
    def compute_ber(original_bits, extracted_bits):
        """
        Compute Bit Error Rate
        
        Args:
            original_bits: Original secret bits
            extracted_bits: Extracted bits
            
        Returns:
            BER value
        """
        mn = min(len(original_bits), len(extracted_bits))
        ber = np.sum(original_bits[:mn] != extracted_bits[:mn]) / mn if mn > 0 else 1.0
        return ber
    
    @staticmethod
    def compute_correlation(cover, stego):
        """
        Compute correlation coefficient between cover and stego
        
        Args:
            cover: Cover image
            stego: Stego image
            
        Returns:
            Correlation coefficient
        """
        c = cover.astype(np.float64).flatten()
        s = stego.astype(np.float64).flatten()
        return np.corrcoef(c, s)[0, 1]
    
    @staticmethod
    def compute_entropy(image):
        """
        Compute image entropy
        
        Args:
            image: Input image
            
        Returns:
            Entropy value
        """
        hist = cv2.calcHist([image.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
        p = hist / hist.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    @staticmethod
    def compute_all_metrics(cover, stego, secret, extracted, n_emb, elapsed):
        """
        Compute all metrics
        
        Args:
            cover: Cover image
            stego: Stego image
            secret: Original secret bits
            extracted: Extracted bits
            n_emb: Number of bits embedded
            elapsed: Processing time
            
        Returns:
            Ordered dictionary of all metrics
        """
        c = cover.astype(np.float64)
        s = stego.astype(np.float64)
        
        mse = SteganographyMetrics.compute_mse(cover, stego)
        psnr = SteganographyMetrics.compute_psnr(cover, stego)
        ssim = SteganographyMetrics.compute_ssim(cover, stego)
        ncc = SteganographyMetrics.compute_ncc(cover, stego)
        uiqi = SteganographyMetrics.compute_uiqi(cover, stego)
        ber = SteganographyMetrics.compute_ber(secret, extracted)
        corr = SteganographyMetrics.compute_correlation(cover, stego)
        
        ent_c = SteganographyMetrics.compute_entropy(cover)
        ent_s = SteganographyMetrics.compute_entropy(stego)
        
        return OrderedDict({
            'PSNR (dB)': round(psnr, 4),
            'SSIM': round(ssim, 6),
            'MSE': round(mse, 4),
            'NCC': round(ncc, 6),
            'UIQI': round(uiqi, 6),
            'BER': round(ber, 6),
            'Correlation': round(corr, 6),
            'Entropy Cover': round(ent_c, 4),
            'Entropy Stego': round(ent_s, 4),
            'Entropy Change': round(abs(ent_s - ent_c), 6),
            'Max Error': round(float(np.max(np.abs(c - s))), 2),
            'Mean Abs Error': round(float(np.mean(np.abs(c - s))), 4),
            'Capacity (bpp)': round(n_emb / (cover.shape[0] * cover.shape[1]), 6),
            'Time (sec)': round(elapsed, 4),
        })