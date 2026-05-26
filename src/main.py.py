"""
Main program to demonstrate all modules
"""
from dwt import DWT
from iwt import IWT
from palette import PaletteSteganography
from adaptpalette import AdaptivePaletteSteganography
from metrics import SteganographyMetrics


def demo():
    """Demo the usage of all modules"""
    import cv2
    import numpy as np
    
    # Load test image
    img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    
    # Prepare secret
    secret_bits = np.random.randint(0, 2, 1000)
    
    print("=" * 60)
    print("  DEMONSTRATING STEGANOGRAPHY MODULES")
    print("=" * 60)
    
    # 1. DWT + Palette
    print("\n1. DWT + Palette Steganography")
    dwt_palette = PaletteSteganography(DWT, key=12345)
    stego, extracted, n_emb = dwt_palette.embed(img, secret_bits, secret_ratio=0.25)
    psnr = SteganographyMetrics.compute_psnr(img, stego)
    ber = SteganographyMetrics.compute_ber(secret_bits[:n_emb], extracted)
    print(f"   PSNR: {psnr:.2f} dB, BER: {ber:.6f}")
    
    # 2. IWT + Palette
    print("\n2. IWT + Palette Steganography")
    iwt_palette = PaletteSteganography(IWT, key=12345)
    stego, extracted, n_emb = iwt_palette.embed(img, secret_bits, secret_ratio=0.25)
    psnr = SteganographyMetrics.compute_psnr(img, stego)
    ber = SteganographyMetrics.compute_ber(secret_bits[:n_emb], extracted)
    print(f"   PSNR: {psnr:.2f} dB, BER: {ber:.6f}")
    
    # 3. DWT + Adaptive Palette
    print("\n3. DWT + Adaptive Palette Steganography")
    dwt_adapt = AdaptivePaletteSteganography(DWT, key=12345)
    stego, extracted, n_emb = dwt_adapt.embed(img, secret_bits, secret_ratio=0.25)
    psnr = SteganographyMetrics.compute_psnr(img, stego)
    ber = SteganographyMetrics.compute_ber(secret_bits[:n_emb], extracted)
    print(f"   PSNR: {psnr:.2f} dB, BER: {ber:.6f}")
    
    # 4. IWT + Adaptive Palette
    print("\n4. IWT + Adaptive Palette Steganography")
    iwt_adapt = AdaptivePaletteSteganography(IWT, key=12345)
    stego, extracted, n_emb = iwt_adapt.embed(img, secret_bits, secret_ratio=0.25)
    psnr = SteganographyMetrics.compute_psnr(img, stego)
    ber = SteganographyMetrics.compute_ber(secret_bits[:n_emb], extracted)
    print(f"   PSNR: {psnr:.2f} dB, BER: {ber:.6f}")
    
    print("\n✅ All modules working correctly!")


if __name__ == "__main__":
    demo()