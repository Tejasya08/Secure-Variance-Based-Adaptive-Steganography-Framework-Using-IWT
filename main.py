"""
Main program integrating all modules
"""
import os
import sys
import time
import warnings
import numpy as np
import cv2
import pandas as pd
from collections import OrderedDict

# Import all modules
from dwt import DWT
from iwt import IWT
from embedding import LSBMethods, KeyBasedPositionScrambler
from palette import PaletteSteganography
from adaptpalette import AdaptivePaletteSteganography
from metrics import SteganographyMetrics
from attacks import AttackSuite, QUICK_ATTACKS, STANDARD_ATTACKS
from visualization import SteganoVisualizer, TECHNIQUE_COLORS, TECHNIQUE_ORDER

warnings.filterwarnings('ignore')

# Configuration
COVER_IMAGES_FOLDER = "/content/drive/MyDrive/cover png images"
RESULTS_FOLDER = "results"
FIGURES_FOLDER = "figures"
STEGO_FOLDER = "stego_images"

SECRET_RATIO = 0.25
WAVELET = 'haar'
SECRET_KEY = 12345
USE_TEXT_SECRET = True
SECRET_MESSAGE = "This is a secret message hidden using wavelet steganography for IEEE conference at NIT Goa 2024"


class SecretData:
    """Secret data handling utilities"""
    
    @staticmethod
    def text_to_bits(text):
        bits = []
        for char in text:
            bits.extend([int(b) for b in format(ord(char), '08b')])
        return np.array(bits, dtype=np.uint8)

    @staticmethod
    def bits_to_text(bits):
        text = ''
        for i in range(0, len(bits) - 7, 8):
            byte = bits[i:i + 8]
            val = int(''.join(str(int(b)) for b in byte), 2)
            if 32 <= val < 127:
                text += chr(val)
            elif val == 0:
                break
        return text

    @staticmethod
    def prepare_secret(max_capacity):
        if USE_TEXT_SECRET:
            msg_bits = SecretData.text_to_bits(SECRET_MESSAGE)
            length_bits = np.array([int(b) for b in format(len(msg_bits), '032b')], dtype=np.uint8)
            secret = np.concatenate([length_bits, msg_bits])
            if len(secret) < max_capacity:
                padding = np.random.randint(0, 2, max_capacity - len(secret), dtype=np.uint8)
                secret = np.concatenate([secret, padding])
            info = f"Text: '{SECRET_MESSAGE[:40]}...' ({len(msg_bits)} bits)"
        else:
            secret = np.random.randint(0, 2, max_capacity, dtype=np.uint8)
            info = f"Random: {max_capacity} bits"
        return secret[:max_capacity], info

    @staticmethod
    def extract_message(bits):
        if len(bits) < 32:
            return "Error: insufficient bits"
        length = int(''.join(str(int(b)) for b in bits[:32]), 2)
        if length <= 0 or length > len(bits) - 32:
            return f"Error: invalid length"
        return SecretData.bits_to_text(bits[32:32 + length])


def setup_directories():
    """Create necessary directories"""
    for folder in [RESULTS_FOLDER, FIGURES_FOLDER, STEGO_FOLDER]:
        os.makedirs(folder, exist_ok=True)


def load_all_images():
    """Load all images from cover folder"""
    if not os.path.exists(COVER_IMAGES_FOLDER):
        print(f"\n❌ Folder '{COVER_IMAGES_FOLDER}/' not found!")
        sys.exit(1)

    extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.bmp', '*.BMP', '*.tif', '*.tiff']
    import glob
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(COVER_IMAGES_FOLDER, ext)))
    paths = sorted(set(paths))

    if not paths:
        print(f"\n❌ No images found!")
        sys.exit(1)

    images = OrderedDict()
    print(f"\n{'=' * 60}")
    print(f"  📸 LOADING IMAGES")
    print(f"{'=' * 60}")

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            img = img[:h - (h % 2), :w - (w % 2)]
            images[name] = img
            print(f"  ✅ {name:20s} | {img.shape[0]}×{img.shape[1]}")

    print(f"\n  Total: {len(images)} images")
    return images


def verify_transforms():
    """Verify DWT and IWT transforms"""
    print(f"\n{'=' * 60}")
    print(f"  🔍 VERIFYING TRANSFORMS")
    print(f"{'=' * 60}")

    test = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    # DWT verification
    LL, LH, HL, HH = DWT.forward(test)
    r = DWT.inverse(LL, LH, HL, HH)
    print(f"  DWT error: {np.mean(np.abs(test.astype(float) - r.astype(float))):.6f}")

    # IWT verification
    LL, LH, HL, HH = IWT.forward(test)
    r = IWT.inverse(LL, LH, HL, HH)
    err = np.mean(np.abs(test.astype(float) - r.astype(float)))
    print(f"  IWT error: {err:.6f} {'✅ LOSSLESS!' if err == 0 else ''}")


def get_techniques():
    """Get all steganography techniques"""
    techniques = OrderedDict()
    
    # DWT + HH only (basic)
    def dwt_basic(cover, secret):
        LL, LH, HL, HH = DWT.forward(cover)
        max_capacity = HH.size
        n_to_embed = min(len(secret), int(max_capacity * SECRET_RATIO))
        HH_stego, n_embedded = LSBMethods.embed(HH, secret[:n_to_embed])
        stego = DWT.inverse(LL, LH, HL, HH_stego)
        _, _, _, HH_ext = DWT.forward(stego)
        extracted = LSBMethods.extract(HH_ext, n_embedded)
        return stego, extracted, n_embedded
    techniques['DWT'] = dwt_basic
    
    # IWT + HH only (basic)
    def iwt_basic(cover, secret):
        LL, LH, HL, HH = IWT.forward(cover)
        max_capacity = HH.size
        n_to_embed = min(len(secret), int(max_capacity * SECRET_RATIO))
        HH_stego, n_embedded = LSBMethods.embed(HH.astype(np.float64), secret[:n_to_embed])
        stego = IWT.inverse(LL, LH, HL, HH_stego.astype(np.int64))
        _, _, _, HH_ext = IWT.forward(stego)
        extracted = LSBMethods.extract(HH_ext, n_embedded)
        return stego, extracted, n_embedded
    techniques['IWT'] = iwt_basic
    
    # DWT + Palette
    dwt_palette = PaletteSteganography(DWT, key=SECRET_KEY)
    techniques['DWT+Palette'] = lambda c, s: dwt_palette.embed(c, s, SECRET_RATIO)
    
    # IWT + Palette
    iwt_palette = PaletteSteganography(IWT, key=SECRET_KEY)
    techniques['IWT+Palette'] = lambda c, s: iwt_palette.embed(c, s, SECRET_RATIO)
    
    # DWT + Adaptive Palette
    dwt_adapt = AdaptivePaletteSteganography(DWT, key=SECRET_KEY)
    techniques['DWT+AdaptPalette'] = lambda c, s: dwt_adapt.embed(c, s, SECRET_RATIO)
    
    # IWT + Adaptive Palette
    iwt_adapt = AdaptivePaletteSteganography(IWT, key=SECRET_KEY)
    techniques['IWT+AdaptPalette'] = lambda c, s: iwt_adapt.embed(c, s, SECRET_RATIO)
    
    return techniques


def experiment_main(images):
    """Main experiment comparing all techniques"""
    print(f"\n{'=' * 80}")
    print(f"  📊 EXPERIMENT 1: MAIN COMPARISON")
    print(f"{'=' * 80}")

    techniques = get_techniques()
    results = []

    first_img = list(images.values())[0]
    test_cap = int((first_img.shape[0] // 2) * (first_img.shape[1] // 2) * SECRET_RATIO)
    _, secret_info = SecretData.prepare_secret(test_cap)
    print(f"\n  🔒 Secret: {secret_info}")
    print(f"  🔑 Key: {SECRET_KEY}")

    for img_name, cover in images.items():
        cap = int((cover.shape[0] // 2) * (cover.shape[1] // 2) * SECRET_RATIO)
        secret, _ = SecretData.prepare_secret(cap)

        print(f"\n  📸 {img_name} ({cover.shape[0]}×{cover.shape[1]}) | {cap} bits")

        for tech_name, tech_func in techniques.items():
            try:
                t0 = time.time()
                stego, extracted, n_emb = tech_func(cover, secret)
                elapsed = time.time() - t0

                metrics = SteganographyMetrics.compute_all_metrics(
                    cover, stego, secret, extracted, n_emb, elapsed
                )
                metrics['Image'] = img_name
                metrics['Technique'] = tech_name
                metrics['Embedded Bits'] = n_emb
                results.append(metrics)

                cv2.imwrite(os.path.join(STEGO_FOLDER,
                            f"{img_name}_{tech_name.replace('+', '_')}.png"), stego)

                print(f"     {tech_name:22s} │ PSNR: {metrics['PSNR (dB)']:7.2f} │ "
                      f"SSIM: {metrics['SSIM']:.4f} │ BER: {metrics['BER']:.6f}")

            except Exception as e:
                print(f"     ❌ {tech_name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_FOLDER, 'all_results.csv'), index=False)
    return df


def experiment_robustness(images):
    """Robustness experiment against attacks"""
    print(f"\n{'=' * 80}")
    print(f"  🔨 EXPERIMENT 2: ROBUSTNESS")
    print(f"{'=' * 80}")

    techniques = get_techniques()
    attack_suite = AttackSuite()
    
    # Use first 3 images for robustness testing
    rob_imgs = dict(list(images.items())[:3])
    results = []

    for img_name, cover in rob_imgs.items():
        cap = int((cover.shape[0] // 2) * (cover.shape[1] // 2) * SECRET_RATIO)
        secret, _ = SecretData.prepare_secret(cap)

        print(f"\n  📸 {img_name}")

        for tech_name, tech_func in techniques.items():
            try:
                stego, _, n_emb = tech_func(cover, secret)

                # Define extraction function for this technique
                def extract_func(img, n):
                    _, ext, _ = tech_func(img, secret[:n])
                    return ext

                # Compute robustness against all attacks
                robustness = attack_suite.compute_attack_robustness(
                    stego, extract_func, secret[:n_emb]
                )

                for attack_name, ber in robustness.items():
                    results.append({
                        'Image': img_name,
                        'Technique': tech_name,
                        'Attack': attack_name,
                        'BER After Attack': round(ber, 6),
                    })

                print(f"     {tech_name:22s} │ {len(robustness)} attacks tested")
            except Exception as e:
                print(f"     ❌ {tech_name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_FOLDER, 'robustness_results.csv'), index=False)
    return df


def main():
    """Main execution function"""
    t0 = time.time()
    setup_directories()

    images = load_all_images()
    verify_transforms()

    print(f"\n  🔧 CONFIGURATION:")
    print(f"     ✅ LSB embedding with position scrambling")
    print(f"     ✅ Secret ratio: {SECRET_RATIO}")
    print(f"     ✅ Wavelet: {WAVELET}")

    # Run experiments
    df = experiment_main(images)
    df_rob = experiment_robustness(images)

    # Generate visualizations
    visualizer = SteganoVisualizer(FIGURES_FOLDER)
    visualizer.plot_comparison_bar_chart(df)
    
    # Generate robustness heatmap if results exist
    if not df_rob.empty:
        visualizer.plot_robustness_heatmap(df_rob)

    total = time.time() - t0
    print(f"\n{'🎉' * 40}")
    print(f"\n  ✅ COMPLETE! {total:.0f}s ({total/60:.1f} min)")
    print(f"\n  📁 OUTPUT FILES:")
    print(f"     📊 {RESULTS_FOLDER}/all_results.csv")
    print(f"     📊 {RESULTS_FOLDER}/robustness_results.csv")
    print(f"     🎨 {FIGURES_FOLDER}/fig1_comparison.png")
    print(f"     🎨 {FIGURES_FOLDER}/fig_robustness_heatmap.png")
    print(f"\n{'🎉' * 40}")


if __name__ == "__main__":
    main()
