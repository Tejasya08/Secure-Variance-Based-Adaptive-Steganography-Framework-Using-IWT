"""
Integer Wavelet Transform (IWT) module for lossless steganography
"""
import numpy as np


class IWT:
    """Integer Wavelet Transform - Lossless wavelet transform"""
    
    @staticmethod
    def forward(image):
        """
        Perform forward IWT on image
        
        Args:
            image: Input grayscale image
            
        Returns:
            LL, LH, HL, HH: Four integer wavelet subbands
        """
        img = image.astype(np.int64)
        rows, cols = img.shape
        hr, hc = rows // 2, cols // 2

        tL = np.zeros((rows, hc), dtype=np.int64)
        tH = np.zeros((rows, hc), dtype=np.int64)

        for i in range(rows):
            even, odd = img[i, 0::2], img[i, 1::2]
            tH[i] = odd - even
            tL[i] = even + (tH[i] >> 1)

        LL = np.zeros((hr, hc), dtype=np.int64)
        LH = np.zeros((hr, hc), dtype=np.int64)
        HL = np.zeros((hr, hc), dtype=np.int64)
        HH = np.zeros((hr, hc), dtype=np.int64)

        for j in range(hc):
            eL, oL = tL[0::2, j], tL[1::2, j]
            LH[:, j] = oL - eL
            LL[:, j] = eL + (LH[:, j] >> 1)
            eH, oH = tH[0::2, j], tH[1::2, j]
            HH[:, j] = oH - eH
            HL[:, j] = eH + (HH[:, j] >> 1)

        return LL, LH, HL, HH

    @staticmethod
    def inverse(LL, LH, HL, HH):
        """
        Perform inverse IWT to reconstruct image
        
        Args:
            LL, LH, HL, HH: Integer wavelet subbands
            
        Returns:
            Losslessly reconstructed image
        """
        LL, LH = LL.astype(np.int64), LH.astype(np.int64)
        HL, HH = HL.astype(np.int64), HH.astype(np.int64)
        hr, hc = LL.shape
        rows, cols = hr * 2, hc * 2

        tL = np.zeros((rows, hc), dtype=np.int64)
        tH = np.zeros((rows, hc), dtype=np.int64)

        for j in range(hc):
            eL = LL[:, j] - (LH[:, j] >> 1)
            tL[0::2, j], tL[1::2, j] = eL, LH[:, j] + eL
            eH = HL[:, j] - (HH[:, j] >> 1)
            tH[0::2, j], tH[1::2, j] = eH, HH[:, j] + eH

        img = np.zeros((rows, cols), dtype=np.int64)
        for i in range(rows):
            even = tL[i] - (tH[i] >> 1)
            img[i, 0::2], img[i, 1::2] = even, tH[i] + even

        return np.clip(img, 0, 255).astype(np.uint8)
    
    @staticmethod
    def verify_lossless(test_image):
        """
        Verify IWT is lossless
        
        Args:
            test_image: Test image for verification
            
        Returns:
            True if lossless, False otherwise
        """
        LL, LH, HL, HH = IWT.forward(test_image)
        reconstructed = IWT.inverse(LL, LH, HL, HH)
        err = np.mean(np.abs(test_image.astype(float) - reconstructed.astype(float)))
        return err == 0