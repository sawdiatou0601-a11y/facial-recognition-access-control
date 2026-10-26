import cv2
import numpy as np
import sys

entree = sys.argv[1] if len(sys.argv) > 1 else "capture_test.jpg"
sortie = sys.argv[2] if len(sys.argv) > 2 else "capture_corrigee.jpg"

img = cv2.imread(entree).astype(np.float32)
moy_b, moy_g, moy_r = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
gris = (moy_b + moy_g + moy_r) / 3
img[..., 0] *= gris / moy_b
img[..., 1] *= gris / moy_g
img[..., 2] *= gris / moy_r
img = np.clip(img, 0, 255).astype(np.uint8)
cv2.imwrite(sortie, img)
print("Image corrigee ->", sortie)
