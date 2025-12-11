import cv2
import numpy as np
img = cv2.imread(r"brightlight.png", 0)
bit0 = (img >> 0) & 1
bit1 = (img >> 1) & 1
bit2 = (img >> 2) & 1
reconstructed = (bit0*1 + bit1*2 + bit2*4).astype(np.uint8)
difference = cv2.subtract(img, reconstructed)
cv2.imwrite("Original-brightlight.png", img)
cv2.imwrite("Reconstructed_raw_0_1_2-brightlight.png", reconstructed)
cv2.imwrite("Difference-brightlight.png", difference)