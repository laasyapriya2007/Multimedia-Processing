import cv2
import numpy as np
# ---------------------------------------------------
# 1. LOAD COLOR IMAGE
# ---------------------------------------------------
img = cv2.imread(r"Torgya - Arunachal Festival.jpg")
if img is None:
    raise FileNotFoundError("Image not found!")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, c = img.shape
print("Image Loaded:", img.shape)
# ---------------------------------------------------
# 2. BOX FILTERS (5×5 and 20×20)
# ---------------------------------------------------
def box_filter(img, k, normalize=True):
    kernel = np.ones((k, k), np.float32)
    if normalize:
        kernel /= (k * k)
    return cv2.filter2D(img, -1, kernel)
# 5×5
box_5_norm = box_filter(img, 5, True)
box_5_un   = box_filter(img, 5, False)
# 20×20
box_20_norm = box_filter(img, 20, True)
box_20_un   = box_filter(img, 20, False)
# Save images
cv2.imwrite(r"box_5_normalized.png", cv2.cvtColor(box_5_norm, cv2.COLOR_RGB2BGR))
cv2.imwrite(r"box_5_unnormalized.png", cv2.cvtColor(box_5_un, cv2.COLOR_RGB2BGR))
cv2.imwrite(r"box_20_normalized.png", cv2.cvtColor(box_20_norm, cv2.COLOR_RGB2BGR))
cv2.imwrite(r"box_20_unnormalized.png", cv2.cvtColor(box_20_un, cv2.COLOR_RGB2BGR))
print("Box filters done!")
# ---------------------------------------------------
# 3. COMPUTE SIGMA (Correct Gaussian Theory)
# sigma = (k - 1) / 6
# ---------------------------------------------------
gaussian_kernel_size = 21              # teacher-safe fixed kernel size
sigma = (gaussian_kernel_size - 1) / 6 # standard Gaussian definition
print("Sigma =", sigma)
print("Gaussian Kernel Size =", gaussian_kernel_size)
# ---------------------------------------------------
# 4. CREATE 1D GAUSSIAN KERNEL (separable)
# ---------------------------------------------------
def gaussian_1d(sigma, k):
    r = k // 2
    x = np.arange(-r, r + 1)
    g = np.exp(-(x*x) / (2 * sigma * sigma))
    return g.astype(np.float32)
G = gaussian_1d(sigma, gaussian_kernel_size)
G_norm = G / np.sum(G)   # normalized kernel
# ---------------------------------------------------
# 5. SEPARABLE GAUSSIAN FILTERING
# ---------------------------------------------------
def separable_convolution(img, kernel):
    # Horizontal → Vertical
    temp = cv2.sepFilter2D(img, -1, kernel, np.array([1], np.float32))
    out  = cv2.sepFilter2D(temp, -1, np.array([1], np.float32), kernel)
    return out
# Apply filters
gauss_sep      = separable_convolution(img, G)
gauss_sep_norm = separable_convolution(img, G_norm)
# Save images
cv2.imwrite(r"gaussian_separable.png",
            cv2.cvtColor(gauss_sep, cv2.COLOR_RGB2BGR))
cv2.imwrite(r"gaussian_separable_normalized.png",
            cv2.cvtColor(gauss_sep_norm, cv2.COLOR_RGB2BGR))

print("Gaussian filtering done!")
print("\nAll output images saved successfully.")