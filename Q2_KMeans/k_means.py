import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import os

image_path = "mandrill1.webp"

K_list = [2,4,8]

def mse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return np.mean((img1 - img2) ** 2)

image_bgr = cv2.imread(image_path)

if image_bgr is None:
    print("❌ ERROR: Could not load the image. Check path and filename.")
    exit()

# Convert to RGB
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w, c = image.shape

pixels = image.reshape(-1, 3)  

image_name = os.path.splitext(os.path.basename(image_path))[0]

# Output folder
output_dir = "quantized_output"
os.makedirs(output_dir, exist_ok=True)

Rates = []
Distortions = []

for K in K_list:
    print(f"\nProcessing K = {K}")

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centroids = kmeans.cluster_centers_.astype(np.uint8)

    quant_pixels = centroids[labels]
    quant_image = quant_pixels.reshape(h, w, 3)

    save_path = f"{output_dir}/{image_name}_K{K}.png"
    cv2.imwrite(save_path, cv2.cvtColor(quant_image, cv2.COLOR_RGB2BGR))
    print(f"✔ Saved quantized image: {save_path}")

    D = mse(image, quant_image)
    R = math.log2(K)

    Rates.append(R)
    Distortions.append(D)

    print(f"Rate = {R:.4f} bits | MSE = {D:.4f}")

plt.figure(figsize=(7,5))
plt.plot(Rates, Distortions, marker='o', linewidth=2)
plt.title(f"Rate–Distortion Curve: {image_name}")
plt.xlabel("Rate (bits/pixel) = log2(K)")
plt.ylabel("Distortion (MSE)")
plt.grid(True)

graph_path = f"{output_dir}/{image_name}_RD_curve.png"
plt.savefig(graph_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n📈 RD curve saved at: {graph_path}")
print("\n🎉 All processing complete! Check the 'quantized_output' folder.")

