import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load tiger image you uploaded
img = cv2.imread(r"input_image.jpg")

def frequency_sampling_rgb(img, factor):
    # Split B, G, R channels (OpenCV reads in BGR order)
    b, g, r = cv2.split(img)
    channels = [b, g, r]
    output_channels = []

    for ch in channels:
        h, w = ch.shape

        F = np.fft.fftshift(np.fft.fft2(ch))
        shrink = int(1/factor) if factor < 1 else int(factor)

        new_h = h // shrink
        new_w = w // shrink

        cy, cx = h//2, w//2
        top = cy - new_h//2
        left = cx - new_w//2

        low = F[top:top+new_h, left:left+new_w]
        padded = np.zeros((h, w), dtype=complex)

        pad_top = (h - new_h)//2
        pad_left = (w - new_w)//2

        padded[pad_top:pad_top+low.shape[0], pad_left:pad_left+low.shape[1]] = low
        ch_back = np.abs(np.fft.ifft2(np.fft.ifftshift(padded)))

        output_channels.append(ch_back.astype(np.uint8))

    final_img = cv2.merge((output_channels[0], output_channels[1], output_channels[2]))
    return final_img


result = frequency_sampling_rgb(img, factor=1/16)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('on')
plt.title("Frequency Sampled RGB ")
plt.show()
