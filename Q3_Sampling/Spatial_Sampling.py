import cv2
import matplotlib.pyplot as plt

def spatial_sampling_rgb(img, factor):
    b, g, r = cv2.split(img)

    sampled_channels = []

    for ch in [b, g, r]:
        small = cv2.resize(
            ch, None,
            fx=1/factor, fy=1/factor,
            interpolation=cv2.INTER_NEAREST
        )

        sampled = cv2.resize(
            small,
            (ch.shape[1], ch.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        sampled_channels.append(sampled)

    sampled_rgb = cv2.merge(sampled_channels)
    return sampled_rgb

img = cv2.imread(r"C:\Users\kgnan\OneDrive\Desktop\disney-animals-asian-sumatran-tigers-00.jpg")

result = spatial_sampling_rgb(img, factor=16) 

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Spatially Sampled (RGB)")
plt.axis('on')
plt.show()
