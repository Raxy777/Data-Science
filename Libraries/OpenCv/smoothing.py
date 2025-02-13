import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Resources/Photos/einstein.jpeg", 0)

noise = np.random.normal(0, 1, image.size).reshape(image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)
# Average blurring
blur = cv2.blur(noisy_image, (7, 7))
# Gaussian blurring
guassian = cv2.GaussianBlur(noisy_image, (7, 7), 0)
# Median
median = cv2.medianBlur(noisy_image, 7)
# Bilateral
bilateral = cv2.bilateralFilter(noisy_image, 7, 75, 75)

images = [image, noisy_image, blur, guassian, median, bilateral]
titles = ['Original', 'Noisy Image', 'Average', 'Gaussian', 'Median', 'Bilatral']

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(titles[i])
    ax.axis('off')

plt.show()
cv2.waitKey(0)