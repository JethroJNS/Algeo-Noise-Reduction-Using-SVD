import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def svd_noise_reduction(image, rank):

    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    # Retain only the top rank singular values
    S_reduced = np.zeros_like(S)
    S_reduced[:rank] = S[:rank]
    # Reconstruct the denoised image
    noise_reduced_image = np.dot(U, np.dot(np.diag(S_reduced), Vt))
    # Clip values to ensure they remain in the valid range [0, 255]
    noise_reduced_image = np.clip(noise_reduced_image, 0, 255)
    return noise_reduced_image.astype(np.uint8)


# Load a grayscale image with noise
image_path = "./test/sample.png"
noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Set the desired rank for noise reduction
rank = 50

noise_reduced_image = svd_noise_reduction(noisy_image, rank)

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Noise-reduced Image")
plt.imshow(noise_reduced_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save the denoised image
output_path = "./test/noise_reduced_image.jpg"
cv2.imwrite(output_path, noise_reduced_image)
