import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#sobel algorithm
# Open the image

# Sobel Operator
class Prewitt_3x3:
    def __init__(self, image_path):
        self.img = np.array(Image.open(image_path)).astype(np.uint8)

    def prewitt_3x3(self):
        gray_img = np.round(0.299 * self.img[:, :, 0] +
                            0.587 * self.img[:, :, 1] +
                            0.114 * self.img[:, :, 2]).astype(np.uint8)

        h, w = gray_img.shape

        # Define filters
        horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # s2
        vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # s1

        # Define images with 0s
        newhorizontalImage = np.zeros((h, w))
        newverticalImage = np.zeros((h, w))
        newgradientImage = np.zeros((h, w))

        # Offset by 1
        #i sütun j satır
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                                 (horizontal[0, 1] * gray_img[i - 1, j]) + \
                                 (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                                 (horizontal[1, 0] * gray_img[i, j - 1]) + \
                                 (horizontal[1, 1] * gray_img[i, j]) + \
                                 (horizontal[1, 2] * gray_img[i, j + 1]) + \
                                 (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                                 (horizontal[2, 1] * gray_img[i + 1, j]) + \
                                 (horizontal[2, 2] * gray_img[i + 1, j + 1])

                newhorizontalImage[i, j] = abs(horizontalGrad)

                verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                               (vertical[0, 1] * gray_img[i - 1, j]) + \
                               (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                               (vertical[1, 0] * gray_img[i, j - 1]) + \
                               (vertical[1, 1] * gray_img[i, j]) + \
                               (vertical[1, 2] * gray_img[i, j + 1]) + \
                               (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                               (vertical[2, 1] * gray_img[i + 1, j]) + \
                               (vertical[2, 2] * gray_img[i + 1, j + 1])

                newverticalImage[i, j] = abs(verticalGrad)

                # Edge Magnitude
                # G=sqrt(Gx^2+Gy^2)
                mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
                newgradientImage[i, j] = mag

        return newgradientImage


