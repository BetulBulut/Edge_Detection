import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#sobel algorithm
# Open the image

# Sobel Operator
class Sobel_5x5:
    def __init__(self, image_path):
        self.img = np.array(Image.open(image_path)).astype(np.uint8)

    def sobel_5x5(self):
        gray_img = np.round(0.299 * self.img[:, :, 0] +
                            0.587 * self.img[:, :, 1] +
                            0.114 * self.img[:, :, 2]).astype(np.uint8)
        h, w = gray_img.shape
        horizontal = np.array(
            [[2, 2, 4, 2, 2], [1, 1, 2, 1, 1], [0, 0, 0, 0, 0], [-1, -1, -2, -1, -1], [-2, -2, -4, -2, -2]])  # s2
        vertical = np.array(
            [[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-4, -2, 0, 2, 4], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]])  # s1
        newhorizontalImage = np.zeros((h, w))
        newverticalImage = np.zeros((h, w))
        newgradientImage = np.zeros((h, w))

        for i in range(1, h - 2):
            for j in range(1, w - 2):
                horizontalGrad = (horizontal[0, 0] * gray_img[i - 2, j - 2]) + \
                                 (horizontal[0, 1] * gray_img[i - 2, j - 1]) + \
                                 (horizontal[0, 2] * gray_img[i - 2, j]) + \
                                 (horizontal[0, 3] * gray_img[i - 2, j + 1]) + \
                                 (horizontal[0, 4] * gray_img[i - 2, j + 2]) + \
                                 (horizontal[1, 0] * gray_img[i - 1, j - 2]) + \
                                 (horizontal[1, 1] * gray_img[i - 1, j + 1]) + \
                                 (horizontal[1, 2] * gray_img[i - 1, j]) + \
                                 (horizontal[1, 3] * gray_img[i - 1, j + 1]) + \
                                 (horizontal[1, 4] * gray_img[i - 1, j + 2]) + \
                                 (horizontal[2, 0] * gray_img[i, j - 2]) + \
                                 (horizontal[2, 1] * gray_img[i, j - 1]) + \
                                 (horizontal[2, 2] * gray_img[i, j]) + \
                                 (horizontal[2, 3] * gray_img[i, j + 1]) + \
                                 (horizontal[2, 4] * gray_img[i, j + 2]) + \
                                 (horizontal[3, 0] * gray_img[i + 1, j - 2]) + \
                                 (horizontal[3, 1] * gray_img[i + 1, j - 1]) + \
                                 (horizontal[3, 2] * gray_img[i + 1, j]) + \
                                 (horizontal[3, 3] * gray_img[i + 1, j + 1]) + \
                                 (horizontal[3, 4] * gray_img[i + 1, j + 2]) + \
                                 (horizontal[4, 0] * gray_img[i + 2, j - 2]) + \
                                 (horizontal[4, 1] * gray_img[i + 2, j - 1]) + \
                                 (horizontal[4, 2] * gray_img[i + 2, j]) + \
                                 (horizontal[4, 3] * gray_img[i + 2, j + 1]) + \
                                 (horizontal[4, 4] * gray_img[i + 2, j + 2])

                newhorizontalImage[i, j] = abs(horizontalGrad)

                verticalGrad = (vertical[0, 0] * gray_img[i - 2, j - 2]) + \
                               (vertical[0, 1] * gray_img[i - 2, j - 1]) + \
                               (vertical[0, 2] * gray_img[i - 2, j]) + \
                               (vertical[0, 3] * gray_img[i - 2, j + 1]) + \
                               (vertical[0, 4] * gray_img[i - 2, j + 2]) + \
                               (vertical[1, 0] * gray_img[i - 1, j - 2]) + \
                               (vertical[1, 1] * gray_img[i - 1, j + 1]) + \
                               (vertical[1, 2] * gray_img[i - 1, j]) + \
                               (vertical[1, 3] * gray_img[i - 1, j + 1]) + \
                               (vertical[1, 4] * gray_img[i - 1, j + 2]) + \
                               (vertical[2, 0] * gray_img[i, j - 2]) + \
                               (vertical[2, 1] * gray_img[i, j - 1]) + \
                               (vertical[2, 2] * gray_img[i, j]) + \
                               (vertical[2, 3] * gray_img[i, j + 1]) + \
                               (vertical[2, 4] * gray_img[i, j + 2]) + \
                               (vertical[3, 0] * gray_img[i + 1, j - 2]) + \
                               (vertical[3, 1] * gray_img[i + 1, j - 1]) + \
                               (vertical[3, 2] * gray_img[i + 1, j]) + \
                               (vertical[3, 3] * gray_img[i + 1, j + 1]) + \
                               (vertical[3, 4] * gray_img[i + 1, j + 2]) + \
                               (vertical[4, 0] * gray_img[i + 2, j - 2]) + \
                               (vertical[4, 1] * gray_img[i + 2, j - 1]) + \
                               (vertical[4, 2] * gray_img[i + 2, j]) + \
                               (vertical[4, 3] * gray_img[i + 2, j + 1]) + \
                               (vertical[4, 4] * gray_img[i + 2, j + 2])

                newverticalImage[i, j] = abs(verticalGrad)
                # G=sqrt(Gx^2+Gy^2)
                mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
                newgradientImage[i, j] = mag
        return newgradientImage


