import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve



"""canny steps:
    gray scale conversion
    noise reduce
    gradient calculation(sobel)
    non-maximum supression
    Double Thresholding and hysteresis"""



#GRAY SCALE CONVERSION




class Canny_3X3:
    def __init__(self,img_path):
        self.img = np.array(Image.open(img_path)).astype(np.uint8)

    def gaussian_kernel(self, size=5, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        return g

    def sobelGradient(self,gray_img):

        h, w = gray_img.shape
        horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # s2
        vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # s1
        newhorizontalImage = np.zeros((h, w))
        newverticalImage = np.zeros((h, w))
        newgradientImage = np.zeros((h, w))
        newThetaMat = np.zeros((h, w))

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
                # G=sqrt(Gx^2+Gy^2)
                mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
                # mag=abs(verticalGrad)+abs(horizontalGrad)
                # mag=np.hypot(verticalGrad,horizontalGrad)
                theta = np.arctan2(verticalGrad, horizontalGrad)
                newgradientImage[i, j] = mag
                newThetaMat[i, j] = theta
        return (newgradientImage, newThetaMat)

    def non_maximum_supression(self,img,thetaMat):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = thetaMat * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0


                except IndexError as e:
                    pass

        return Z

    def threshold(self,img):
        highThreshold = img.max() * 0.15;
        lowThreshold = highThreshold * 0.05;

        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(75)
        strong = np.int32(255)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        return (res)

    def hysteresis(self,img):
        M, N = img.shape
        weak = 75
        strong = 255

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (img[i, j] == weak):
                    try:
                        if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                                or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                                or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                        img[i - 1, j + 1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img


    def canny_3x3(self):
        gray_img = np.round(0.299 * self.img[:, :, 0] +
                            0.587 * self.img[:, :, 1] +
                            0.114 * self.img[:, :, 2]).astype(np.uint8)
        smoothed_img=convolve(gray_img,self.gaussian_kernel())
        gradient_img,theta_mat=self.sobelGradient(smoothed_img)
        nonMaxImg=self.non_maximum_supression(gradient_img,theta_mat)
        thresholdImg=self.threshold(nonMaxImg)
        hysteresisImage=self.hysteresis(thresholdImg)
        return hysteresisImage








