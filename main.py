from Sobel_3x3 import Sobel_3X3
from Sobel_5x5 import Sobel_5x5
from Prewitt_3x3 import Prewitt_3x3
from Prewitt_5x5 import Prewitt_5x5
from canny_3x3 import Canny_3X3
from Canny_5X5 import Canny_5X5
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

#sobele gittikten sonra ön işlem olarak orada da gauss un uygulanması gerekiyor

"""image = cv2.imread('deneme.jpg')
blurred = cv2.GaussianBlur(image, ksize = (5,5), sigmaX = 0)
blurred_img = Image.fromarray(blurred)
blurred_img.save('blurred_img.jpg')"""

#CANNY
"""canny = Canny_3X3('blurred_img.jpg')
img1=canny.canny_3x3()"""



"""canny5=Canny_5X5('blurred_img.jpg')
img2=canny5.canny_5x5()"""


"""sobel = Sobel_3X3('blurred_img.jpg')
img3 = sobel.sobel_3x3()"""


"""sobel5 = Sobel_5x5('blurred_img.jpg')
img4 = sobel5.sobel_5x5()"""


"""prewitt = Prewitt_3x3('blurred_img.jpg')
img5 = prewitt.prewitt_3x3()"""


prewitt5 = Prewitt_5x5('blurred_img.jpg')
img6 = prewitt5.prewitt_5x5()



plt.figure()


plt.imshow(img6, cmap='gray')
plt.imsave('prewitt_5x5.png', img6, cmap='gray', format='png')
plt.show()











