from Sobel_3x3 import Sobel_3X3
from Canny_3x3 import Canny_3X3
from Canny_5X5 import Canny_5X5
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#sobele gittikten sonra ön işlem olarak orada da gauss un uygulanması gerekiyor

#SOBEL 3X3
"""img_path = 'spider.png'
sobel3 = Sobel_3X3(img_path)
img = sobel3.sobel_3x3()"""


#CANNY
canny = Canny_3X3('deneme.jpg')
img=canny.canny_3x3()

"""canny3=Canny_5X5('deneme.jpg')
img=canny3.canny_5x5()"""







plt.figure()
plt.title('dancing-spider-sobel.png')
# plt.imsave('sobel_3x3.png', sobel3, cmap='gray', format='png')
plt.imshow(img, cmap='gray')
plt.show()
