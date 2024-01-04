import cv2
import matplotlib.pyplot as plt

# Open the image
img = cv2.imread('deneme.jpg')

# Apply Canny
blurred_img=gf = cv2.GaussianBlur(img, ksize = (5,5), sigmaX = 0)
edges = cv2.Canny(blurred_img, 100, 200, 3, L2gradient=True)

plt.figure()
plt.title('Spider')
plt.imsave('dancing-spider-canny.png', edges, cmap='gray', format='png')
plt.imshow(edges, cmap='gray')
plt.show()