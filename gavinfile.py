import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('brick photos\Wallpaper-Kemra-ClassicRedBrick-NEW-1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')

blur = cv2.GaussianBlur(gray, (23, 23), 0)
plt.imshow(blur, cmap='gray')

canny = cv2.Canny(blur, 30,94, 3)
plt.imshow(canny, cmap='gray')
cv2.imwrite('canny.jpg', canny)

dilated = cv2.dilate(canny, (1, 1), iterations=2)
plt.imshow(dilated, cmap='gray')
cv2.imwrite('dilated.jpg', dilated)

(cnt, hierarchy) = cv2.findContours(
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
 
plt.imshow(rgb)
cv2.imshow('rgb_test1.jpg',rgb)
cv2.waitKey(0)