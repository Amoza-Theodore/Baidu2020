import cv2
import numpy as np

# 黄色
lower_hsv_yellow = np.array([7, 37, 46])
upper_hsv_yellow = np.array([40, 255, 255])

# 锥桶蓝色
lower_hsv_blue = np.array([99, 160, 145])
upper_hsv_blue = np.array([102, 220, 170])

# 红色
lower_hsv_red = np.array([156, 120, 120])
upper_hsv_red = np.array([180, 255, 255])

def mouseColor(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("HSV is", hsv[y, x])

img = cv2.imread('picture.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_yellow = cv2.inRange(hsv, lowerb=lower_hsv_yellow, upperb=upper_hsv_yellow)
mask_blue = cv2.inRange(hsv, lowerb=lower_hsv_blue, upperb=upper_hsv_blue)
mask_red = cv2.inRange(hsv, lowerb=lower_hsv_red, upperb=upper_hsv_red)
mask = mask_blue + mask_red + mask_yellow

# cv2.namedWindow("Color Picker")
# cv2.setMouseCallback("Color Picker", mouseColor)
# cv2.imshow('Color Picker', img)

cv2.imshow('mask', mask)

if cv2.waitKey(0):
    cv2.destroyAllWindows()

# np.savetxt("img.txt", mask.astype('int'), fmt='%d')

