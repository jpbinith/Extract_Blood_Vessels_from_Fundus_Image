import cv2
import numpy as np


def apply_threshold_with_denoising(image):
    image = cv2.adaptiveThreshold(image, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    image = cv2.fastNlMeansDenoising(image, 1.5, 5, 5)
    return image


def delete_small_components(image, size):
    _, blackAndWhite = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1]  # get CC_STAT_AREA component
    image = np.zeros(labels.shape, np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 150:  # filter small dotted regions
            image[labels == i + 1] = 255

    return cv2.bitwise_not(image)


def kernel(num1, num2):
    return np.ones((num1, num2), np.uint8)


def resize(img):
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def get_large_vessels(image):
    struct_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 4))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, struct_kernel, iterations=1)

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 250:
            cv2.drawContours(opening, [c], -1, (0, 0, 0), -1)

    return opening


def get_small_vessels(both, large):
    large = cv2.dilate(large, kernel(3, 3), iterations=5)
    subtract = cv2.subtract(both, large)
    return subtract


def remove_background(image, mask):
    image = cv2.bitwise_and(cv2.bitwise_not(image), cv2.bitwise_not(image), mask=mask)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel(2, 2))
    return image


img = cv2.imread('Fundus Image.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

l_b = np.array([0, 0, 30])
u_b = np.array([255, 255, 255])

mask = cv2.inRange(hsv, l_b, u_b)
mask = cv2.erode(mask, kernel(2, 2), iterations=5)

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold = apply_threshold_with_denoising(grayscale)

kernel22 = cv2.dilate(threshold, kernel(2, 2), iterations=2)
# cv2.imshow('dilation with kernel (2, 2)', kernel22)
remove_small1kernel22 = delete_small_components(kernel22, 5)
# cv2.imshow('dilation with kernel (2, 2) remove small', remove_small1kernel22)

dilation = cv2.dilate(threshold, kernel(2, 1), iterations=2)
# cv2.imshow('dilation with kernel (2, 1)', dilation)
remove_small1 = delete_small_components(dilation, 150)
# cv2.imshow('dilation with kernel (2, 1) remove small', remove_small1)

dilation = cv2.dilate(threshold, kernel(1, 2), iterations=2)
# cv2.imshow('dilation with kernel (1, 2)', dilation)
remove_small2 = delete_small_components(dilation, 150)
# cv2.imshow('dilation with kernel (1, 2) remove small', remove_small2)

merge = cv2.addWeighted(remove_small1, 0.5, remove_small2, 0.5, 0)
# cv2.imshow('merge', merge)
threshold_merge = apply_threshold_with_denoising(merge)
# cv2.imshow('threshold merge', threshold_merge)

remove_small3 = delete_small_components(threshold_merge, 150)
# cv2.imshow('merge remove small', remove_small3)

large_vessels = get_large_vessels(remove_background(remove_small3, mask))
cv2.imshow('Large blood vessles', large_vessels)
cv2.imwrite('Large blood vessles.jpg', large_vessels)

small_vessels = get_small_vessels(remove_background(remove_small3, mask), large_vessels)
cv2.imshow('small blood vessles', small_vessels)
cv2.imwrite('small blood vessles.jpg', small_vessels)

cv2.waitKey(0)
cv2.destroyAllWindows()