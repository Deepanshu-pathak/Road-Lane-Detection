import cv2
import numpy as np
# import matplotlib.pyplot as plt  #(used to know region of interest)

#function for canny method
def canny_method(img):
    # gray scaling image for fewer comparisons
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # reducing noise from image using gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # applying canny method
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    #bitwise AND between canny image and mask image
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def make_coordinates(img,line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(img,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line=make_coordinates(img, left_fit_average)
    right_line=make_coordinates(img, right_fit_average)
    return np.array([left_line, right_line])

#function to display lines
def display_lines(img,lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for  x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1, y1), (x2, y2),(0, 255, 0), 10)
    return line_img

#lane detection algorithm on image or single frame
# image = cv2.imread('test_image.jpg')
# lane_img = np.copy(image)
# canny_img = canny_method(lane_img)
# cropped_image = region_of_interest(canny_img)
# #hough transform
# lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_img,lines)
# line_img= display_lines(lane_img,averaged_lines)
# combo_image = cv2.addWeighted(lane_img,0.8,line_img,1,1)
# cv2.imshow("result",combo_image)
# cv2.waitKey(0)

#video capture and algorithm on video
cap = cv2.VideoCapture("test2.mp4")
while cap.isOpened():
    _, frame = cap.read()
    canny_img = canny_method(frame)
    cropped_image = region_of_interest(canny_img)

    # hough transform
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()