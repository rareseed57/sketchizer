import os

import numpy as np
import cv2
import vectorizer
import tests
import analyzer
import random as rng
from tkinter import *
import webbrowser


path = os.path.dirname(os.path.realpath(__file__))
img_list = os.listdir(path+'/input')
# img_list = ['logo.jpg', 'debug.png', 'bottiglia.png', 'colors.jpg']
img_name = 'input/'+img_list[0]
img_index = 0

brush_name = 'pencil'
brush_list = ['pencil', 'ink']
brush_index = 0

more_controls = False

n_run = 0

WIN_SIZE = 400

# default parameters
thresh1hue = 0
thresh2hue = 179
sample_step = 5
corner_thresh = 80
n_chances = 0
blur_ratio = 3
thresh1can = 100
thresh2can = 200
approx_factor = 0.3

# -----------
test_image = np.zeros((WIN_SIZE, WIN_SIZE, 3), np.uint8)
test_copy = test_image.copy()
isDrawing = False
isCrop = False
refPoints = [[-1, -1]]
xTemp, yTemp = 0, 0


def remap_coords(coords):
    global test_image
    size = test_image.shape[0], test_image.shape[1]
    max_size = max(size)
    min_size = min(size)
    new_coords = list(coords)

    zoom_factor = WIN_SIZE / max_size

    SHORT_WIN_SIZE = min_size * zoom_factor

    if max_size == size[0]:
        new_coords[0] = int((coords[0] - (WIN_SIZE - SHORT_WIN_SIZE) / 2) / zoom_factor)
        new_coords[1] = int(coords[1] / zoom_factor)
    else:
        new_coords[0] = int(coords[0] / zoom_factor)
        new_coords[1] = int((coords[1] - (WIN_SIZE - SHORT_WIN_SIZE) / 2) / zoom_factor)

    return new_coords


def click_to_crop(event, x, y, flags, param):
    global isCrop, refPoints, isDrawing, xTemp, yTemp, test_copy
    x, y = remap_coords((x, y))
    # left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        isDrawing = True
        if x < 1:
            x = 1
        elif x > test_image.shape[1]:
            x = test_image.shape[1] - 1
        if y < 1:
            y = 1
        elif y > test_image.shape[0]:
            y = test_image.shape[0] - 1
        refPoints = [[x, y]]
        xTemp, yTemp = x, y
    # mouse move
    elif event == cv2.EVENT_MOUSEMOVE and isDrawing:
        if x < 1:
            x = 1
        elif x > test_image.shape[1]:
            x = test_image.shape[1] - 1
        if y < 1:
            y = 1
        elif y > test_image.shape[0]:
            y = test_image.shape[0] - 1
        copy = test_image.copy()
        xTemp, yTemp = x, y
        cv2.rectangle(copy, tuple(refPoints[0]), (xTemp, yTemp), (255, 255, 255), 2)
        cv2.imshow("Test", resizeAR(copy))

    # left mouse was release
    elif event == cv2.EVENT_LBUTTONUP:
        if x == refPoints[0][0] and y == refPoints[0][1]:
            cv2.imshow("Test", resizeAR(test_image))
        if x < 1:
            x = 1
        elif x > test_image.shape[1]:
            x = test_image.shape[1] - 1
        if y < 1:
            y = 1
        elif y > test_image.shape[0]:
            y = test_image.shape[0] - 1
        refPoints.append([x, y])
        isDrawing = False
        isCrop = True

        # verification of start point to draw
        for i in (0, 1):
            if refPoints[0][i] > refPoints[1][i]:
                xTemp = refPoints[1][i]
                refPoints[1][i] = refPoints[0][i]
                refPoints[0][i] = xTemp
        cv2.rectangle(test_copy, tuple(refPoints[0]), tuple(refPoints[1]), (0, 255, 0), 1)

        cv2.imshow("Test", resizeAR(test_image[refPoints[0][1]:refPoints[1][1], refPoints[0][0]:refPoints[1][0]]))
    elif event == cv2.EVENT_RBUTTONDOWN:
        refPoints = [[-1, -1]]
        isCrop = False
        refresh()


def resizeAR(image, width=WIN_SIZE, height=None, inter=cv2.INTER_AREA):
    global WIN_SIZE
    background = np.zeros((WIN_SIZE, WIN_SIZE, 3), np.uint8)

    (h, w) = image.shape[:2]
    if h == 0 or w == 0:
        return image
    if max(h, w) == h:
        width = None
        height = WIN_SIZE
    else:
        height = None
        width = WIN_SIZE
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    image = cv2.resize(image, dim, interpolation=inter)
    start_roi_row = int(WIN_SIZE / 2 - image.shape[1] / 2)
    start_roi_col = int(WIN_SIZE / 2 - image.shape[0] / 2)
    end_roi_row = int(WIN_SIZE / 2 + image.shape[1] / 2)
    end_roi_col = int(WIN_SIZE / 2 + image.shape[0] / 2)
    background[start_roi_col:end_roi_col, start_roi_row:end_roi_row] = image
    return background


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print('Coords: x = ', x, ' y = ', y)


# For HSV: hue range is [0,179], saturation range is [0,255], and value range is [0,255].

def refresh():
    print('---------------------------------------------------------------------------')
    global n_run, thresh1hue, thresh2hue, sample_step, corner_thresh, n_chances, blur_ratio, thresh1can, thresh2can, approx_factor, test_image
    # Image reading and conversion to HSV color space to carry out a tresholding ###############
    img = cv2.imread(img_name)
    valid_crop = True
    for p in refPoints:
        if p[0] < 1:
            valid_crop = False
        elif p[0] > img.shape[1]:
            valid_crop = False
        if p[1] < 1:
            valid_crop = False
        elif p[1] > img.shape[0]:
            valid_crop = False
    if valid_crop:
        img = img[refPoints[0][1]:refPoints[1][1], refPoints[0][0]:refPoints[1][0]]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if more_controls:
        ### Getting and checking all the values of the trackbars into variables #################
        thresh1hue = cv2.getTrackbarPos('Hue L', 'Original')
        thresh2hue = cv2.getTrackbarPos('Hue H', 'Original')

        # thresh1sat = cv2.getTrackbarPos('Saturation 1', 'Original')
        # thresh2sat = cv2.getTrackbarPos('Saturation 2', 'Original')

        # thresh1val = cv2.getTrackbarPos('Value 1', 'Original')
        # thresh2val = cv2.getTrackbarPos('Value 2', 'Original')
        blur_ratio = round(cv2.getTrackbarPos('Blur ratio', 'Lines') * 0.06)

        if blur_ratio == 0:
            blur_ratio = 1

        thresh1can = cv2.getTrackbarPos('Canny L', 'Lines')
        thresh2can = cv2.getTrackbarPos('Canny H', 'Lines')

        ### Inverting the thresholds if inconsistent #################

        if thresh2hue < thresh1hue:
            tmp = thresh2hue
            thresh2hue = thresh1hue
            thresh1hue = tmp

        if thresh2can < thresh1can:
            tmp = thresh2can
            thresh2can = thresh1can
            thresh1can = tmp

    sample_step = cv2.getTrackbarPos('Sampling step', 'Test')
    corner_thresh = cv2.getTrackbarPos('Corner thresh', 'Test') / 1000
    n_chances = cv2.getTrackbarPos('#Chances', 'Test')
    approx_factor = cv2.getTrackbarPos('Approx factor', 'Test') / 1000
    if approx_factor > 1 or approx_factor < 0:
        approx_factor = 0.3

    mask = cv2.inRange(hsv, (thresh1hue, 0, 0), (thresh2hue, 255, 255))
    h, s, v = cv2.split(hsv)

    hsv = cv2.merge([h, s, np.multiply(v, -mask)])

    img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if more_controls:
        cv2.imshow('Original', resizeAR(img_bgr))

    img_bn = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(img_bn, (blur_ratio, blur_ratio))
    edges = cv2.Canny(blurred, thresh1can, thresh2can)

    lines_img, lines = analyzer.findlines(edges)
    lines_img = cv2.cvtColor(lines_img, cv2.COLOR_HSV2BGR)
    totalsum = 0
    lenght = len(lines)
    if lenght == 0:
        lenght = 1
    print('Number of lines: ' + str(lenght))
    for i in lines.values():
        totalsum += len(i)
    print('Average lenght: ' + str(totalsum / lenght) + '\n')

    closures = analyzer.check_closures(lines, 2)

    shapes = analyzer.detect_shapes(lines, closures, approx_factor=1 - approx_factor)

    test = tests.test_closures(lines_img.copy(), lines, closures)
    if more_controls:
        cv2.imshow("Lines", resizeAR(test))

    sampled = analyzer.sample(lines.copy(), sample_step)

    derivatives = analyzer.compute_derivatives(sampled)
    # tests.test_is_segment(_segment_map, _lines.copy(), _lines_img.copy())

    flexes = analyzer.find_flexes(derivatives, sampled, corner_thresh, closures, n_chances)
    test = tests.test_shapes(lines_img.copy(), lines, shapes.copy())
    test = tests.test_flexes(test, sampled.copy(), flexes.copy())

    # cv2.namedWindow("Test", cv2.WINDOW_FULLSCREEN)
    test_image = test
    cv2.imshow("Test", resizeAR(test))
    cv2.setMouseCallback("Test", click_to_crop)
    # print('sampled:' + str(sampled))
    # print('derivatives:' + str(derivatives))
    # print('segments:' + str(segment_map))
    # print('flexes:' + str(flexes))

    # vectorizer.vectorize_samples(edges.shape[0], edges.shape[1], sampled, filter=brush_name)
    vectorizer.vectorize_flexes(edges.shape[0], edges.shape[1], flexes, closures, shapes, filter=brush_name)
    tests.test_drawing()

    cv2.imwrite('output/linesoutput.png', lines_img)
    # if n_run == 1:
    #   webbrowser.open("file://" + path + "/output/animation.html")

    n_run += 1


def change_img():
    global img_index, img_name, img_list, refPoints, isCrop
    img_index += 1
    if img_index > len(img_list) - 1:
        img_index = 0
    img_name = 'input/' + img_list[img_index]
    print('Now using ' + img_name)
    btn_img.config(text=img_name)
    refPoints = [[-1, -1]]
    isCrop = False
    cv2.imshow("Test", resizeAR(test_image))
    refresh()


def change_brush():
    global brush_index, brush_name, brush_list
    brush_index += 1
    if brush_index > len(brush_list) - 1:
        brush_index = 0
    brush_name = brush_list[brush_index]
    print('Now using ' + brush_name + ' brush.')
    btn_brush.config(text=brush_name)
    refresh()


def change_controls():
    global more_controls
    more_controls = not more_controls
    if not more_controls:
        print('Now using less controls.')
        btn_controls.config(text='Off')
        cv2.destroyWindow("Original")
        cv2.destroyWindow("Lines")
    else:
        print('Now using more controls.')
        btn_controls.config(text='On')
        cv2.namedWindow("Original", cv2.WINDOW_FULLSCREEN)
        cv2.namedWindow("Lines", cv2.WINDOW_FULLSCREEN)
        create_trackbars()

    refresh()


def sketchize():
    refresh()
    webbrowser.open("file://" + path + "/output/animation.html")


def create_trackbars():
    # Trackbar creation to adjust tresholds and settings #########################################
    # Trackbars for tresholding
    cv2.createTrackbar('Hue L', 'Original', 0, 179, lambda x: refresh())
    cv2.createTrackbar('Hue H', 'Original', 179, 179, lambda x: refresh())
    # cv2.createTrackbar('Saturation 1', 'Original', 0, 255, lambda x: refresh())
    # cv2.createTrackbar('Saturation 2', 'Original', 255, 255, lambda x: refresh())
    # cv2.createTrackbar('Value 1', 'Original', 0, 255, lambda x: refresh())
    # cv2.createTrackbar('Value 2', 'Original', 255, 255, lambda x: refresh())

    # Trackbars to adjust image clean-up and edge detection filters (before starting the actual algorithm)

    cv2.createTrackbar('Blur ratio', 'Lines', 50, 100, lambda x: refresh())
    cv2.createTrackbar('Canny L', 'Lines', 100, 500, lambda x: refresh())
    cv2.createTrackbar('Canny H', 'Lines', 200, 500, lambda x: refresh())

    # Trackbars to adjust the analyzer settings

    cv2.createTrackbar('Sampling step', 'Test', 5, 100, lambda x: refresh())
    cv2.createTrackbar('Derivative thresh', 'is_segment test', 1, 1000, lambda: refresh())

    cv2.createTrackbar('Corner thresh', 'Test', 80, 1000, lambda x: refresh())
    cv2.createTrackbar('#Chances', 'Test', 0, 10, lambda x: refresh())

    cv2.createTrackbar('Approx factor', 'Test', 30, 1000, lambda x: refresh())


if __name__ == '__main__':
    cv2.namedWindow("Original", cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("Lines", cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("Test", cv2.WINDOW_FULLSCREEN)

    refresh()

    create_trackbars()

    refresh()

    cv2.destroyWindow("Original")
    cv2.destroyWindow("Lines")

    root = Tk()
    root.geometry("250x350")

    Label(root, text=" ").pack()

    label_img = Label(root, text="SELECT IMAGE")
    label_img.pack()

    btn_img = Button(root, text=img_name, padx=5, pady=5, command=change_img)
    btn_img.pack()

    Label(root, text=" ").pack()

    label_brush = Label(root, text='SELECT BRUSH')
    label_brush.pack()

    btn_brush = Button(root, text=brush_name, padx=5, pady=5, command=change_brush)
    btn_brush.pack()

    Label(root, text=" ").pack()

    label_brush = Label(root, text='SHOW MORE CONTROLS')
    label_brush.pack()
    btn_controls = Button(root, text='off', padx=5, pady=5, command=change_controls)
    btn_controls.pack()

    Label(root, text=" ").pack()

    label_result = Label(root, text='Result')
    label_result.pack()
    btn_result = Button(root, text='Sketch it!', padx=5, pady=5, command=sketchize)
    btn_result.pack()

    root.mainloop()

    cv2.waitKey(0)
