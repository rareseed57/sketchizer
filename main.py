import numpy as np
import cv2
import vectorizer
import tests
import analyzer
import random as rng
from tkinter import *
import webbrowser

img_name = 'input/logo.jpg'
img_list = ['logo.jpg', 'bottiglia.PNG', 'colors.jpg', 'debug.jpg', 'debug2.png']
img_index = 0

brush_name = 'pencil'
brush_list = ['pencil', 'ink']
brush_index = 0

n_run = 0


def resizeAR(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


# For HSV: hue range is [0,179], saturation range is [0,255], and value range is [0,255].

def refresh():
    global n_run
    # Image reading and conversion to HSV color spasce to carry out a tresholding ###############
    img = cv2.imread(img_name)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ### Getting and checking all the values of the trackbars into variables #################
    thresh1hue = cv2.getTrackbarPos('Hue L', 'Original')
    thresh2hue = cv2.getTrackbarPos('Hue H', 'Original')

    # thresh1sat = cv2.getTrackbarPos('Saturation 1', 'Original')
    # thresh2sat = cv2.getTrackbarPos('Saturation 2', 'Original')

    # thresh1val = cv2.getTrackbarPos('Value 1', 'Original')
    # thresh2val = cv2.getTrackbarPos('Value 2', 'Original')

    sample_step = cv2.getTrackbarPos('Sampling step', 'Test')
    der_thresh = cv2.getTrackbarPos('Derivative thresh', 'is_segment test')
    flex_thresh = cv2.getTrackbarPos('Flex thresh', 'Test')

    n_chances = cv2.getTrackbarPos('#Chances', 'Test')

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

    mask = cv2.inRange(hsv, (thresh1hue, 0, 0), (thresh2hue, 255, 255))
    h, s, v = cv2.split(hsv)

    hsv = cv2.merge([h, s, np.multiply(v, -mask)])

    img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Original', resizeAR(img_bgr, height=500))

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

    cv2.imshow("Lines", resizeAR(lines_img, height=500))

    sampled = analyzer.sample(lines.copy(), sample_step)

    segment_map, derivatives = analyzer.is_segment(sampled, der_thresh / 1000, img.shape[1], lines_img.copy())
    # tests.test_is_segment(_segment_map, _lines.copy(), _lines_img.copy())

    flexes = analyzer.find_flex(derivatives, sampled, flex_thresh / 1000000, n_chances)

    cv2.imshow("Test", resizeAR(tests.test(lines_img.copy(), sampled, flexes.copy()), height=500))
    # print('sampled:' + str(sampled))
    # print('derivatives:' + str(derivatives))
    # print('segments:' + str(segment_map))
    # print('flexes:' + str(flexes))

    vectorizer.vectorize_samples(edges.shape[0], edges.shape[1], sampled, filter=brush_name)
    tests.test_drawing()

    cv2.imwrite('output/linesoutput.png', lines_img)
    if n_run == 1:
        webbrowser.open("file://C:/Users/rares/PycharmProjects/CVProject/output/animation.html")
    n_run += 1


def change_img():
    global img_index, img_name, img_list
    img_index += 1
    if img_index > len(img_list) - 1:
        img_index = 0
    img_name = 'input/' + img_list[img_index]
    print('Now using ' + img_name)
    btn_img.config(text=img_name)
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


if __name__ == '__main__':
    # Trackbar creation to adjust tresholds and settings #########################################

    refresh()

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

    cv2.createTrackbar('Sampling step', 'Test', 10, 100, lambda x: refresh())
    cv2.createTrackbar('Derivative thresh', 'is_segment test', 1, 1000, lambda x: refresh())

    cv2.createTrackbar('Flex thresh', 'Test', 300, 1000, lambda x: refresh())
    cv2.createTrackbar('#Chances', 'Test', 2, 10, lambda x: refresh())

    refresh()

    root = Tk()
    root.geometry("250x180")

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

    root.mainloop()

    cv2.waitKey(0)
