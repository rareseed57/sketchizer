import numpy as np
import cv2
import vectorizer
import tests
import analyzer


# For HSV: hue range is [0,179], saturation range is [0,255], and value range is [0,255].

def refresh(hsv):

    ### Getting and checking all the values of the trackbars into variables #################

    thresh1hue = cv2.getTrackbarPos('Hue L', 'Original')
    thresh2hue = cv2.getTrackbarPos('Hue H', 'Original')

    # thresh1sat = cv2.getTrackbarPos('Saturation 1', title_window)
    # thresh2sat = cv2.getTrackbarPos('Saturation 2', title_window)

    # thresh1val = cv2.getTrackbarPos('Value 1', title_window)
    # thresh2val = cv2.getTrackbarPos('Value 2', title_window)

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
    # mask = cv2.inRange(hsv, (thresh1hue, thresh1sat, thresh1val), (thresh2hue, thresh2sat, thresh2val))
    h, s, v = cv2.split(hsv)

    hsv = cv2.merge([h, s, np.multiply(v, -mask)])

    img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Original', img_bgr)

    img_bn = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(img_bn, (blur_ratio, blur_ratio))
    edges = cv2.Canny(blurred, thresh1can, thresh2can)

    lines_img, lines = analyzer.findlines(edges)
    lines_img_rough = lines_img.copy()
    lines_img, lines = analyzer.mergelines(lines, lines_img)
    print('lines:' + str(lines))

    cv2.imshow("Lines", lines_img_rough)

    sampled = analyzer.sample(lines.copy(), sample_step)

    segment_map, derivatives = analyzer.is_segment(sampled, der_thresh / 1000, img.shape[1], lines_img.copy())
    # tests.test_is_segment(_segment_map, _lines.copy(), _lines_img.copy())

    flexes = analyzer.find_flex(derivatives, sampled, flex_thresh / 1000000, n_chances)

    tests.test(lines_img.copy(), sampled, flexes.copy())
    # print('sampled:' + str(sampled))
    # print('derivatives:' + str(derivatives))
    # print('segments:' + str(segment_map))
    # print('flexes:' + str(flexes))

    vectorizer.vectorize_samples(edges.shape[0], edges.shape[1], sampled)
    print('\n')


def on_trackbar(val, hsv):
    refresh(hsv)


if __name__ == '__main__':
    img_name = 'input/bottiglia.PNG'
    #img_name = 'input/colors.jpg'
    img_name = 'input/logo.jpg'

    # Image reading and conversion to HSV color spasce to carry out a tresholding ###############

    img = cv2.imread(img_name)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Trackbar creation to adjust tresholds and settings #########################################

    refresh(img_hsv.copy())

    # Trackbars for tresholding

    cv2.createTrackbar('Hue L', 'Original', 0, 179, lambda x: on_trackbar(x, img_hsv.copy()))
    cv2.createTrackbar('Hue H', 'Original', 179, 179, lambda x: on_trackbar(x, img_hsv.copy()))
    # cv2.createTrackbar('Saturation 1', title_window, 0, 255, lambda x: on_trackbar(x, img_hsv.copy()))
    # cv2.createTrackbar('Saturation 2', title_window, 255, 255, lambda x: on_trackbar(x, img_hsv.copy()))
    # cv2.createTrackbar('Value 1', title_window, 0, 255, lambda x: on_trackbar(x, img_hsv.copy()))
    # cv2.createTrackbar('Value 2', title_window, 255, 255, lambda x: on_trackbar(x, img_hsv.copy()))

    # Trackbars to adjust image clean-up and edge detection filters (before starting the actual algorithm)

    cv2.createTrackbar('Blur ratio', 'Lines', 50, 100, lambda x: on_trackbar(x, img_hsv.copy()))
    cv2.createTrackbar('Canny L', 'Lines', 100, 500, lambda x: on_trackbar(x, img_hsv.copy()))
    cv2.createTrackbar('Canny H', 'Lines', 200, 500, lambda x: on_trackbar(x, img_hsv.copy()))

    # Trackbars to adjust the analyzer settings

    cv2.createTrackbar('Sampling step', 'Test', 10, 100, lambda x: on_trackbar(x, img_hsv.copy()))
    cv2.createTrackbar('Derivative thresh', 'is_segment test', 1, 1000, lambda x: on_trackbar(x, img_hsv.copy()))

    cv2.createTrackbar('Flex thresh', 'Test', 300, 1000, lambda x: on_trackbar(x, img_hsv.copy()))
    cv2.createTrackbar('#Chances', 'Test', 2, 10, lambda x: on_trackbar(x, img_hsv.copy()))

    refresh(img_hsv.copy())

    cv2.waitKey(0)
