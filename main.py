import numpy as np
import cv2
import vectorizer
import tests
import analyzer


# For HSV: hue range is [0,179], saturation range is [0,255], and value range is [0,255].

def refresh(hsv):
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

    _img_bn = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _blurred = cv2.blur(_img_bn, (blur_ratio, blur_ratio))
    _edges = cv2.Canny(_blurred, thresh1can, thresh2can)

    _lines_img, _lines = analyzer.findlines(_edges)
    print(_lines)

    cv2.imshow("Lines", _lines_img)

    _sampled = analyzer.sample(_lines.copy(), sample_step)

    _segment_map, _derivatives = analyzer.is_segment(_sampled, der_thresh / 1000, img.shape[1], _lines_img.copy())
    # tests.test_is_segment(_segment_map, _lines.copy(), _lines_img.copy())

    _flexes = analyzer.find_flex(_derivatives, flex_thresh / 1000, n_chances)

    tests.test(_lines_img.copy(), _sampled, _flexes.copy())
    print('sampled:' + str(_sampled))
    print('derivatives:' + str(_derivatives))
    print('segments:' + str(_segment_map))
    print('flexes:' + str(_flexes))

    vectorizer.vectorize_samples(_edges.shape[0], _edges.shape[1], _sampled)
    print('\n')


def on_trackbar(val, hsv):
    refresh(hsv)


if __name__ == '__main__':
    img_name = 'input/bottiglia.PNG'
    # img_name = 'input/colors.jpg'
    # img_name = 'input/logo.jpg'

    # Image reading and conversion to HSV color spasce to carry out a tresholding ###

    img = cv2.imread(img_name)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    refresh(img_hsv.copy())

    cv2.createTrackbar('Hue L', 'Original', 0, 179, lambda x: on_trackbar(x, img_hsv.copy()))
    cv2.createTrackbar('Hue H', 'Original', 179, 179, lambda x: on_trackbar(x, img_hsv.copy()))
    # cv2.createTrackbar('Saturation 1', title_window, 0, 255, lambda x: on_trackbar(x, img_hsv.copy()))
    # cv2.createTrackbar('Saturation 2', title_window, 255, 255, lambda x: on_trackbar(x, img_hsv.copy()))
    # cv2.createTrackbar('Value 1', title_window, 0, 255, lambda x: on_trackbar(x, img_hsv.copy()))
    # cv2.createTrackbar('Value 2', title_window, 255, 255, lambda x: on_trackbar(x, img_hsv.copy()))

    cv2.createTrackbar('Blur ratio', 'Lines', 50, 100, lambda x: on_trackbar(x, img_hsv.copy()))
    cv2.createTrackbar('Canny L', 'Lines', 100, 500, lambda x: on_trackbar(x, img_hsv.copy()))
    cv2.createTrackbar('Canny H', 'Lines', 200, 500, lambda x: on_trackbar(x, img_hsv.copy()))

    cv2.createTrackbar('Sampling step', 'Test', 10, 100, lambda x: on_trackbar(x, img_hsv.copy()))
    cv2.createTrackbar('Derivative thresh', 'is_segment test', 1, 1000, lambda x: on_trackbar(x, img_hsv.copy()))

    cv2.createTrackbar('Flex thresh', 'Test', 300, 1000, lambda x: on_trackbar(x, img_hsv.copy()))
    cv2.createTrackbar('#Chances', 'Test', 2, 10, lambda x: on_trackbar(x, img_hsv.copy()))

    refresh(img_hsv.copy())

    cv2.waitKey(0)
