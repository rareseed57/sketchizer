import numpy as np
import cv2 as cv2
import svgwrite


# For HSV: hue range is [0,179], saturation range is [0,255], and value range is [0,255].

def vectorize(img_x, img_y, _sampled):
    svg = svgwrite.Drawing(filename="output/Vectorized.svg", size=(str(img_y) + 'px', str(img_x) + 'px'))
    path = ''

    marker = svg.marker(insert=(1, 1), size=(4, 4), orient='auto')
    marker.add(svg.circle((1, 1), r=0.5, fill='black'))
    svg.defs.add(marker)

    for k in _sampled.keys():
        if len(_sampled[k]) > 2:
            count = 0
            for node in _sampled[k]:
                if count == 0:
                    path = 'M ' + str(node[1]) + ',' + str(node[0]) + ' ' + str(node[1]) + ',' + str(node[0]) + ' \n'
                elif count % 2 != 0 and count != len(_sampled[k]) - 1:
                    path = path + 'S ' + str(node[1]) + ',' + str(node[0])
                elif count % 2 == 0:
                    path = path + ' ' + str(node[1]) + ',' + str(node[0]) + '\n'
                else:
                    path = path + 'S ' + str(node[1]) + ',' + str(node[0]) + ' ' + str(node[1]) + ',' + str(
                        node[0]) + ' \n'
                count = count + 1

            # print('Adding\n' + path + '\n')

            path = svg.path(d=path,
                            stroke="#000",
                            fill="none",
                            stroke_width=3)

            path.set_markers((marker, False, marker))

            svg.add(path)
    svg.save()
    return svg


def on_trackbar(val, hsv):
    thresh1hue = cv2.getTrackbarPos('Hue L', 'Original')
    thresh2hue = cv2.getTrackbarPos('Hue H', 'Original')

    # thresh1sat = cv2.getTrackbarPos('Saturation 1', title_window)
    # thresh2sat = cv2.getTrackbarPos('Saturation 2', title_window)

    # thresh1val = cv2.getTrackbarPos('Value 1', title_window)
    # thresh2val = cv2.getTrackbarPos('Value 2', title_window)

    sample_step = cv2.getTrackbarPos('Sampling step', 'Test')
    der_thresh = cv2.getTrackbarPos('Derivative thresh', 'is_segment test')
    flex_thresh = cv2.getTrackbarPos('Flex thresh', 'Test')

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
    # cv2.imshow("Canny", _edges)

    _lines_img, _lines = findlines(_edges)
    print(_lines)

    cv2.imshow("Lines", _lines_img)

    _sampled = sample(_lines.copy(), sample_step)

    _segment_map, _derivatives = is_segment(_sampled, der_thresh / 1000, img.shape[1], _lines_img.copy())
    # test_is_segment(_segment_map, _lines.copy(), _lines_img.copy())

    _flexes = find_flex(_derivatives, flex_thresh / 1000)

    test(_lines_img.copy(), _sampled, _flexes.copy())
    print('sampled:' + str(_sampled))
    print('derivatives:' + str(_derivatives))
    print('segments:' + str(_segment_map))
    print('flexes:' + str(_flexes))

    vectorize(_edges.shape[0], _edges.shape[1], _sampled)
    # x = {}
    # y = {}

    # for k in _sampled.keys():
    #     x[k] = []
    #     y[k] = []
    #     if len(_sampled[k]) > 2:
    #         for node in _sampled[k]:
    #             y[k].append(-node[0])
    #             x[k].append(node[1])
    #         f = interp1d(x[k], y[k])
    #         plt.plot(x[k], f(x[k]), '-')

    # plt.savefig('C:/Users/rares/PycharmProjects/CVProject/graph.svg')
    # plt.show()

    print('\n')


def checkandappend(_r, _c, _r2, _c2, _lines, _lines_list):
    if _r2 < 0 or _r2 >= _lines.shape[0] or _c2 < 0 or _c2 >= _lines.shape[1]:
        return False
    if _lines[_r2, _c2][0] != 0:
        # print(_lines[_r2][_c2])
        index = str(_lines[_r2][_c2][0]) + str(_lines[_r2][_c2][1])
        if _r2 == _lines_list[index][-1][0] and _c2 == _lines_list[index][-1][1]:
            _lines[_r][_c] = _lines[_r2][_c2]
            _lines_list[index].append((_r, _c))
            return True
    return False


def findlines(_edges):
    current_hue = 1
    current_sat = 255
    lines_dict = {}
    _lines_img = np.zeros((_edges.shape[0], _edges.shape[1], 3), np.uint8)

    _lines_img = cv2.cvtColor(_lines_img, cv2.COLOR_RGB2HSV)

    for r in range(0, _edges.shape[0]):
        for c in range(0, _edges.shape[1]):
            if current_hue > 179:
                current_sat = current_sat - 40
                current_hue = 1
            if _edges[r, c]:
                if not checkandappend(r, c, r, c - 1, _lines_img, lines_dict):
                    if not checkandappend(r, c, r - 1, c - 1, _lines_img, lines_dict):
                        if not checkandappend(r, c, r - 1, c, _lines_img, lines_dict):
                            if not checkandappend(r, c, r - 1, c + 1, _lines_img, lines_dict):
                                if not checkandappend(r, c, r, c + 1, _lines_img, lines_dict):
                                    if not checkandappend(r, c, r + 1, c + 1, _lines_img, lines_dict):
                                        if not checkandappend(r, c, r + 1, c, _lines_img, lines_dict):
                                            if not checkandappend(r, c, r + 1, c - 1, _lines_img, lines_dict):
                                                # Create new line
                                                _lines_img[r, c] = [current_hue, current_sat, 255]
                                                lines_dict[str(current_hue) + str(current_sat)] = [(r, c)]
                                                current_hue = current_hue + 1

    return cv2.cvtColor(_lines_img, cv2.COLOR_HSV2BGR), lines_dict


def sample(_lines, _step):
    if _step == 0:
        _step = 1
    samples = {}
    for line_key in _lines.keys():
        count = 0
        for pixel in _lines[line_key]:
            last = count == len(_lines[line_key]) - 1
            if count % _step == 0 or last:
                if count == 0:
                    samples[line_key] = []
                samples[line_key].append(pixel)
            count = count + 1
        if len(samples[line_key]) < 3:
            del samples[line_key]
    return samples


def find_flex(_derivatives, flex_thresh):
    rising = False
    _flexes = {}
    for line_key in _derivatives:
        _flexes[line_key] = []
        count = 1
        ders = _derivatives[line_key]
        for der in ders:
            if count <= len(ders) - 1:
                if abs(ders[count][1] - der[1]) > flex_thresh:
                    if not rising:
                        _flexes[line_key].append(ders[count][0])
                        rising = True
                else:
                    if rising:
                        _flexes[line_key].append(ders[count][0])
                        rising = False
            count = count + 1
        if len(_flexes[line_key]) == 0:
            del _flexes[line_key]
    return _flexes


def test(_lines_img, _sampled, _flexes):
    for line_key in _sampled.keys():
        for pixel in _sampled[line_key]:
            cv2.circle(_lines_img, (pixel[1], pixel[0]), 1, (255, 255, 255), 1)

    for line_key in _flexes.keys():
        for pixel in _flexes[line_key]:
            cv2.circle(_lines_img, (pixel[1], pixel[0]), 1, (0, 255, 0), 2)

    cv2.imshow("Test", _lines_img)


def is_segment(_sampled, dev_thresh, _img_height, _lines_img):
    # img height should be the maximum derivative possible
    _derivatives = {}
    for line_key in _sampled.keys():
        if len(_sampled[line_key]) > 1:
            count = 1
            _derivatives[line_key] = []
        else:
            continue
        for der in _sampled[line_key]:
            delta_x = der[0] - _sampled[line_key][count - 2][0]
            if delta_x == 0:
                _derivatives[line_key].append(((_sampled[line_key][count - 2]), 1))
            else:
                # normalized derivative = ( Δy / Δx ) / max_derivative
                _derivatives[line_key].append(((_sampled[line_key][count - 2]),
                                               (der[1] - _sampled[line_key][count - 2][1] / delta_x) / _img_height))
            count = count + 1

    _segment_map = {}

    for line_key in _derivatives.keys():
        delta_x = _sampled[line_key][-1][0] - _sampled[line_key][0][0]
        if delta_x == 0:
            overall_derivative = 1
        else:
            overall_derivative = (_sampled[line_key][-1][1] - _sampled[line_key][0][1] / delta_x) / _img_height
        count = 0
        for der in _derivatives[line_key]:
            delta_der = abs((der[1] - overall_derivative))
            if delta_der > dev_thresh:
                _segment_map[line_key] = (overall_derivative, False)
                continue
            elif count == len(_derivatives[line_key][1]) - 1:
                _segment_map[line_key] = (overall_derivative, True)
            count = count + 1
    return _segment_map, _derivatives


def test_is_segment(_segment_map, _lines, _lines_img):
    for k in _segment_map.keys():
        if _segment_map[k][1]:
            if k in _lines.keys():
                for i in _lines[k]:
                    _lines_img[i] = (255, 255, 255)

    cv2.imshow("is_segment test", _lines_img)


if __name__ == '__main__':
    img_name = 'input/bottiglia.PNG'
    # img_name = 'input/colors.jpg'
    img_name = 'input/logo.jpg'
    img_name = 'input/logo2.jpeg'

    img = cv2.imread(img_name)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bn = cv2.imread(img_name, 0)
    blurred = cv2.blur(img_bn, (3, 3))
    edges = cv2.Canny(blurred, 100, 200)

    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite('output/edges_' + img_name, edges)

    lines_img, lines = findlines(edges)

    cv2.imwrite('output/lines_' + img_name, lines_img)
    cv2.imshow('output/Original', img)
    # cv2.imshow("output/Canny", edges)
    cv2.imshow("output/Lines", lines_img)

    sampled = sample(lines, 10)

    segment_map, derivatives = is_segment(sampled.copy(), 0.3, img.shape[1], lines_img.copy())
    # test_is_segment(segment_map, lines, lines_img.copy())

    flexes = find_flex(derivatives, 0.3)

    test(lines_img.copy(), sampled.copy(), flexes.copy())
    print('sampled:' + str(sampled))
    print('derivatives:' + str(derivatives))
    print('segments:' + str(segment_map))
    print('flexes:' + str(flexes))

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

    cv2.waitKey(0)
