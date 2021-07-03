import cv2
import numpy as np


def checkandappend(r, c, r2, c2, lines, lines_list):
    if r2 < 0 or r2 >= lines.shape[0] or c2 < 0 or c2 >= lines.shape[1]:
        return False
    if lines[r2, c2][0] != 0:
        # print(_lines[_r2][_c2])
        index = str(lines[r2][c2][0]) + str(lines[r2][c2][1])
        if r2 == lines_list[index][-1][0] and c2 == lines_list[index][-1][1]:
            lines[r][c] = lines[r2][c2]
            lines_list[index].append((r, c))
            return True
    return False


def findlines(edges):
    current_hue = 1
    current_sat = 255
    lines_dict = {}
    lines_img = np.zeros((edges.shape[0], edges.shape[1], 3), np.uint8)

    lines_img = cv2.cvtColor(lines_img, cv2.COLOR_RGB2HSV)

    p = [0, -1, -1, -1, 0, +1, -1, +1]
    s = [-1, -1, 0, +1, +1, +1, 0, -1]

    for i in zip(p, s):
        if not all(i):
            print('h' + str(i))

    for r in range(0, edges.shape[0]):
        for c in range(0, edges.shape[1]):
            if current_hue > 179:
                current_sat = current_sat - 40
                current_hue = 1
            if edges[r, c]:
                if not any(checkandappend(r, c, r + i[0], c + i[1], lines_img, lines_dict) for i in zip(p, s)):
                    # Create new line
                    lines_img[r, c] = [current_hue, current_sat, 255]
                    lines_dict[str(current_hue) + str(current_sat)] = [(r, c)]
                    current_hue = current_hue + 1

    return cv2.cvtColor(lines_img, cv2.COLOR_HSV2BGR), lines_dict


def sample(lines, step):
    if step == 0:
        step = 1
    samples = {}
    for line_key in lines.keys():
        count = 0
        for pixel in lines[line_key]:
            last = count == len(lines[line_key]) - 1
            if count % step == 0 or last:
                if count == 0:
                    samples[line_key] = []
                samples[line_key].append(pixel)
            count = count + 1
        if len(samples[line_key]) < 3:
            del samples[line_key]
    return samples


def find_flex(derivatives, flex_thresh, n_chances=2):
    second_chance = False
    candidate_flex = ()
    rising = False
    flexes = {}
    for line_key in derivatives:
        flexes[line_key] = []
        count = 1
        ders = derivatives[line_key]
        for der in ders:
            if count <= len(ders) - 1:
                corner = abs(ders[count][1] - der[1] > flex_thresh)
                if ders[count][1] > der[1]:
                    if not rising:
                        if not second_chance or not corner:
                            second_chance = True
                            candidate_flex = ders[count][0]
                        else:
                            second_chance = False
                            flexes[line_key].append((candidate_flex, corner))
                            candidate_flex = ()
                            rising = True
                    else:
                        second_chance = False
                        candidate_flex = ()
                else:
                    if rising:
                        if not second_chance:
                            second_chance = True
                            candidate_flex = ders[count][0]
                        else:
                            second_chance = False
                            flexes[line_key].append((candidate_flex, corner))
                            rising = False
                            candidate_flex = ()
                    else:
                        second_chance = False
                        candidate_flex = ()
            count = count + 1
        if len(flexes[line_key]) == 0:
            del flexes[line_key]
    return flexes


def is_segment(sampled, dev_thresh, img_height, lines_img):
    # img height should be the maximum derivative possible
    derivatives = {}
    for line_key in sampled.keys():
        if len(sampled[line_key]) > 1:
            count = 1
            derivatives[line_key] = []
        else:
            continue
        for der in sampled[line_key]:
            delta_x = der[0] - sampled[line_key][count - 2][0]
            if delta_x == 0:
                derivatives[line_key].append(((sampled[line_key][count - 2]), 1))
            else:
                # normalized derivative = ( Δy / Δx ) / max_derivative
                derivatives[line_key].append(((sampled[line_key][count - 2]),
                                              (der[1] - sampled[line_key][count - 2][1] / delta_x) / img_height))
            count = count + 1

    segment_map = {}

    for line_key in derivatives.keys():
        delta_x = sampled[line_key][-1][0] - sampled[line_key][0][0]
        if delta_x == 0:
            overall_derivative = 1
        else:
            overall_derivative = (sampled[line_key][-1][1] - sampled[line_key][0][1] / delta_x) / img_height
        count = 0
        for der in derivatives[line_key]:
            delta_der = abs((der[1] - overall_derivative))
            if delta_der > dev_thresh:
                segment_map[line_key] = (overall_derivative, False)
                continue
            elif count == len(derivatives[line_key][1]) - 1:
                segment_map[line_key] = (overall_derivative, True)
            count = count + 1
    return segment_map, derivatives
