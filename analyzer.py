import math

import cv2
import numpy as np

# Data structs to manage checks in a range of 1 pixels
p = [0, -1, 0, +1, +1, -1, -1, +1]
s = [-1, 0, +1, 0, -1, -1, +1, +1]


# Generates all the couples with the above vectors, basing on the index
# for j in zip(p, s):
#     print(j)


def checkinrange(center, target, r=1):
    if target[0] in range(center[0] - r, center[0] + r + 1) and target[1] in range(center[1] - r, center[1] + r + 1):
        return True
    return False


def check1around(coords, edges_img, lines_img):
    r, c = coords
    for i in zip(p, s):
        r2 = r + i[0]
        c2 = c + i[1]
        if not (r2 < 0 or r2 >= edges_img.shape[0] or c2 < 0 or c2 >= edges_img.shape[1]):
            if edges_img[r2][c2] and not lines_img[r2][c2][2]:
                return r2, c2
    return False


def findlines(edges_img):
    current_hue = 0
    cycle = 0
    lines_dict = {}
    lines_img = np.zeros((edges_img.shape[0], edges_img.shape[1], 3), np.uint8)

    lines_img = cv2.cvtColor(lines_img, cv2.COLOR_RGB2HSV)

    for r in range(0, edges_img.shape[0] - 1):
        for c in range(0, edges_img.shape[1] - 1):
            if edges_img[r][c] and not lines_img[r][c][2]:
                # Define the color
                if current_hue > 179:
                    cycle += 1
                    current_hue = 0
                index = str(current_hue) + str(cycle)
                color = [current_hue, 255, 255]
                # Create new line
                lines_img[r, c] = color
                lines_dict[index] = [(r, c)]
                # Region growing(tail)
                neighbor = check1around((r, c), edges_img, lines_img)
                while neighbor:
                    # Add neighbor
                    lines_img[neighbor] = color
                    lines_dict[index].append(neighbor)
                    # Find next pixel
                    neighbor = check1around(neighbor, edges_img, lines_img)
                # Region growing(head)
                neighbor = check1around(lines_dict[index][0], edges_img, lines_img)
                while neighbor:
                    # Add neighbor
                    lines_img[neighbor] = color
                    lines_dict[index].insert(0, neighbor)
                    # Find previous pixel
                    neighbor = check1around(neighbor, edges_img, lines_img)
                current_hue = current_hue + 20
    return lines_img, lines_dict


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
        if len(samples[line_key]) < 2:
            del samples[line_key]
    return samples


def find_flexes(derivatives, sampled, corner_thresh, closures, max_chances=2):
    ### Inizialitation of variables ###
    flexes = {}  # Dict to save the flexes
    given_chanches = 0
    for line_key in derivatives.keys():
        flexes[line_key] = []  # Init of the list of flexes in the current segment
        count = 1
        ders = derivatives[line_key]  # Get the derivatives of the current segment
        # Add the first and last sample as corners
        if line_key not in closures:
            flexes[line_key].append((sampled[line_key][0], True))
            flexes[line_key].append((sampled[line_key][-1], True))
        rising = ders[1][1] > ders[0][1]

        while count <= len(ders) - 1:  # Check of the loop
            der = ders[count - 1]
            next_der = ders[count]
            candidate_flex = der[0]
            corner = abs(next_der[1] - der[1]) > corner_thresh
            if corner:
                given_chanches = 0
                flexes[line_key].append((candidate_flex, True))
                count += 1
                continue
            # If the second derivative is greater than the
            # treshold, set the
            # flex as a corner
            candidate_flex = der[0] if given_chanches == 0 else candidate_flex
            if next_der[1] > der[1]:  # If i'm rising (the derivative is greater than before)
                if not rising:  # and i wasn't rising
                    if given_chanches < max_chances:
                        given_chanches += 1
                    else:
                        given_chanches = 0
                        flexes[line_key].append((candidate_flex, False))
                    rising = True
                else:  # and i was already rising
                    given_chanches = 0
            else:  # If i'm descending (the derivative is lower than before)
                if rising:  # If i was rising
                    if given_chanches < max_chances:
                        given_chanches += 1
                    else:
                        given_chanches = 0
                        flexes[line_key].append((candidate_flex, False))
                    rising = False
                else:  # If i wasn't rising already
                    given_chanches = 0
            count += 1
        if len(flexes[line_key]) == 0:  # Delete empty flexes arrays
            del flexes[line_key]
    return flexes


def compute_derivatives(sampled):
    derivatives = {}
    for line_key in sampled.keys():
        if len(sampled[line_key]) > 1:
            count = 1
            derivatives[line_key] = []
        else:
            continue
        for pixel in sampled[line_key]:
            coord1 = pixel
            coord2 = sampled[line_key][count - 2]
            delta_x = coord1[0] - coord2[0]
            delta_y = coord1[1] - coord2[1]
            # Using atan as a slope measure cause it can be evaluated between 0 and 1
            slope = (math.atan2(delta_y, delta_x) / math.pi)
            derivatives[line_key].append((coord1, slope))
            count = count + 1
    ''' 
    DEPRECATED #####################################
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
        '''
    return derivatives


def check_closures(lines, r):
    closures = []
    for line_key in lines.keys():
        if checkinrange(lines[line_key][0], lines[line_key][-1], r):
            closures.append(line_key)
    return closures


def floodfill_line(line):
    # Create an image containing the line
    max_x = line[0][0]
    max_y = line[0][1]
    for pixel in line:
        if pixel[0] > max_x:
            max_x = pixel[0]
        if pixel[1] > max_y:
            max_y = pixel[1]
    img = np.zeros((max_x + 2, max_y + 2, 1), np.uint8)
    for pixel in line:
        img[pixel] = 255

    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img, mask, (0, 0), 255)
    img = cv2.bitwise_not(img)

    return img


def detect_shapes(lines, closures, approx_factor=0.9):
    shapes = {}
    for line_key in lines.keys():
        img = floodfill_line(lines[line_key])
        img = cv2.blur(img, (5, 5))
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1,
                                   param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None and len(circles) == 1:
            shapes[line_key] = "circle"
            # else check ellipses (not supported)
        else:  # else check other shapes
            if line_key not in closures:
                continue  # discard lines that aren't closed

            line = np.float32(lines[line_key])
            perimeter = cv2.arcLength(line, True)
            approx = cv2.approxPolyDP(line, approx_factor * perimeter, True)
            # if the shape is a triangle, it will have 3 vertices
            if len(approx) == 3:
                shapes[line_key] = "triangle"

            # if the shape has 4 vertices, it is either a square or
            # a rectangle
            elif len(approx) == 4:
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)

                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                shapes[line_key] = "square" if 0.95 <= ar <= 1.05 else "rectangle"

            # if the shape is a pentagon, it will have 5 vertices
            elif len(approx) == 5:
                shapes[line_key] = "pentagon"

    return shapes


'''
DEPRECATED #####################################
def mergelines(lines_dict, lines_img):
    trash = []

    for current in lines_dict:
        if current not in trash:
            for connected in lines_dict:
                if connected not in trash:
                    head_connected = checkinrange(lines_dict[current][0], lines_dict[connected][-1], 2)
                    tail_connected = checkinrange(lines_dict[current][-1], lines_dict[connected][0], 2)
                    if head_connected or tail_connected:
                        color = lines_img[lines_dict[current][0]]
                        if head_connected:
                            tmp = lines_dict[connected].copy()
                            tmp.extend(lines_dict[current])
                            lines_dict[current] = tmp
                        elif tail_connected:
                            tmp = lines_dict[current].copy()
                            tmp.extend(lines_dict[connected])
                            lines_dict[current] = tmp
                        for pixel in lines_dict[connected]:
                            lines_img[pixel] = color
                        trash.append(connected)
                        break

    for i in trash:
        del lines_dict[i]

    return cv2.cvtColor(lines_img, cv2.COLOR_HSV2BGR), lines_dict

def checkandappend(r, c, r2, c2, lines, lines_list):
    if r2 < 0 or r2 >= lines.shape[0] or c2 < 0 or c2 >= lines.shape[1]:
        return False
    if lines[r2, c2][0] != 0:
        # print(_lines[_r2][_c2])
        index = str(lines[r2][c2][0]) + str(lines[r2][c2][1])
        last = lines_list[index][-1]
        # if r2 == last[0] and c2 == last[1]:
        # if any(last[0] == r + i[0] and last[1] == c + i[1] for i in zip(p, s)):
        # if last == (r, c - 1) or last == (r - 1, c - 1) or last == (r - 1, c) or last == (r - 1, c) or last == (
        # r - 1, c + 1) or last == (r + 1, c + 1) or last == (r + 1, c) or last == (r + 1, c + 1):
        if checkinrange((r, c), last):
            lines[r][c] = lines[r2][c2]
            lines_list[index].append((r, c))
            return True
    return False


def findlines_deprecated(edges):
    current_hue = 1
    current_sat = 255
    lines_dict = {}
    lines_img = np.zeros((edges.shape[0], edges.shape[1], 3), np.uint8)

    lines_img = cv2.cvtColor(lines_img, cv2.COLOR_RGB2HSV)

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

    return lines_img, lines_dict
'''
