import cv2
import numpy as np

p = [0, -1, -1, -1, 0, +1, +1, +1]
s = [-1, -1, 0, +1, +1, +1, 0, -1]

for j in zip(p, s):
    print(j)


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
            if edges_img[r2, c2] and not lines_img[r2, c2][2]:
                return r2, c2
    return False


def findlines(edges_img):
    current_hue = 0
    cycle = 0
    lines_dict = {}
    lines_img = np.zeros((edges_img.shape[0], edges_img.shape[1], 3), np.uint8)

    lines_img = cv2.cvtColor(lines_img, cv2.COLOR_RGB2HSV)

    print(edges_img.shape[0])
    print(edges_img.shape[1])

    for r in range(0, edges_img.shape[0] - 1):
        for c in range(0, edges_img.shape[1] - 1):
            if edges_img[r, c] and not lines_img[r, c][2]:
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


def find_flex(derivatives, sampled, flex_thresh, n_chances=2):
    ### Inizialitation of variables ###
    candidate_flex = ()  # Registers the candidate flex before confirming it
    rising = False  # True if the derivative is rising
    flexes = {}  # Dict to save the flexes
    corner = False
    nTemp_chances = 0
    for line_key in derivatives:
        flexes[line_key] = []  # Init of the list of flexes in the current segment
        count = 1
        ders = derivatives[line_key]  # Get the derivatives of the current segment
        for der in ders:
            if count <= len(ders) - 1:  # Check of the loop
                corner = corner or abs(ders[count][1] - der[1]) > flex_thresh
                flexes[line_key].append((sampled[line_key][0], True))
                flexes[line_key].append((sampled[line_key][-1], True))
                # If the second derivative is greater than the
                # treshold, set the
                # flex as a corner
                if ders[count][1] > der[1]:  # If the derivative is greater than before (I'M RISING)
                    if not rising:  # If i wasn't rising
                        if nTemp_chances <= n_chances:
                            nTemp_chances += 1
                            candidate_flex = ders[count][0] if nTemp_chances == 1 else candidate_flex
                        else:
                            nTemp_chances = 0
                            flexes[line_key].append((candidate_flex, corner))
                            corner = False
                            candidate_flex = ()
                            rising = True
                    else:  # If i was rising already
                        nTemp_chances = 0
                        candidate_flex = ()
                        corner = False
                else:  # If the derivative is lower than before (I'M DESCENDING)
                    if rising:  # If i was rising
                        if nTemp_chances <= n_chances:
                            nTemp_chances += 1
                            candidate_flex = ders[count][0] if nTemp_chances == 1 else candidate_flex
                        else:
                            nTemp_chances = 0
                            flexes[line_key].append((candidate_flex, corner))
                            corner = False
                            rising = False
                            candidate_flex = ()
                    else:  # If i wasn't rising already
                        nTemp_chances = 0
                        candidate_flex = ()
                        corner = False
            count = count + 1
        if len(flexes[line_key]) == 0:  # Delete empty flexes arrays
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
            delta_y = der[1] - sampled[line_key][count - 2][1]
            if delta_x == 0:
                derivatives[line_key].append(((sampled[line_key][count - 2]), 1 if delta_y > 0 else -1))
            else:
                # normalized derivative = ( Δy / Δx ) / max_derivative
                derivatives[line_key].append(((sampled[line_key][count - 2]),
                                              (delta_y / delta_x) / img_height))
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


''' DEPRECATED #####################################
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
