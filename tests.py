import cv2


def test(lines_img, sampled, flexes):
    for line_key in sampled.keys():
        for pixel in sampled[line_key]:
            cv2.circle(lines_img, (pixel[1], pixel[0]), 1, (255, 255, 255), 1)

    for line_key in flexes.keys():
        for pixel in flexes[line_key]:
            cv2.circle(lines_img, (pixel[0][1], pixel[0][0]), 1, (0, 255, 0) if not pixel[1] else (0, 0, 255), 2)

    cv2.imshow("Test", lines_img)


def test_is_segment(segment_map, lines, lines_img):
    for k in segment_map.keys():
        if segment_map[k][1]:
            if k in lines.keys():
                for i in lines[k]:
                    lines_img[i] = (255, 255, 255)

    cv2.imshow("is_segment test", lines_img)
