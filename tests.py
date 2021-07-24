import cv2

filter_pencil = '<filter x="0%" y="0%" width="100%" height="100%" filterUnits="objectBoundingBox" ' \
                'id="pencil"><feTurbulence type="fractalNoise" baseFrequency="0.5" numOctaves="5" ' \
                'stitchTiles="stitch" result="f1"></feTurbulence><feColorMatrix type="matrix" values="0 0 0 0 0, ' \
                '0 0 0 0 0, 0 0 0 0 0, 0 0 0 -1.5 1.5" result="f2"></feColorMatrix><feComposite operator="in" ' \
                'in2="f2b" in="SourceGraphic" result="f3"></feComposite><feTurbulence type="fractalNoise" ' \
                'baseFrequency="1.2" numOctaves="3" result="noise"></feTurbulence><feDisplacementMap ' \
                'xChannelSelector="R" yChannelSelector="G" scale="2.5" in="f3" ' \
                'result="f4"></feDisplacementMap></filter> '
filter_ink = '<filter id="ink"><feTurbulence baseFrequency="0"/><feDisplacementMap in="SourceGraphic" ' \
             'scale="10"/></filter> '


def test_flexes(lines_img, sampled, flexes):
    for line_key in sampled.keys():
        for pixel in sampled[line_key]:
            cv2.rectangle(lines_img, (pixel[1], pixel[0]), (pixel[1], pixel[0]), (255, 255, 255), 2)

    for line_key in flexes.keys():
        for pixel in flexes[line_key]:
            cv2.circle(lines_img, (pixel[0][1], pixel[0][0]), 1, (0, 255, 0) if not pixel[1] else (0, 0, 255), 2)

    return lines_img


def test_closures(img, lines, closures):
    for line_key in lines.keys():
        if line_key not in closures:
            for pixel in lines[line_key]:
                img[pixel] = (255, 255, 255)

    return img


def test_shapes(img, lines, shapes):
    for line_key in shapes.keys():
        shape = shapes[line_key]
        line = lines[line_key]
        pixel = line[0]
        color = (int(img[pixel][0]), int(img[pixel][1]), int(img[pixel][2]))
        img = cv2.putText(img, shape, (pixel[1], pixel[0]), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, color, 1, cv2.LINE_AA)
    return img


def test_is_segment(segment_map, lines, lines_img):
    for k in segment_map.keys():
        if segment_map[k][1]:
            if k in lines.keys():
                for i in lines[k]:
                    lines_img[i] = (255, 255, 255)

    return lines_img


def test_drawing():
    svg_file = open('output/Vectorized_sample.svg', "r")
    html_file = open('output/animation.html', "w")
    svg = svg_file.read()
    html_file.write(
        '<html lang="it"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, '
        'initial-scale=1.0"><link href="style.css" rel="stylesheet" type="text/css" media="all" /><script '
        'type="text/javascript" src="script.js" defer></script><title> Animation </title></head><body><div '
        'id="container"> </div> '
        + svg.split('<defs />')[0] + '<defs>' + filter_ink + filter_pencil + '</defs>' + svg.split('<defs />')[1]
        + '</body></html>'
    )
    svg_file.close()
    html_file.close()
