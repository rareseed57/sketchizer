import math

import numpy as np
import svgwrite


def init(img_x, img_y):
    svg = svgwrite.Drawing(filename="output/Vectorized.svg", size=(str(img_y) + 'px', str(img_x) + 'px'))
    return svg


def vectorize_flexes(img_x, img_y, flexes, closures, shapes, filter='pencil'):
    svg = init(img_x, img_y)
    path = ''
    filter = "url(#" + filter + ")"

    for line_key in flexes.keys():
        if len(flexes[line_key]) > 2:
            corner = False
            count = 0
            line = flexes[line_key]

            # svg interprets coordinates in the (y,x) format
            if line_key in shapes.keys():
                shape = shapes[line_key][0]
                data = np.uint16(np.around(shapes[line_key][1]))
                if shape == "circle":
                    svg.add(svg.circle(
                        center=(int(data[0]), int(data[1])),
                        r=int(data[2]),
                        stroke="#000",
                        fill="none",
                        stroke_width=3,
                        filter=filter
                    ))
                elif shape == "triangle" or "pentagon":
                    points = []
                    for i in data[:]:
                        points.append((int(i[0][1]), int(i[0][0])))
                    svg.add(svg.polygon(
                        points=points,
                        stroke="#000",
                        fill="none",
                        stroke_width=3,
                        filter=filter
                    ))
                elif shape == "square":
                    p00x = data[0][0][0]
                    p00y = data[0][0][1]
                    p11x = data[0][0][0]
                    p11y = data[0][0][1]
                    for i in data[:]:
                        i = i[0]
                        if p00x > i[0]:
                            p00x = i[0]
                        if p11x < i[0]:
                            p11x = i[0]
                        if p00y > i[1]:
                            p00y = i[1]
                        if p11y < i[1]:
                            p11y = i[1]
                    insert = (int(p00x), int(p00y))
                    size = int(p11y - p00y), int(p11y - p00y)
                    svg.add(svg.rect(insert, size,
                                     stroke="#000",
                                     fill="none",
                                     stroke_width=5,
                                     filter=filter),
                            )
                elif shape == "rectangle":
                    p00x = data[0][0][0]
                    p00y = data[0][0][1]
                    p11x = data[0][0][0]
                    p11y = data[0][0][1]
                    for i in data[:]:
                        i = i[0]
                        if p00x > i[0]:
                            p00x = i[0]
                        if p11x < i[0]:
                            p11x = i[0]
                        if p00y > i[1]:
                            p00y = i[1]
                        if p11y < i[1]:
                            p11y = i[1]
                    insert = (int(p00x), int(p00y))
                    size = int(p11y - p00y), int(p11x - p00x)
                    svg.add(svg.rect(insert, size,
                                     stroke="#000",
                                     fill="none",
                                     stroke_width=5,
                                     filter=filter),
                            )
                continue

            for node in line:
                coords = node[0]
                corner = node[1]  # The node will be a corner
                der_out = node[2] * math.pi
                der_in = der_out
                if corner:
                    der_in = node[3] * math.pi
                previous_coords = flexes[line_key][count - 1]
                previous_corner = previous_coords[1]
                handle_out = (coords[0] + math.cos(der_out), coords[1] + math.sin(der_out))
                handle_in = handle_out
                if corner:
                    handle_in = (coords[0] + math.cos(der_in), coords[1] + math.sin(der_in))
                if count == 0:
                    # M y,x  y,x \n
                    path = 'M ' + \
                           str(coords[1]) + ',' + str(coords[0]) + ' \n'
                elif count <= len(flexes[line_key]) - 1:
                    # DA CORNER A CORNER (LINEA)
                    if previous_corner and corner:
                        path = path + 'L ' + str(coords[1]) + ',' + str(coords[0]) + ' \n'
                    # DA FLESSO A CORNER
                    elif not previous_corner and corner:
                        path = path + 'C ' \
                               + str(handle_in[1]) + ',' + str(handle_in[0]) + ' ' \
                               + str(handle_out[1]) + ',' + str(handle_out[0]) + ' ' \
                               + str(coords[1]) + ',' + str(coords[0]) + '\n'
                    # DA FLESSO/CORNER A FLESSO
                    elif not corner:
                        path = path + 'S ' + str(handle_out[1]) + ',' + str(handle_out[0]) + ' ' \
                               + str(coords[1]) + ',' + str(coords[0]) + '\n'

                count = count + 1
            if line_key in closures:
                path = path + 'Z'

            # print('Adding\n' + path + '\n')

            path = svg.path(d=path,
                            stroke="#000",
                            fill="none",
                            stroke_width=3,
                            filter=filter)

            svg.add(path)
    svg.save()
    return svg


def vectorize_samples(img_x, img_y, sampled, filter='pencil'):
    svg = init(img_x, img_y)
    path = ''
    filter = "url(#" + filter + ")"

    for k in sampled.keys():
        if len(sampled[k]) > 2:
            count = 0
            for node in sampled[k]:
                if count == 0:
                    path = 'M ' + str(node[1]) + ',' + str(node[0]) + ' ' + str(node[1]) + ',' + str(node[0]) + ' \n'
                elif count % 2 != 0 and count != len(sampled[k]) - 1:
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
                            stroke_width=4,
                            filter=filter)

            svg.add(path)
    svg.save()
    return svg
