import svgwrite


def init(img_x, img_y):
    svg = svgwrite.Drawing(filename="output/Vectorized_sample.svg", size=(str(img_y) + 'px', str(img_x) + 'px'))
    marker = svg.marker(insert=(1, 1), size=(4, 4), orient='auto')
    marker.add(svg.circle((1, 1), r=0.5, fill='black'))
    svg.defs.add(marker)
    return svg, marker


def vectorize_flexes(img_x, img_y, flexes):
    svg, marker = init(img_x, img_y)
    path = ''

    for k in flexes.keys():
        if len(flexes[k]) > 2:
            corner = False
            count = 0
            for node in flexes[k]:
                if node[1]:
                    corner = True  # The node will be a cusp
                if count == 0:
                    path = 'M ' + str(node[0][1]) + ',' + str(node[0][0]) + ' ' + str(node[0][1]) + ',' + str(
                        node[0][0]) + ' \n'
                elif count % 2 != 0 and count != len(flexes[k]) - 1:
                    if corner:
                        path = path + 'C ' + str(node[0][1]) + ',' + str(node[0][0]) + ',' + str(
                            node[0][0]) + ',' + str(node[0][1])
                    else:
                        path = path + 'S ' + str(node[0][1]) + ',' + str(node[0][0])
                elif count % 2 == 0:
                    path = path + ' ' + str(node[0][1]) + ',' + str(node[0][0]) + '\n'
                else:
                    if corner:
                        path = path + 'C ' + str(node[0][1]) + ',' + str(node[0][0]) + ' ' + str(
                            node[0][1]) + ',' + str(
                            node[0][0]) + ' \n'
                    else:
                        path = path + 'S ' + str(node[0][1]) + ',' + str(node[0][0]) + ' ' + str(
                            node[0][1]) + ',' + str(
                            node[0][0]) + ' \n'
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


def vectorize_samples(img_x, img_y, sampled):
    svg, marker = init(img_x, img_y)
    path = ''

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
                            stroke_width=3)

            path.set_markers((marker, False, marker))

            svg.add(path)
    svg.save()
    return svg
