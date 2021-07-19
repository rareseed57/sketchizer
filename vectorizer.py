import svgwrite

def init(img_x, img_y):
    svg = svgwrite.Drawing(filename="output/Vectorized_sample.svg", size=(str(img_y) + 'px', str(img_x) + 'px'))
    return svg


def vectorize_flexes(img_x, img_y, flexes, filter='pencil'):
    svg = init(img_x, img_y)
    path = ''
    filter = "url(#" + filter + ")"

    for k in flexes.keys():
        if len(flexes[k]) > 2:
            corner = False
            count = 0
            for node in flexes[k]:
                corner = node[1]  # The node will be a cusp
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
