# coding=utf-8
import cv2
import numpy as np

# 每一级索引所占用字符宽度
LEVEL_LENGTH = [1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8]


def getIndexByPixel(x, y, img, level):
    width = img.shape[1]
    height = img.shape[0]
    block_num_width = pow(2, level - 1)
    block_num_height = pow(2, level - 1)
    block_width = width / block_num_width
    block_height = height / block_num_height
    col = int(x / block_width)
    row = int(y / block_height)
    return row * block_num_width + col + 1


def getQuadListByPixel(x, y, img, depth):
    indices = []
    for i in range(depth):
        indices.append(getIndexByPixel(x, y, img, i + 1))
    return indices


def cvtQuadListToQuadCode(indices):
    quad_code = ""
    for i in range(len(indices)):
        quad_code += indices[i].__str__().zfill(LEVEL_LENGTH[i])
    return quad_code


def getQuadCodeByPixel(x, y, img, depth):
    indices = getQuadListByPixel(x, y, img, depth)
    code = cvtQuadListToQuadCode(indices)
    return code


def saveQuadCodes(model, img, save_path):
    center_point = []
    max_depth = 0

    for i in range(model.heap.__len__()):
        x_min = model.heap[i][2].box[0]
        y_min = model.heap[i][2].box[1]
        x_max = model.heap[i][2].box[2]
        y_max = model.heap[i][2].box[3]
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        depth = model.heap[i][2].depth + 1

        if max_depth < depth:
            max_depth = depth
        center_point.append((center_x, center_y, depth))

    codes = []
    for i in range(len(center_point)):
        code = getQuadCodeByPixel.getQuadCodeByPixel(center_point[i][0], center_point[i][1], img, center_point[i][2])
        codes.append(code)

    f = open(save_path, 'w')
    f.write((max_depth).__str__() + "\n")
    for i in range(len(codes)):
        f.write(codes[i] + "\t" + "200" + "\n")
    f.close()


if __name__ == '__main__':
    img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
    code = getQuadCodeByPixel(300, 800, img, 4)
    print code
    # indices = getQuadListByPixel(300, 800, img, 4)
    # print indices
