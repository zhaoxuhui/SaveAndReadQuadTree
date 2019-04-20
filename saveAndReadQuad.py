# coding=utf-8
import numpy as np
import cv2

LEVEL_LENGTH = [1, 1, 2, 2, 3, 4, 4, 5]
INDEX_TABLE = [0, 1, 2, 4, 6, 9, 13, 17, 22, 27]
LEVEL_NUM = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536]


def drawTree(tree, node_color=(0, 0, 0), final_color=(0, 0, 255), node_size=20, node_space=50, margin_size=50,
             level_height=200, showNone=True):
    height = (tree.__len__() - 1) * level_height + margin_size * 2
    width = node_space * (tree[-1].__len__() - 1) + margin_size * 2
    center_h = width / 2
    # print height, width
    blank_img = np.zeros([height, width, 3], np.uint8) + 255
    for level in range(len(tree)):
        for node in range(len(tree[level])):
            start_x = center_h - (len(tree[level]) - 1) * (len(tree) - level) * node_space / 2
            if tree[level][node] is not None:
                if tree[level][node] == 1:
                    cv2.circle(blank_img,
                               (start_x + (len(tree) - level) * node_space * node, margin_size + level_height * level),
                               node_size,
                               color=node_color,
                               thickness=-1,
                               lineType=cv2.LINE_AA)
                else:
                    cv2.circle(blank_img,
                               (start_x + (len(tree) - level) * node_space * node, margin_size + level_height * level),
                               node_size,
                               color=final_color,
                               thickness=-1,
                               lineType=cv2.LINE_AA)
            else:
                if showNone:
                    cv2.circle(blank_img,
                               (start_x + (len(tree) - level) * node_space * node, margin_size + level_height * level),
                               node_size,
                               color=node_color,
                               thickness=1,
                               lineType=cv2.LINE_AA)
                else:
                    pass
    return blank_img


def drawNodePath(tree, quadCode, line_color=(0, 0, 255), line_width=2, node_in_path_color=(0, 0, 255),
                 node_color=(0, 0, 0),
                 node_size=20, node_space=50,
                 margin_size=50, level_height=200):
    height = (tree.__len__() - 1) * level_height + margin_size * 2
    width = node_space * (tree[-1].__len__() - 1) + margin_size * 2
    center_h = width / 2
    print height, width
    blank_img = np.zeros([height, width, 3], np.uint8) + 255
    path = getNodeSearchPath(quadCode)
    points = []
    for level in range(len(tree)):
        for node in range(len(tree[level])):
            start_x = center_h - (len(tree[level]) - 1) * (len(tree) - level) * node_space / 2
            if node == path[level]:
                points.append((start_x + (len(tree) - level) * node_space * node, margin_size + level_height * level))
                cv2.circle(blank_img,
                           (start_x + (len(tree) - level) * node_space * node, margin_size + level_height * level),
                           node_size,
                           color=node_in_path_color,
                           thickness=-1,
                           lineType=cv2.LINE_AA)
            else:
                if level == 0:
                    points.append(
                        (start_x + (len(tree) - level) * node_space * node, margin_size + level_height * level))
                cv2.circle(blank_img,
                           (start_x + (len(tree) - level) * node_space * node, margin_size + level_height * level),
                           node_size,
                           color=node_color,
                           thickness=1,
                           lineType=cv2.LINE_AA)
    for i in range(len(points) - 1):
        cv2.line(blank_img, points[i], points[i + 1], color=line_color, lineType=cv2.LINE_AA, thickness=line_width)
    return blank_img


def drawNodePathOnTree(blank_img, tree, quadCode, line_color=(0, 0, 255), line_width=2, node_space=50,
                       margin_size=50, level_height=200):
    height = (tree.__len__() - 1) * level_height + margin_size * 2
    width = node_space * (tree[-1].__len__() - 1) + margin_size * 2
    center_h = width / 2
    # print height, width
    path = getNodeSearchPath(quadCode)
    print path
    points = []

    depth = getDepthByLength(len(quadCode))
    for level in range(depth):
        for node in range(len(tree[level])):
            start_x = center_h - (len(tree[level]) - 1) * (len(tree) - level) * node_space / 2
            if node == path[level] - 1:
                points.append((start_x + (len(tree) - level) * node_space * node, margin_size + level_height * level))
            else:
                if level == 0:
                    points.append(
                        (start_x + (len(tree) - level) * node_space * node, margin_size + level_height * level))
    for i in range(len(points) - 1):
        cv2.line(blank_img, points[i], points[i + 1], color=line_color, lineType=cv2.LINE_AA, thickness=line_width)
    return blank_img


def buildFullQuad(maxDepth):
    tree = []
    for i in range(maxDepth):
        # print LEVEL_NUM[i]
        tree.append([None] * LEVEL_NUM[i])
    return tree


def getDepthByLength(code_length):
    if code_length == 1:
        return 1
    elif code_length == 2:
        return 2
    elif code_length == 4:
        return 3
    elif code_length == 6:
        return 4
    elif code_length == 9:
        return 5
    elif code_length == 13:
        return 6
    elif code_length == 17:
        return 7
    elif code_length == 22:
        return 8
    elif code_length == 27:
        return 9
    else:
        return -1


def getNodeSearchPath(quadCode):
    length = len(quadCode)
    depth = getDepthByLength(length)
    # print depth
    search_path = []
    for i in range(depth):
        # print i + 1, quadCode[INDEX_TABLE[i]:INDEX_TABLE[i + 1]]
        search_path.append(int(quadCode[INDEX_TABLE[i]:INDEX_TABLE[i + 1]]))
    return search_path


def readQuadCode(quadCode):
    code_length = len(quadCode)
    # print code_length
    depth = getDepthByLength(code_length)
    return depth


def insertNode(tree, quadCode, value=0):
    path = getNodeSearchPath(quadCode)
    # print path
    depth = getDepthByLength(len(quadCode))
    for i in range(len(path)):
        tree[i][path[i] - 1] = 1
    tree[depth - 1][path[-1] - 1] = value
    return tree


def saveNodes(nodes, save_path="nodes.txt"):
    f = open(save_path, 'w')
    max_depth = 0
    for i in range(nodes.__len__()):
        depth = getDepthByLength(len(nodes[i][0]))
        if depth > max_depth:
            max_depth = depth
    f.write(max_depth.__str__() + "\n")
    for i in range(nodes.__len__()):
        f.write(nodes[i][0] + "\t" + nodes[i][1].__str__() + "\n")
    f.close()


def readNodes(node_path):
    f = open(node_path, 'r')
    max_depth = int(f.readline())
    nodes = []
    line = f.readline()
    while line:
        code = line.split("\t")[0]
        value = line.split("\t")[1]
        nodes.append((code, float(value)))
        line = f.readline()
    return nodes, max_depth


def cvtPosListToFullQuadCodeList(pos_list):
    depth = len(pos_list)
    code = []
    for i in range(depth - 1, -1, -1):
        if i - 1 < 0:
            code.append(pos_list[0])
        else:
            code.append((pos_list[i - 1] - 1) * 4 + pos_list[i])
    code.reverse()
    return code


def cvtPosListToFullQuadCodeStr(pos_list):
    code_str = ""
    code = cvtPosListToFullQuadCodeList(pos_list)
    for i in range(len(code)):
        code_str += code[i].__str__().zfill(INDEX_TABLE[i])
    return code_str


def cvtPosStrToFullQuadCodeList(pos_str):
    path = getNodeSearchPath(pos_str)
    code_list = cvtPosListToFullQuadCodeList(path)
    return code_list


def cvtPosStrToFullQuadCodeStr(pos_str):
    code_str = ""
    code = cvtPosStrToFullQuadCodeList(pos_str)
    for i in range(len(code)):
        code_str += code[i].__str__().zfill(INDEX_TABLE[i])
    return code_str


def cvtQuadCodeStrToPosList(code_str):
    paths = getNodeSearchPath(code_str)
    depth = len(paths)
    pos = []
    for i in range(depth - 1, -1, -1):
        if i - 1 < 0:
            pos.append(paths[0])
        else:
            pos.append(paths[i] - (paths[i - 1] - 1) * 4)
    pos.reverse()
    return pos


def cvtQuadCodeListToPosList(code_list):
    depth = len(code_list)
    pos = []
    for i in range(depth - 1, -1, -1):
        if i - 1 < 0:
            pos.append(code_list[0])
        else:
            pos.append(code_list[i] - (code_list[i - 1] - 1) * 4)
    pos.reverse()
    return pos


def cvtQuadCodeStrToPosStr(code_str):
    pos_list = cvtQuadCodeStrToPosList(code_str)
    pos_str = ""
    for i in range(len(pos_list)):
        pos_str += pos_list[i].__str__().zfill(INDEX_TABLE[i])
    return pos_str


def cvtQuadCodeListToPosStr(code_list):
    pos_list = cvtQuadCodeListToPosList(code_list)
    depth = len(pos_list)
    pos_str = ""
    for i in range(depth):
        pos_str += pos_list[i].__str__().zfill(INDEX_TABLE[i])
    return pos_str


if __name__ == '__main__':
    nodes = []
    nodes.append((cvtPosStrToFullQuadCodeStr("1"), 200))
    nodes.append((cvtPosStrToFullQuadCodeStr("14"), 100))
    nodes.append((cvtPosStrToFullQuadCodeStr("1301"), 100))
    nodes.append((cvtPosStrToFullQuadCodeStr("1303"), 100))
    nodes.append((cvtPosStrToFullQuadCodeStr("1103"), 100))
    nodes.append((cvtPosStrToFullQuadCodeStr("1104"), 100))
    saveNodes(nodes)
    nodes, max_depth = readNodes("nodes.txt")
    tree = buildFullQuad(max_depth)
    for i in range(len(nodes)):
        tree = insertNode(tree, nodes[i][0], nodes[i][1])
    tree_nodes = drawTree(tree)
    cv2.imwrite("nodes.png", tree_nodes)
    tree_img = drawTree(tree, showNone=False)
    for i in range(len(nodes)):
        tree_img = drawNodePathOnTree(tree_img, tree, nodes[i][0])
    cv2.imwrite("paths.png", tree_img)
