# coding=utf-8
import numpy as np
import cv2

# 每一级索引所占用字符宽度
LEVEL_LENGTH = [1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8]
# 拆分字符串索引
INDEX_TABLE = [0, 1, 2, 4, 6, 9, 13, 17, 22, 27, 33, 40, 47, 55]
# 每一级最多容纳节点个数
LEVEL_NUM = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]


def drawNodes(tree, node_color=(0, 0, 0), final_color=(0, 0, 255), node_size=20, node_space=50, margin_size=50,
              level_height=200, showNone=True):
    """
    绘制树所包含的节点，没有连线。
    空白圆圈表示没有节点，黑色圆圈表示有下一级节点的节点，红色圆圈表示无下一级节点的节点

    :param tree: 需要画的树
    :param node_color: 非最终一级节点颜色，默认黑色
    :param final_color: 最终一级节点颜色，默认红色
    :param node_size: 节点大小，默认20
    :param node_space: 节点所占空间，默认50
    :param margin_size: 边缘大小，默认50
    :param level_height: 每一级高度，默认200
    :param showNone: 是否绘制空节点，默认绘制
    :return: 影像
    """
    height = (tree.__len__() - 1) * level_height + margin_size * 2
    width = node_space * (tree[-1].__len__() - 1) + margin_size * 2
    center_h = width / 2
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


def drawNodes2(tree, node_color=(0, 0, 0), final_color=(0, 0, 255), node_size=20, node_space=50, margin_size=50,
               level_height=200):
    """
    绘制树所包含的节点，没有连线。
    空白圆圈表示没有节点，黑色圆圈表示有下一级节点的节点，红色圆圈表示无下一级节点的节点

    :param tree: 需要画的树
    :param node_color: 非最终一级节点颜色，默认黑色
    :param final_color: 最终一级节点颜色，默认红色
    :param node_size: 节点大小，默认20
    :param node_space: 节点所占空间，默认50
    :param margin_size: 边缘大小，默认50
    :param level_height: 每一级高度，默认200
    :param showNone: 是否绘制空节点，默认绘制
    :return: 影像
    """

    counter = 0
    real_elmt_nums = []
    for i in range(len(tree)):
        for j in range(tree[i].__len__()):
            if tree[i][j] is not None:
                counter += 1
        real_elmt_nums.append(counter)
        counter = 0
    max_elmt_num = max(real_elmt_nums)

    height = (tree.__len__() - 1) * level_height + margin_size * 2
    width = node_space * (max_elmt_num - 1) + margin_size * 2
    max_draw_width = width - margin_size * 2
    center_w = width / 2
    blank_img = np.zeros([height, width, 3], np.uint8) + 255
    for level in range(len(tree)):
        if real_elmt_nums[level] == 1:
            cv2.circle(blank_img,
                       (center_w, margin_size + level_height * level),
                       node_size,
                       color=final_color,
                       thickness=-1,
                       lineType=cv2.LINE_AA)
        else:
            block_width = int((real_elmt_nums[level] * 1.0 / max_elmt_num) * max_draw_width)
            start_x = center_w - block_width / 2
            node_counter = 0
            step_length = block_width / (real_elmt_nums[level] - 1)
            for node in range(len(tree[level])):
                if tree[level][node] is not None:
                    if tree[level][node] == 1:
                        cv2.circle(blank_img,
                                   (start_x + step_length * node_counter, margin_size + level_height * level),
                                   node_size,
                                   color=node_color,
                                   thickness=-1,
                                   lineType=cv2.LINE_AA)
                    else:
                        cv2.circle(blank_img,
                                   (start_x + step_length * node_counter, margin_size + level_height * level),
                                   node_size,
                                   color=final_color,
                                   thickness=-1,
                                   lineType=cv2.LINE_AA)
                    node_counter += 1
    return blank_img


def drawPathOnNodes(blank_img, tree, quadCode, line_color=(128, 128, 128), line_width=2, node_space=50,
                    margin_size=50, level_height=200):
    """
    在已经绘制好的节点图上绘制连接线

    :param blank_img: 已经绘制好的节点图
    :param tree: 待绘制的树
    :param quadCode: 四叉树编码
    :param line_color: 连接线颜色，默认为灰色
    :param line_width: 连接线宽度，默认为2
    :param node_space: 节点空间大小，默认为50
    :param margin_size: 边缘大小，默认为50
    :param level_height: 每一层的高度，默认为200
    :return: 影像
    """

    height = (tree.__len__() - 1) * level_height + margin_size * 2
    width = node_space * (tree[-1].__len__() - 1) + margin_size * 2
    center_h = width / 2
    path = getNodeSearchPath(quadCode)
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


def drawPathOnNodes2(blank_img, tree, quadCode, line_color=(128, 128, 128), line_width=2, node_space=50,
                     margin_size=50, level_height=200):
    """
    在已经绘制好的节点图上绘制连接线

    :param blank_img: 已经绘制好的节点图
    :param tree: 待绘制的树
    :param quadCode: 四叉树编码
    :param line_color: 连接线颜色，默认为灰色
    :param line_width: 连接线宽度，默认为2
    :param node_space: 节点空间大小，默认为50
    :param margin_size: 边缘大小，默认为50
    :param level_height: 每一层的高度，默认为200
    :return: 影像
    """

    counter = 0
    real_elmt_nums = []
    for i in range(len(tree)):
        for j in range(tree[i].__len__()):
            if tree[i][j] is not None:
                counter += 1
        real_elmt_nums.append(counter)
        counter = 0
    max_elmt_num = max(real_elmt_nums)

    height = (tree.__len__() - 1) * level_height + margin_size * 2
    width = node_space * (max_elmt_num - 1) + margin_size * 2
    max_draw_width = width - margin_size * 2
    center_w = width / 2
    path = getNodeSearchPath(quadCode)
    points = []
    depth = getDepthByLength(len(quadCode))
    for level in range(depth):
        if real_elmt_nums[level] == 1:
            points.append((center_w, margin_size + level_height * level))
        else:
            block_width = int((real_elmt_nums[level] * 1.0 / max_elmt_num) * max_draw_width)
            start_x = center_w - block_width / 2
            node_counter = 0
            step_length = block_width / (real_elmt_nums[level] - 1)
            for node in range(len(tree[level])):
                if tree[level][node] is not None:
                    if node == path[level] - 1:
                        points.append((start_x + step_length * node_counter, margin_size + level_height * level))
                    node_counter += 1
    for i in range(len(points) - 1):
        cv2.line(blank_img, points[i], points[i + 1], color=line_color, lineType=cv2.LINE_AA, thickness=line_width)
    return blank_img


def drawTree(tree, nodes, showNone=False):
    """
    绘制带有节点与连接线的树结构

    :param tree: 待绘制的树
    :param nodes: 节点列表
    :param showNone: 是否绘制空节点，默认不绘制
    :return: 绘制好的影像
    """
    tree_img = drawNodes(tree, showNone=showNone)
    for i in range(len(nodes)):
        tree_img = drawPathOnNodes(tree_img, tree, nodes[i][0])
    return tree_img


def drawTree2(tree, nodes):
    """
    绘制带有节点与连接线的树结构

    :param tree: 待绘制的树
    :param nodes: 节点列表
    :param showNone: 是否绘制空节点，默认不绘制
    :return: 绘制好的影像
    """
    tree_img = drawNodes2(tree)
    for i in range(len(nodes)):
        tree_img = drawPathOnNodes2(tree_img, tree, nodes[i][0])
    return tree_img


def buildFullQuad(maxDepth):
    """
    依据最大深度建立完全四叉树

    :param maxDepth: 最大深度
    :return: 完全四叉树
    """
    tree = []
    for i in range(maxDepth):
        tree.append([None] * LEVEL_NUM[i])
    return tree


def getDepthByLength(code_length):
    """
    根据四叉树编码字符串长度获取其深度

    :param code_length: 四叉树编码字符串长度
    :return: 深度
    """

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
    elif code_length == 33:
        return 10
    elif code_length == 40:
        return 11
    elif code_length == 47:
        return 12
    elif code_length == 55:
        return 13
    else:
        return -1


def getNodeSearchPath(quadCode):
    """
    依据四叉树编码获得其在每一层中的索引位置

    :param quadCode: 四叉树编码字符串
    :return: 包含每一层索引位置的list
    """

    length = len(quadCode)
    depth = getDepthByLength(length)
    search_path = []
    for i in range(depth):
        search_path.append(int(quadCode[INDEX_TABLE[i]:INDEX_TABLE[i + 1]]))
    return search_path


def insertNode(tree, quadCode, value=0):
    """
    向树中插入节点，以及节点内容

    :param tree: 待插入节点的树
    :param quadCode: 四叉树编码
    :param value: 节点值
    :return: 插入节点后的四叉树
    """
    path = getNodeSearchPath(quadCode)
    depth = getDepthByLength(len(quadCode))
    for i in range(len(path)):
        tree[i][path[i] - 1] = 1
    tree[depth - 1][path[-1] - 1] = value
    return tree


def saveNodes(nodes, save_path="nodes.txt"):
    """
    保存四叉树数据结构，默认保存名称为nodes.txt

    :param nodes: 待保存的四叉树
    :param save_path: 保存文件的名称
    :return: 无
    """

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
    """
    读取四叉树文件内容

    :param node_path: 四叉树文件路径
    :return: 读取到的四叉树数据结构以及存储的内容
    """

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
    """
    将真实四叉树位置list转为四叉树编码list

    :param pos_list: 真实四叉树位置list
    :return: 四叉树编码list
    """
    depth = len(pos_list)
    code = []
    for i in range(depth):
        if i == 0:
            code.append(pos_list[0])
        else:
            code.append((code[i - 1] - 1) * 4 + pos_list[i])
    return code


def cvtPosListToFullQuadCodeStr(pos_list):
    """
    将真实四叉树位置list转为四叉树编码str

    :param pos_list: 真实四叉树位置list
    :return: 四叉树编码str
    """
    code_str = ""
    code = cvtPosListToFullQuadCodeList(pos_list)
    for i in range(len(code)):
        code_str += code[i].__str__().zfill(LEVEL_LENGTH[i])
    return code_str


def cvtPosStrToFullQuadCodeList(pos_str):
    """
    将真实四叉树位置str转为四叉树编码list

    :param pos_str: 真实四叉树位置str
    :return: 四叉树编码list
    """

    path = getNodeSearchPath(pos_str)
    code_list = cvtPosListToFullQuadCodeList(path)
    return code_list


def cvtPosStrToFullQuadCodeStr(pos_str):
    """
    将真实四叉树位置str转为四叉树编码str

    :param pos_str: 真实四叉树位置str
    :return: 四叉树编码str
    """
    code_str = ""
    code = cvtPosStrToFullQuadCodeList(pos_str)
    for i in range(len(code)):
        code_str += code[i].__str__().zfill(LEVEL_LENGTH[i])
    return code_str


def cvtQuadCodeStrToPosList(code_str):
    """
    将四叉树编码str转为真实四叉树位置list

    :param code_str: 四叉树编码str
    :return: 真实四叉树位置list
    """

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
    """
    将四叉树编码list转为真实四叉树位置list

    :param code_list: 四叉树编码list
    :return: 真实四叉树位置list
    """

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
    """
    将四叉树编码str转为真实四叉树位置str

    :param code_str: 四叉树编码str
    :return: 真实四叉树位置str
    """

    pos_list = cvtQuadCodeStrToPosList(code_str)
    pos_str = ""
    for i in range(len(pos_list)):
        pos_str += pos_list[i].__str__().zfill(LEVEL_LENGTH[i])
    return pos_str


def cvtQuadCodeListToPosStr(code_list):
    """
    将四叉树编码list转为真实四叉树位置str

    :param code_list: 四叉树编码list
    :return: 真实四叉树位置str
    """

    pos_list = cvtQuadCodeListToPosList(code_list)
    depth = len(pos_list)
    pos_str = ""
    for i in range(depth):
        pos_str += pos_list[i].__str__().zfill(INDEX_TABLE[i])
    return pos_str


if __name__ == '__main__':
    # step 1 :建立节点列表用于存放待插入树的节点
    nodes = []

    # 第一层根节点
    nodes.append((cvtPosListToFullQuadCodeStr([1]), 200))

    # 第二层
    nodes.append((cvtPosListToFullQuadCodeStr([1, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 4]), 100))

    # 第三层第一部分
    nodes.append((cvtPosListToFullQuadCodeStr([1, 1, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 1, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 1, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 1, 4]), 100))
    # 第三层第二部分
    nodes.append((cvtPosListToFullQuadCodeStr([1, 3, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 3, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 3, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 3, 4]), 100))

    # 第四层第一部分
    nodes.append((cvtPosListToFullQuadCodeStr([1, 3, 4, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 3, 4, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 3, 4, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 3, 4, 4]), 100))
    # 第四层第二部分
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 4]), 100))

    # 第五层第一部分
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 4]), 100))

    # 第六层
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4]), 100))

    # 第七层
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 4]), 100))

    # 第八层第一部分
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 2, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 2, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 2, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 2, 4]), 100))
    # 第八层第二部分
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 4]), 100))

    # 第九层
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 4]), 100))

    # 第十层
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 4]), 100))

    # 第十一层第一部分
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 1, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 1, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 1, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 1, 4]), 100))
    # 第十一层第二部分
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 4, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 4, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 4, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 4, 4]), 100))

    # 第十二层
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 4, 2, 1]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 4, 2, 2]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 4, 2, 3]), 100))
    nodes.append((cvtPosListToFullQuadCodeStr([1, 2, 1, 2, 3, 4, 1, 3, 3, 4, 2, 4]), 100))

    # step 2:将节点列表中的节点保存为文件
    saveNodes(nodes, "nodes.tree")

    # step 3:读取节点文件中的数据结构及数据
    nodes, max_depth = readNodes("nodes.tree")

    # step 4:根据读取的数据恢复四叉树结构(无效节点为None)，并将节点依次插入
    tree = buildFullQuad(max_depth)
    for i in range(len(nodes)):
        tree = insertNode(tree, nodes[i][0], nodes[i][1])

    # step 5:绘制恢复的四叉树结构并保存
    img_tree = drawTree2(tree, nodes)
    cv2.imwrite("quadtree.png", img_tree)
