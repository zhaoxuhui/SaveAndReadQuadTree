# coding=utf-8
import cv2
import quadTree
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os


def findAllFiles(root_dir, filter):
    """
    在指定目录查找指定类型文件

    :param root_dir: 查找目录
    :param filter: 文件类型
    :return: 路径、名称、文件全路径

    """

    print("Finding files ends with \'" + filter + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    files = []
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print (names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    return paths, names, files


def drawplot(counters, xs, ys, name='figure'):
    max_elmt = max(counters)
    x = []
    y = []
    z = []
    dx = []
    dy = []
    dz = []
    colors = []
    for i in range(len(counters)):
        if counters[i] != 0:
            x.append(xs[i])
            y.append(ys[i])
            z.append(0)
            dx.append(20)
            dy.append(20)
            dz.append(counters[i])
            if counters[i] < 0.2 * max_elmt:
                colors.append('steelblue')
            elif 0.2 * max_elmt <= counters[i] < 0.4 * max_elmt:
                colors.append('lightskyblue')
            elif 0.4 * max_elmt <= counters[i] < 0.6 * max_elmt:
                colors.append('yellowgreen')
            elif 0.6 * max_elmt <= counters[i] < 0.8 * max_elmt:
                colors.append('orange')
            elif 0.8 * max_elmt <= counters[i] <= 1.0 * max_elmt:
                colors.append('orangered')

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(name)
    ax.set_xlabel('pixels in x direction')
    ax.set_ylabel('pixels in y direction')
    ax.set_zlabel('height error')
    ax.bar3d(x, y, z, dx, dy, dz, color=colors)
    plt.show()


if __name__ == '__main__':
    img = cv2.imread("test.tif", cv2.IMREAD_GRAYSCALE)
    _, names, trees = findAllFiles(".", ".tree")

    mean_x = []
    mean_errors = []
    counter = 5000
    x = []
    y = []
    errors = []
    for k in range(len(trees)):
        nodes = quadTree.decompressAndLoadData(trees[k])
        restore = quadTree.restoreImageFromNodes(nodes, img.shape[1], img.shape[0])
        cv2.imwrite("restore_" + trees[k] + ".png", restore)
        error = []
        for i in range(0, img.shape[1], 400):
            for j in range(0, img.shape[0], 400):
                if k == 0:
                    x.append(i)
                    y.append(j)
                error.append(abs(int(restore[j, i]) - int(img[j, i])))

        errors.append(error)
        mean_error = np.mean(error)
        mean_errors.append(mean_error)
        mean_x.append(counter)
        counter += 5000
        print trees[k], 'mean error', mean_error

    drawplot(errors[0], x, y, names[0] + ' error')
    drawplot(errors[-1], x, y, names[-1] + ' error')

    plt.plot(mean_x, mean_errors)
    plt.show()
