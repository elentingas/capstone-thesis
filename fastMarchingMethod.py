import numpy as np
import cv2
import math
import queue
import heapq
import matplotlib.pyplot as plt
from random import *

INF = 99999

class Image:

    def __init__(self):
        self.image = None

    def get_height(self):
        return len(self.image)

    def read_image(self, path):
        self.image = cv2.imread(path, 1)

    def write_image(self, title):
        cv2.imwrite(title + '.jpg', self.image)

    def show_image(self, title):
        cv2.imshow(title, self.image)

    def covert_to_grayscale_avg(self):
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                value = self.image[i, j][0] * 1. / 3 + self.image[i, j][1] * 1. / 3 + self.image[i, j][2] * 1. / 3
                self.image[i, j] = [value, value, value]

    def get_width(self):
        return len(self.image[0])

    def wrap_boundary_copy(self):
        newImage = np.zeros((self.get_height() + 2, self.get_width() + 2, 3), np.uint8)

        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                newImage[i + 1][j + 1] = self.image[i][j]

        for i in range(len(self.image[0])):
            newImage[0][i + 1] = self.image[0][i]
            newImage[len(newImage) - 1][i + 1] = self.image[len(self.image) - 1][i]

        for i in range(len(self.image)):
            newImage[i + 1][0] = self.image[i][0]
            newImage[i + 1][len(newImage[0]) - 1] = self.image[i][len(self.image[0]) - 1]

        newImage[0][0] = self.image[0][0]
        newImage[0][len(newImage[0]) - 1] = self.image[0][len(self.image[0]) - 1]
        newImage[len(newImage) - 1][0] = self.image[len(self.image) - 1][0]
        newImage[len(newImage) - 1][len(newImage[0]) - 1] = self.image[len(self.image) - 1][len(self.image[0]) - 1]

        self.image = newImage

    def copy_image(self, other_image):
        self.image = np.copy(other_image.image)

    def fastMarching(self, x, y, fg, picture, u_array):

        # thresholds = [[350, [randint(20, 200), randint(20, 200), randint(20, 200)]],
        #              [300, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
        #              [250, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
        #              [200, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
        #              [150, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
        #              [100, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
        #              [50, [randint(20, 200), randint(20, 200), randint(20, 200)]]]

        thresholds = [
                     [2400, [randint(20, 200), randint(20, 200), randint(20, 200)]],
                     [1600, [randint(20, 200), randint(20, 200), randint(20, 200)]],
                     [1200, [randint(20, 200), randint(20, 200), randint(20, 200)]],
                     [1000, [randint(20, 200), randint(20, 200), randint(20, 200)]],
                     [800, [randint(20, 200), randint(20, 200), randint(20, 200)]],
                     [600, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
                     [300, [randint(20, 200), randint(20, 200), randint(20, 200)]]]
        
        # bg = [99, 30, 233]
        bg = [0, 0, 0]
        #if not fg:
        # fg = [212, 188, 0]

        x = (np.int32)(x)
        y = (np.int32)(y)

        discovery_matrix = np.zeros((self.get_height(), self.get_width()), np.uint8)

        startPixel = [y, x]
        start = (np.int32)(self.image[y, x][0])

        pixels = []
        u_array[y][x] = 0
        heapq.heappush(pixels, (0, startPixel))

        while len(pixels) != 0:

            pop_min = heapq.heappop(pixels)
            current = pop_min[1]

            #print('Currently on ', current)

            if pop_min[0] <= u_array[current[0]][current[1]] and current[0] > 1 and current[1] > 1 and current[0] < (self.get_height() - 2) and current[1] < (self.get_width() - 2):

                pixel1 = [current[0] + 1, current[1]]  # horizontal

                pixel1uprev = u_array[pixel1[0]][pixel1[1]]
                if pixel1uprev > u_array[current[0]][current[1]]:
                    pixel1directions = self.find_directions(u_array, pixel1)
                    pixel1unew = self.solve_u(u_array, pixel1, pixel1directions[0], pixel1directions[1])

                    if pixel1unew < pixel1uprev:
                        u_array[pixel1[0]][pixel1[1]] = pixel1unew
                        heapq.heappush(pixels, (pixel1unew, pixel1))

                pixel2 = [current[0] - 1, current[1]]  # horizontal

                pixel2uprev = u_array[pixel2[0]][pixel2[1]]
                if pixel2uprev > u_array[current[0]][current[1]]:
                    pixel2directions = self.find_directions(u_array, pixel2)
                    pixel2unew = self.solve_u(u_array, pixel2, pixel2directions[0], pixel2directions[1])

                    if pixel2unew < pixel2uprev:
                        u_array[pixel2[0]][pixel2[1]] = pixel2unew
                        heapq.heappush(pixels, (pixel2unew, pixel2))

                pixel3 = [current[0], current[1] + 1]  # vertical

                pixel3uprev = u_array[pixel3[0]][pixel3[1]]
                if pixel3uprev > u_array[current[0]][current[1]]:
                    pixel3directions = self.find_directions(u_array, pixel3)
                    pixel3unew = self.solve_u(u_array, pixel3, pixel3directions[0], pixel3directions[1])

                    if pixel3unew < pixel3uprev:
                        u_array[pixel3[0]][pixel3[1]] = pixel3unew
                        heapq.heappush(pixels, (pixel3unew, pixel3))

                pixel4 = [current[0], current[1] - 1]  # vertical

                pixel4uprev = u_array[pixel4[0]][pixel4[1]]
                if pixel4uprev > u_array[current[0]][current[1]]:
                    pixel4directions = self.find_directions(u_array, pixel4)
                    pixel4unew = self.solve_u(u_array, pixel4, pixel4directions[0], pixel4directions[1])

                    if pixel4unew < pixel4uprev:
                        u_array[pixel4[0]][pixel4[1]] = pixel4unew
                        heapq.heappush(pixels, (pixel4unew, pixel4))

        for r in range(len(thresholds)):
            for i in range(len(u_array)):
                for j in range(len(u_array[0])):
                    if u_array[i][j] <= thresholds[r][0]:
                        picture.image[i][j] = thresholds[r][1]
  
        return picture

    def find_directions(self, u_array, p):

        px1 = [p[0] + 1, p[1]]
        px2 = [p[0] - 1, p[1]]

        py1 = [p[0], p[1] + 1]
        py2 = [p[0], p[1] - 1]

        pv = u_array[p[0]][p[1]]

        px1v = u_array[px1[0]][px1[1]]
        px2v = u_array[px2[0]][px2[1]]

        py1v = u_array[py1[0]][py1[1]]
        py2v = u_array[py2[0]][py2[1]]

        ux = min(px1v, px2v)

        if ux >= pv:
            ux = INF

        uy = min(py1v, py2v)

        if uy >= pv:
            uy = INF


        return [ux, uy]

    def solve_u(self, u_array, p, ux, uy):

        gm = (np.int64)(self.image[p[0]][p[1]][0])
        fInv = gm*gm + 1

        a = 0
        b = 0
        c = - fInv * fInv
        # c = - 1 / 4.

        if uy != INF:
            a = a + 1
            b = b - 2 * uy
            c = c + uy * uy
        if ux != INF:
            a = a + 1
            b = b - 2 * ux
            c = c + ux * ux

        disc = b * b - 4 * a * c
        if disc >= 0:
            sqrt_disc = math.sqrt(disc)
            u1 = (-b + sqrt_disc) / (2. * a)
            #print('Root is -> ', u1)
        else:
            #print("Discriminant negative")
            u1 = fInv + min(ux, uy)
        return u1



def main():
    def u1(gm):
        return gm + 1

    def u2(gm):
        return math.exp(gm)

    img = Image()
    imgColor = Image()

    img.read_image('GM.jpg')

    imgColor.read_image('image.jpg')

    for i in range(len(imgColor.image)):
        for j in range(len(imgColor.image[0])):
            imgColor.image[i, j] = [imgColor.image[i, j][2], imgColor.image[i, j][1], imgColor.image[i, j][0]]

    imgColor.wrap_boundary_copy()
    img.wrap_boundary_copy()
    imgColor.wrap_boundary_copy()
    imgColor.wrap_boundary_copy()
    img.wrap_boundary_copy()

    plt.figure(1)

    ax = plt.gca()

    fig = plt.gcf()

    implot = ax.imshow(imgColor.image)

    plt.title('Image')

    u_array = np.zeros((img.get_height(), img.get_width()), np.float64)

    for i in range(len(u_array)):
        for j in range(len(u_array[0])):
            u_array[i][j] = INF

    def onclick(event):

        if event.xdata != None and event.ydata != None:
            xx = round(event.xdata)
            yy = round(event.ydata)
            print('Exploring ', xx, yy)

            fg = [randint(20, 200), randint(20, 200), randint(20, 200)]

            img.fastMarching(xx, yy, fg, imgColor, u_array)

            plt.figure(1)
            plt.imshow(imgColor.image)
            plt.title('Fast Marching')
            plt.show()

            for i in range(len(imgColor.image)):
                for j in range(len(imgColor.image[0])):
                    imgColor.image[i, j] = [imgColor.image[i, j][2], imgColor.image[i, j][1], imgColor.image[i, j][0]]
            imgColor.write_image('fastMarching')



    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


if __name__ == "__main__":
    main()

