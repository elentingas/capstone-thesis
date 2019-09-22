import numpy as np
import cv2
import math
import queue
import heapq
from random import *
import matplotlib.pyplot as plt

INF = 9999999

class Node:
    def __init__(self):
        self.value = -1
        self.i = -1
        self.j = -1
        self.is_min = False
        self.visitors = queue.Queue( maxsize = (8) ) 
        self.visits = [-1, -1]
        self.color = [-1, -1, -1]
        self.region_id = -1

class Image:

    def __init__(self):
        self.image = None
        self.region_counter = 1

    def get_height(self):
        return len(self.image)

    def read_image(self, path):
        self.image = cv2.imread(path, 1)

    def write_image(self, title):
        cv2.imwrite( title + '.jpg', self.image)

    def show_image(self, title):
        cv2.imshow(title, self.image)

    def covert_to_grayscale_avg(self):
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                value = self.image[i, j][0] * 1./3 + self.image[i, j][1] * 1./3 + self.image[i, j][2] * 1./3
                self.image[i, j] = [value, value, value]
    
    def get_width(self):
        return len(self.image[0])

    def wrap_boundary_copy(self):
        newImage = np.zeros((self.get_height() + 2, self.get_width() + 2, 3), np.uint8)

        for i in range(len(self.image)):
             for j in range(len(self.image[0])):
                newImage[i+1][j+1] = self.image[i][j]

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

    def watershedBiggestFromSmalls(self):

        picture = Image()
        picture.copy_image(self)

        nodes = [[Node() for j in range(len(picture.image[0]))] for i in range(len(picture.image))]

        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):

                nodes[i][j].i = i
                nodes[i][j].j = j
                nodes[i][j].value = picture.image[i, j][0]

        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):
                picture.watershedVisit(nodes[i][j], nodes)

        print('Path Building Done')

        counter = 0

        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):
                if nodes[i][j].is_min:
                    counter += 1
                    picture.colorPaths(nodes[i][j], nodes)

        print(counter, 'Coloring Done')

        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):
                picture.image[i, j] = nodes[i][j].color
                self.image[i, j] = picture.image[i, j]
                print(nodes[i][j].region_id)
                
        return nodes

    def watershedVisit(self, p, nodes):
        if not p.is_min:
            if p.visits[0] == -1:

                pTarget = self.findTargetPixel(p, nodes)

                if pTarget.i == p.i and pTarget.j == p.j:
                    p.is_min = True
                    p.color = [ randint(20,200), randint(20,200), randint(20,200) ]
                    self.region_counter = self.region_counter + 1
                    p.region_id = self.region_counter
                else:
                    p.color = [0,0,0]
                    pTarget.visitors.put(p)
                    p.visits = [pTarget.i, pTarget.j]
                    self.watershedVisit(pTarget, nodes)

    def findTargetPixel(self, p, nodes):

        neighbors = self.getNeighbors(p, nodes)

        minP = p
        flag = False
        for i in range(len(neighbors)):
            if neighbors[i].value != -1:
                if not flag:
                    if neighbors[i].value < minP.value:
                        minP = neighbors[i]
                        flag = True
                if flag and neighbors[i].value > minP.value and neighbors[i].value < p.value:
                    minP = neighbors[i]
                    
        
        return minP
    
    def getNeighbors(self, p, nodes):

        a = [Node()] * 8

        p0i = p.i + 1
        p0j = p.j + 1

        p1i = p.i + 1
        p1j = p.j - 1
        
        p2i = p.i + 1
        p2j = p.j
        
        p3i = p.i - 1
        p3j = p.j
        
        p4i = p.i - 1
        p4j = p.j + 1
        
        p5i = p.i - 1
        p5j = p.j - 1
        
        p6i = p.i 
        p6j = p.j + 1
        
        p7i = p.i
        p7j = p.j - 1


        if p0i >= 0 and p0i < len(nodes) and p0j >= 0 and p0j < len(nodes[0]):
            a[0] = nodes[p0i][p0j]

        if p1i >= 0 and p1i < len(nodes) and p1j >= 0 and p1j < len(nodes[0]):
            a[1] = nodes[p1i][p1j]

        if p2i >= 0 and p2i < len(nodes) and p2j >= 0 and p2j < len(nodes[0]):
            a[2] = nodes[p2i][p2j]

        if p3i >= 0 and p3i < len(nodes) and p3j >= 0 and p3j < len(nodes[0]):
            a[3] = nodes[p3i][p3j]

        if p4i >= 0 and p4i < len(nodes) and p4j >= 0 and p4j < len(nodes[0]):
            a[4] = nodes[p4i][p4j]

        if p5i >= 0 and p5i < len(nodes) and p5j >= 0 and p5j < len(nodes[0]):
            a[5] = nodes[p5i][p5j]

        if p6i >= 0 and p6i < len(nodes) and p6j >= 0 and p6j < len(nodes[0]):
            a[6] = nodes[p6i][p6j]

        if p7i >= 0 and p7i < len(nodes) and p7j >= 0 and p7j < len(nodes[0]):
            a[7] = nodes[p7i][p7j]

        return a 
    
    def colorPaths(self, p, nodes):
        
        q = queue.Queue( maxsize = (len(nodes) * len(nodes[0])) ) 
        q.put(p)
        color = p.color
        region_id = p.region_id
        
        while not q.empty():
            pixel = q.get()
            q2 = pixel.visitors
            while not q2.empty():
                newP = q2.get()
                newP.color = color
                newP.region_id = region_id
                q.put(newP)

    def watershedRegionGrowing(self, x, y, mark_visited):

        threshold = 40
        bg = [99, 30, 233]
        fg2 = [212, 188, 0]
        fg = [255, 255, 255]

        x = (np.int32)(x)
        y = (np.int32)(y)

        discovery_matrix = np.zeros((self.get_height() , self.get_width()), np.uint8)

        picture = Image()
        picture.copy_image(self)

        startPixel = [y, x]
        start = (np.int32)(picture.image[y, x][0])

        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):

                picture.image[i, j] = bg

        pixels = queue.Queue( maxsize = (self.get_height() * self.get_width()) ) 
        pixels.put(startPixel)

        while not pixels.empty():

            current = pixels.get()

            # print('Currently on ',current)

            pixel = (np.int32)(picture.image[current[0], current[1]][0])

            if discovery_matrix[current[0], current[1]] == 0 and current[0] > 1 and current[1] > 1  and current[0] < (self.get_height() - 1) and current[1] < (self.get_width() - 1):

                discovery_matrix[current[0], current[1]] = 1
                mark_visited[current[0], current[1]] = 1

                pixel1 = [ current[0] + 1, current[1] ]
                pixel1V = (np.int32)(self.image[current[0] + 1, current[1]][0])

                pixel2 = [ current[0] + 1, current[1] + 1 ]
                pixel2V = (np.int32)(self.image[current[0] + 1, current[1] + 1 ][0])

                pixel3 = [ current[0] + 1, current[1] - 1 ]
                pixel3V = (np.int32)(self.image[current[0] + 1, current[1] - 1 ][0])

                pixel4 = [ current[0] - 1, current[1] ]
                pixel4V = (np.int32)(self.image[current[0] - 1, current[1] ][0])

                pixel5 = [ current[0] - 1, current[1] + 1 ]
                pixel5V = (np.int32)(self.image[current[0] - 1, current[1] + 1 ][0])

                pixel6 = [ current[0] - 1, current[1] - 1 ]
                pixel6V = (np.int32)(self.image[current[0] - 1, current[1] - 1 ][0])

                pixel7 = [ current[0], current[1] + 1 ]
                pixel7V = (np.int32)(self.image[current[0], current[1] + 1 ][0])

                pixel8 = [ current[0], current[1] - 1 ]
                pixel8V = (np.int32)(self.image[current[0], current[1] - 1 ][0])


                if discovery_matrix[pixel1[0], pixel1[1]] == 0:
                    if pixel1V > 0:
                        picture.image[pixel1[0], pixel1[1]] = fg
                        pixels.put(pixel1)           
                # else:
                    # print('already discovered ', pixel1)

                if discovery_matrix[pixel2[0], pixel2[1]] == 0:
                    if pixel2V > 0:
                        picture.image[pixel2[0], pixel2[1]] = fg
                        pixels.put(pixel2)         
                # else:
                    # print('already discovered ', pixel2)         


                if discovery_matrix[pixel3[0], pixel3[1]] == 0:
                    if pixel3V > 0:
                        picture.image[pixel3[0], pixel3[1]] = fg
                        pixels.put(pixel3)               
                # else:
                    # print('already discovered ', pixel3)  


                if discovery_matrix[pixel4[0], pixel4[1]] == 0:
                    if pixel4V > 0:
                        picture.image[pixel4[0], pixel4[1]] = fg
                        pixels.put(pixel4)            
                # else:
                    # print('already discovered ', pixel4)   


                if discovery_matrix[pixel5[0], pixel5[1]] == 0:
                    if pixel5V > 0:
                        picture.image[pixel5[0], pixel5[1]] = fg
                        pixels.put(pixel5)             
                # else:
                    # print('already discovered ', pixel5)   


                if discovery_matrix[pixel6[0], pixel6[1]] == 0:
                    if pixel6V > 0:
                        picture.image[pixel6[0], pixel6[1]] = fg
                        pixels.put(pixel6)             
                # else:
                    # print('already discovered ', pixel6) 


                if discovery_matrix[pixel7[0], pixel7[1]] == 0:
                    if pixel7V > 0:
                        picture.image[pixel7[0], pixel7[1]] = fg
                        pixels.put(pixel7)           
                # else:
                    # print('already discovered ', pixel7)    


                if discovery_matrix[pixel8[0], pixel8[1]] == 0:
                    if pixel8V > 0:
                        picture.image[pixel8[0], pixel8[1]] = fg
                        pixels.put(pixel8)            
                # else:
                    # print('already discovered ', pixel8)    

        print('Watershed Algorithm Finished')
        return picture                  

    def watershedLC(self):

        picture = Image()
        picture.copy_image(self)

        lc = [[0 for j in range(len(picture.image[0]))] for i in range(len(picture.image))]

        q = queue.Queue( maxsize = (len(lc) * len(lc[0])) ) 
        
        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):
                lc[i][j] = 0
                if picture.hasSmallerDifferenceNeighbor([i, j]):
                    lc[i][j] = -1
                    q.put([i, j])

        dist = 1
        q.put([-1, -1])

        while not q.empty():
            p = q.get()
            if p[0] == -1 and p[1] == -1:
                if not q.empty():
                    q.put([-1, -1])
                    dist = dist + 1
            else:
                lc[p[0]][p[1]] = dist
                n = picture.getNeighborPixels(p)
                for k in range(len(n)):
                    if n[k][0] != -1:
                        if picture.image[n[k][0], n[k][1]][0] == picture.image[p[0], p[1]][0] and lc[n[k][0]][n[k][1]] == 0:
                            q.put(n[k])
                            lc[n[k][0]][n[k][1]] = -1

        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):
                if lc[i][j] != 0:
                    lc[i][j] = dist * (np.int32)(picture.image[i, j][0]) + lc[i][j] - 1

        # maxLC = picture.getMaximumFromPixelValues(lc)

        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):
                # lc[i][j] = lc[i][j] / maxLC * 255
                picture.image[i, j] = lc[i][j]
                # print(lc[i][j])

        return picture

    def getNeighborPixels(self, p):

        self_value = self.image[p[0]][p[1]][0]
        a = [[-1, -1]] * 8

        p0i = p[0] + 1
        p0j = p[1] + 1

        p1i = p[0] + 1
        p1j = p[1] - 1
        
        p2i = p[0] + 1
        p2j = p[1]
        
        p3i = p[0] - 1
        p3j = p[1]
        
        p4i = p[0] - 1
        p4j = p[1] + 1
        
        p5i = p[0] - 1
        p5j = p[1] - 1
        
        p6i = p[0] 
        p6j = p[1] + 1
        
        p7i = p[0]
        p7j = p[1] - 1

        if p0i >= 0 and p0i < len(self.image) and p0j >= 0 and p0j < len(self.image[0]):
            a[0] = [p0i,p0j]

        if p1i >= 0 and p1i < len(self.image) and p1j >= 0 and p1j < len(self.image[0]):
            a[1] = [p1i,p1j]

        if p2i >= 0 and p2i < len(self.image) and p2j >= 0 and p2j < len(self.image[0]):
            a[2] = [p2i,p2j]

        if p3i >= 0 and p3i < len(self.image) and p3j >= 0 and p3j < len(self.image[0]):
            a[3] = [p3i,p3j]

        if p4i >= 0 and p4i < len(self.image) and p4j >= 0 and p4j < len(self.image[0]):
            a[4] = [p4i,p4j]

        if p5i >= 0 and p5i < len(self.image) and p5j >= 0 and p5j < len(self.image[0]):
            a[5] = [p5i,p5j]

        if p6i >= 0 and p6i < len(self.image) and p6j >= 0 and p6j < len(self.image[0]):
            a[6] = [p6i,p6j]

        if p7i >= 0 and p7i < len(self.image) and p7j >= 0 and p7j < len(self.image[0]):
            a[7] = [p7i,p7j]

        return a 
    
    def hasSmallerDifferenceNeighbor(self, p):

        self_value = self.image[p[0], p[1]][0]
        flag = False

        a = [[-1, -1]] * 8

        p0i = p[0] + 1
        p0j = p[1] + 1

        p1i = p[0] + 1
        p1j = p[1] - 1
        
        p2i = p[0] + 1
        p2j = p[1]
        
        p3i = p[0] - 1
        p3j = p[1]
        
        p4i = p[0] - 1
        p4j = p[1] + 1
        
        p5i = p[0] - 1
        p5j = p[1] - 1
        
        p6i = p[0] 
        p6j = p[1] + 1
        
        p7i = p[0]
        p7j = p[1] - 1

        if p0i >= 0 and p0i < len(self.image) and p0j >= 0 and p0j < len(self.image[0]):
            a[0] = [p0i,p0j]

        if p1i >= 0 and p1i < len(self.image) and p1j >= 0 and p1j < len(self.image[0]):
            a[1] = [p1i,p1j]

        if p2i >= 0 and p2i < len(self.image) and p2j >= 0 and p2j < len(self.image[0]):
            a[2] = [p2i,p2j]

        if p3i >= 0 and p3i < len(self.image) and p3j >= 0 and p3j < len(self.image[0]):
            a[3] = [p3i,p3j]

        if p4i >= 0 and p4i < len(self.image) and p4j >= 0 and p4j < len(self.image[0]):
            a[4] = [p4i,p4j]

        if p5i >= 0 and p5i < len(self.image) and p5j >= 0 and p5j < len(self.image[0]):
            a[5] = [p5i,p5j]

        if p6i >= 0 and p6i < len(self.image) and p6j >= 0 and p6j < len(self.image[0]):
            a[6] = [p6i,p6j]

        if p7i >= 0 and p7i < len(self.image) and p7j >= 0 and p7j < len(self.image[0]):
            a[7] = [p7i,p7j]


        if a[0][0] != -1:
            if self.image[a[0][0],a[0][1]][0] < self_value:
                flag = True
        if a[1][0] != -1:
            if self.image[a[1][0],a[1][1]][0] < self_value:
                flag = True
        if a[2][0] != -1:
            if self.image[a[2][0],a[2][1]][0] < self_value:
                flag = True
        if a[3][0] != -1:
            if self.image[a[3][0],a[3][1]][0] < self_value:
                flag = True
        if a[4][0] != -1:
            if self.image[a[4][0],a[4][1]][0] < self_value:
                flag = True
        if a[5][0] != -1:
            if self.image[a[5][0],a[5][1]][0] < self_value:
                flag = True
        if a[6][0] != -1:
            if self.image[a[6][0],a[6][1]][0] < self_value:
                flag = True
        if a[7][0] != -1:
            if self.image[a[7][0],a[7][1]][0] < self_value:
                flag = True

        return flag 
    
    def fastDashing(self, x, y, fg, picture, u_array, nodes):

        thresholds = [[350, [randint(20, 200), randint(20, 200), randint(20, 200)]],
                     [300, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
                     [250, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
                     [200, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
                     [150, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
                     [100, [randint(20, 200), randint(20, 200), randint(20, 200)]], 
                     [50, [randint(20, 200), randint(20, 200), randint(20, 200)]]]



        # thresholds = [[100, [randint(20, 200), randint(20, 200), randint(20, 200)]],
        #              [90, [randint(20, 200), randint(20, 200), randint(20, 200)]],
        #              [80, [randint(20, 200), randint(20, 200), randint(20, 200)]],
        #              [70, [randint(20, 200), randint(20, 200), randint(20, 200)]],
        #              [50, [randint(20, 200), randint(20, 200), randint(20, 200)]],
        #              [35, [randint(20, 200), randint(20, 200), randint(20, 200)]],
        #              [25, [randint(20, 200), randint(20, 200), randint(20, 200)]]]
        
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

                    if nodes[pixel1[0]][pixel1[1]][0] != nodes[current[0]][current[1]][0] or nodes[pixel1[0]][pixel1[1]][1] != nodes[current[0]][current[1]][1]:

                        pixel1directions = self.find_directions(u_array, pixel1, nodes)
                        pixel1unew = self.solve_u(u_array, pixel1, pixel1directions[0], pixel1directions[1], nodes)

                        if pixel1unew < pixel1uprev:
                            u_array[pixel1[0]][pixel1[1]] = pixel1unew
                            heapq.heappush(pixels, (pixel1unew, pixel1))
                    else:
                        u_array[pixel1[0]][pixel1[1]] = u_array[current[0]][current[1]]
                        heapq.heappush(pixels, (u_array[current[0]][current[1]], pixel1))


                pixel2 = [current[0] - 1, current[1]]  # horizontal

                pixel2uprev = u_array[pixel2[0]][pixel2[1]]
                if pixel2uprev > u_array[current[0]][current[1]]:

                    if nodes[pixel2[0]][pixel2[1]][0] != nodes[current[0]][current[1]][0] or nodes[pixel2[0]][pixel2[1]][1] != nodes[current[0]][current[1]][1]:
                            
                        pixel2directions = self.find_directions(u_array, pixel2, nodes)
                        pixel2unew = self.solve_u(u_array, pixel2, pixel2directions[0], pixel2directions[1], nodes)

                        if pixel2unew < pixel2uprev:
                            u_array[pixel2[0]][pixel2[1]] = pixel2unew
                            heapq.heappush(pixels, (pixel2unew, pixel2))
                    else:
                        u_array[pixel2[0]][pixel2[1]] = u_array[current[0]][current[1]]
                        heapq.heappush(pixels, (u_array[current[0]][current[1]], pixel2))


                pixel3 = [current[0], current[1] + 1]  # vertical

                pixel3uprev = u_array[pixel3[0]][pixel3[1]]
                if pixel3uprev > u_array[current[0]][current[1]]:

                    if nodes[pixel3[0]][pixel3[1]][0] != nodes[current[0]][current[1]][0] or nodes[pixel3[0]][pixel3[1]][1] != nodes[current[0]][current[1]][1]:

                        pixel3directions = self.find_directions(u_array, pixel3, nodes)
                        pixel3unew = self.solve_u(u_array, pixel3, pixel3directions[0], pixel3directions[1], nodes)

                        if pixel3unew < pixel3uprev:
                            u_array[pixel3[0]][pixel3[1]] = pixel3unew
                            heapq.heappush(pixels, (pixel3unew, pixel3))
                    else:
                        u_array[pixel3[0]][pixel3[1]] = u_array[current[0]][current[1]]
                        heapq.heappush(pixels, (u_array[current[0]][current[1]], pixel3))


                pixel4 = [current[0], current[1] - 1]  # vertical

                pixel4uprev = u_array[pixel4[0]][pixel4[1]]
                if pixel4uprev > u_array[current[0]][current[1]]:

                    if nodes[pixel4[0]][pixel4[1]][0] != nodes[current[0]][current[1]][0] or nodes[pixel4[0]][pixel4[1]][1] != nodes[current[0]][current[1]][1]:
                            
                        pixel4directions = self.find_directions(u_array, pixel4, nodes)
                        pixel4unew = self.solve_u(u_array, pixel4, pixel4directions[0], pixel4directions[1], nodes)

                        if pixel4unew < pixel4uprev:
                            u_array[pixel4[0]][pixel4[1]] = pixel4unew
                            heapq.heappush(pixels, (pixel4unew, pixel4))
                    else:
                        u_array[pixel4[0]][pixel4[1]] = u_array[current[0]][current[1]]
                        heapq.heappush(pixels, (u_array[current[0]][current[1]], pixel4))



        for r in range(len(thresholds)):
            for i in range(len(u_array)):
                for j in range(len(u_array[0])):
                    if u_array[i][j] <= thresholds[r][0]:
                        picture.image[i][j] = thresholds[r][1]
  
        return picture

    def find_directions(self, u_array, p, nodes):

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

    def solve_u(self, u_array, p, ux, uy, nodes):

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

    def getNeighborPixelsFour(self, p):

        a = [[-1, -1]] * 4

        p0i = p[0] + 1
        p0j = p[1]
        
        p1i = p[0] - 1
        p1j = p[1]
        
        p2i = p[0] 
        p2j = p[1] + 1
        
        p3i = p[0]
        p3j = p[1] - 1

        if p0i >= 0 and p0i < len(self.image) and p0j >= 0 and p0j < len(self.image[0]):
            a[0] = [p0i,p0j]

        if p1i >= 0 and p1i < len(self.image) and p1j >= 0 and p1j < len(self.image[0]):
            a[1] = [p1i,p1j]

        if p2i >= 0 and p2i < len(self.image) and p2j >= 0 and p2j < len(self.image[0]):
            a[2] = [p2i,p2j]

        if p3i >= 0 and p3i < len(self.image) and p3j >= 0 and p3j < len(self.image[0]):
            a[3] = [p3i,p3j]

        return a 


    def watershedMR1998(self):

        picture = Image()
        picture.copy_image(self)

        resolved_picture = Image()
        resolved_picture.copy_image(self)

        constructionResult = picture.constructSLN() # steepest lower neighbor array
        sln = constructionResult[0] # steepest lower neighbor array
        CEarr = constructionResult[1]

        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):
                p = picture.resolve([i, j], sln, CEarr)
                CEarr[i][j] = p
                if p[0] != -1 and p[1] != -1:
                    resolved_picture.image[i][j] = [230, 230, 230]
                else:
                    resolved_picture.image[i][j] = [20, 20, 20]

        #for i in range(len(picture.image)):
        #  for j in range(len(picture.image[0])):
        #      print(CEarr[i][j], end = '      ')
        #  print('')

        # combining the minima
        disc_matrix = np.full((len(self.image),len(self.image[0])), 0)
        color_space = np.full((len(self.image),len(self.image[0]), 3), 0)
        q = queue.Queue( maxsize = (len(self.image) * len(self.image[0])) )
        
        for i in range(len(picture.image)):
          for j in range(len(picture.image[0])):
              if CEarr[i][j][0] == i and CEarr[i][j][1] == j:
                  if disc_matrix[i][j] == 0:
                      q.put([i, j])
                      while not q.empty():
                            px = q.get()
                            if disc_matrix[px[0]][px[1]] == 0:
                                disc_matrix[px[0]][px[1]] = 1
                                CEarr[px[0]][px[1]] = [i, j]                                
                                n = self.getNeighborPixelsFour(px)
                                for r in range(len(n)):
                                        if n[r][0] != -1:
                                            if CEarr[n[r][0]][n[r][1]][0] == n[r][0] and CEarr[n[r][0]][n[r][1]][1] == n[r][1]:
                                                q.put(n[r])
                                        


        for i in range(len(picture.image)):
          for j in range(len(picture.image[0])):
              color_space[i][j] = [ randint(20,200), randint(20,200), randint(20,200) ]
              #if sln[i][j][0][0] != -7:
              #    CEarr[i][j] = CEarr[CEarr[i][j][0]][CEarr[i][j][1]]


        #for i in range(len(picture.image)):
        #  for j in range(len(picture.image[0])):
        #      resolved_picture.image[i][j] = color_space[CEarr[i][j][0]][CEarr[i][j][1]]
              
        #for i in range(len(picture.image)):
        #  for j in range(len(picture.image[0])):
        #      print(CEarr[i][j], end = '      ')
        #  print('')                            
                  
          
        return CEarr

    def resolve(self, p, sln, CEarr):

        # Returns canonical element of pixel p, or
        # WSHED=(-1,-1) in case p lies on a watershed.
        
        i = 1
        ce = [0, 0]

        while i <= 4 and ce[0] != -1:
            if sln[p[0]][p[1]][i][0] != -2:
                if sln[p[0]][p[1]][i][0] != -1 and sln[p[0]][p[1]][0][0] != -7:
                    sln[p[0]][p[1]][i] = self.resolve(sln[p[0]][p[1]][i] ,sln, CEarr)
                if i == 1:
                    ce = sln[p[0]][p[1]][1]
                elif sln[p[0]][p[1]][i][0] != ce[0] or sln[p[0]][p[1]][i][1] != ce[1]:
                    ce = [-1, -1]
                    for i in range(5):
                        if i != 0:
                            sln[p[0]][p[1]][i] = [-1, -1]
            i = i + 1                

        return ce

    def constructSLN(self):
        # sln[i][j][0] = [-7, -7] means the i, j pixel is a minimum
        sln = np.full((len(self.image),len(self.image[0]),5,2), -2)
        CEarr = np.full((len(self.image),len(self.image[0]),2), -3)

        
        #for i in range(len(self.image)):
        #    for j in range(len(self.image[0])):
        #        print(CEarr[i][j], end = '      ')
        #    print('')

        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                for k in range(5):
                    if k != 0:
                        sln[i][j][k] = self.get_ordered_neighbor(i, j, k, sln)
                if self.image[i][j][0] == 0:
                    sln[i][j][0] = [-7, -7]
        return [sln, CEarr]

    def get_ordered_neighbor(self, i, j, k, sln):
        # [-2, -2] means the kth minima does not exist
        p = [i, j]
        n = self.getNeighborPixelsFour([i, j])
        dif_array = [[-2,-2,-2], [-2,-2,-2], [-2,-2,-2], [-2,-2,-2]]

        max1 = [-2, -2]
        max2 = [-2, -2]
        max3 = [-2, -2]
        max4 = [-2, -2]

        max1dif = [-2, -2, 0]
        max2dif = [-2, -2, 0]
        max3dif = [-2, -2, 0]
        max4dif = [-2, -2, 0]

        for r in range(len(n)):
            if n[r][0] != -1:
                if (np.int32)(self.image[i, j][0]) - (np.int32)(self.image[n[r][0], n[r][1]][0]) > 0:
                    dif_array[r] = [n[r][0], n[r][1], (np.int32)(self.image[i, j][0]) - (np.int32)(self.image[n[r][0], n[r][1]][0])]


        index1 = -2
        for m in range(len(n)):
            if dif_array[m][0] != -2:
                if dif_array[m][2] > max1dif[2]:
                    max1dif = dif_array[m]
                    index1 = m            

        max1 = [ max1dif[0], max1dif[1] ]
        
        

        if index1 == -2:
            sln[p[0]][p[1]][0] = [-7, -7]
            return p
        else:
            
            temp = dif_array[0]
            dif_array[0] = dif_array[index1]
            dif_array[index1] = temp
            
            if k == 1:
                    return [ dif_array[0][0], dif_array[0][1] ]
            if k == 2:
                if dif_array[1][2] ==  dif_array[0][2]:
                    return [ dif_array[1][0], dif_array[1][1] ]
                else:
                    return [-2, -2]
            if k == 3:
                if dif_array[2][2] ==  dif_array[0][2]:
                    return [ dif_array[2][0], dif_array[2][1] ]
                else:
                    return [-2, -2]
            if k == 4:
                if dif_array[3][2] ==  dif_array[0][2]:
                    return [ dif_array[3][0], dif_array[3][1] ]
                else:
                    return [-2, -2]
    

def main ():
    
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

    picture = Image()
    picture = img.watershedLC()
    nodes = picture.watershedMR1998()

    def onclick(event):

        if event.xdata != None and event.ydata != None:
            xx = round(event.xdata)
            yy = round(event.ydata)
            print('Exploring ', xx, yy)

            fg = [randint(20, 200), randint(20, 200), randint(20, 200)]

            img.fastDashing(xx, yy, fg, imgColor, u_array, nodes)

            plt.figure(1)
            plt.imshow(imgColor.image)
            plt.title('Fast Dashing')
            plt.show()

            for i in range(len(imgColor.image)):
                for j in range(len(imgColor.image[0])):
                    imgColor.image[i, j] = [imgColor.image[i, j][2], imgColor.image[i, j][1], imgColor.image[i, j][0]]
            imgColor.write_image('fastDashing')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


if __name__=="__main__":
    main()

cv2.waitKey(0)
cv2.destroyAllWindows()            
