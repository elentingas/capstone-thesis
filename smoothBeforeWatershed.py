import numpy as np
import cv2
import math
import queue
from random import *
import matplotlib.pyplot as plt

class Node:
    def __init__(self):
        self.value = -1
        self.i = -1
        self.j = -1
        self.is_min = False
        self.visitors = queue.Queue( maxsize = (8) ) 
        self.visits = [-1, -1]
        self.color = [-1, -1, -1]

class Image:

    def __init__(self):
        self.image = None

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

        maxLC = picture.getMaximumFromPixelValues(lc)

        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):
                # lc[i][j] = lc[i][j] / maxLC * 255
                picture.image[i, j] = lc[i][j]
                print(lc[i][j])

        return picture

    def getMaximumFromPixelValues(self, lc):

        max = lc[0][0]
        for i in range(len(lc)):
            for j in range(len(lc[0])):
                if lc[i][j] > max:
                    max = lc[i][j]
        return max    

    def watershedVisit(self, p, nodes):
        if not p.is_min:
            if p.visits[0] == -1:

                pTarget = self.findTargetPixel(p, nodes)

                if pTarget.i == p.i and pTarget.j == p.j:
                    p.is_min = True
                    p.color = [ randint(20,200), randint(20,200), randint(20,200) ]
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
    
    def colorPaths(self, p, nodes):
        
        q = queue.Queue( maxsize = (len(nodes) * len(nodes[0])) ) 
        q.put(p)
        color = p.color
        
        while not q.empty():
            pixel = q.get()
            q2 = pixel.visitors
            while not q2.empty():
                newP = q2.get()
                newP.color = color
                q.put(newP)

    def watershedRegionGrowing(self, x, y):

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

        # print('Region Growing Algorithm Finished')
        return picture                  

    def watershedMR1998(self):

        picture = Image()
        picture.copy_image(self)

        resolved_picture = Image()
        resolved_picture.copy_image(self)

        sln = picture.constructSLN() # steepest lower neighbor array

        for i in range(len(picture.image)):
            for j in range(len(picture.image[0])):
                p = picture.resolve([i, j], sln)
                if p[0] != -1 and p[1] != -1:
                    resolved_picture.image[i][j] = [0, 0, 0]
                else:
                    resolved_picture.image[i][j] = [255, 255, 255]
                
        return resolved_picture

    def resolve(self, p, sln):

        # Returns canonical element of pixel p, or
        # WSHED=(-1,-1) in case p lies on a watershed
        
        i = 1
        ce = [0, 0]

        while i <= 4 and ce[0] != -1 and ce[1] != -1 and sln[p[0]][p[1]][i][0] != -2:
            if sln[p[0]][p[1]][i][0] != p[0] and sln[p[0]][p[1]][i][1] != p[1] and sln[p[0]][p[1]][i][0] != -1 and sln[p[0]][p[1]][i][1] != -1:
                sln[p[0]][p[1]][i] = self.resolve(self, sln[p[0]][p[1]][i] ,sln)
            if i == 1:
                ce = sln[p[0]][p[1]][1]
            else:
                for i in range(5):
                    if i != 0:
                        sln[p[0]][p[1]][i] = [-1, -1]
            i = i + 1                

        return ce

    def constructSLN(self):

        sln = np.full((len(self.image),len(self.image[0]),5,2), 0)
        
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                for k in range(5):
                    if k != 0:
                        sln[i][j][k] = self.get_ordered_neighbor(i, j, k)                
        return sln

    def get_ordered_neighbor(self, i, j, k):
        # [-2, -2] means the kth minima does not exist
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
        if index1 != -2:
            dif_array[index1] = [-2, -2, -2]


        index2 = -2
        for m in range(len(n)):
            if dif_array[m][0] != -2:
                if dif_array[m][2] > max2dif[2]:
                    max2dif = dif_array[m]
                    index2 = m            

        max2 = [ max2dif[0], max2dif[1] ] 
        if index2 != -2:
            dif_array[index2] = [-2, -2, -2]


        index3 = -2
        for m in range(len(n)):
            if dif_array[m][0] != -2:
                if dif_array[m][2] > max3dif[2]:
                    max3dif = dif_array[m]
                    index3 = m            

        max3 = [ max3dif[0], max3dif[1] ] 
        if index3 != -2:
            dif_array[index3] = [-2, -2, -2]

        index4 = -2
        for m in range(len(n)):
            if dif_array[m][0] != -2:
                if dif_array[m][2] > max4dif[2]:
                    max4dif = dif_array[m]
                    index4 = m            

        max4 = [ max4dif[0], max4dif[1] ] 
        if index4 != -2:
            dif_array[index4] = [-2, -2, -2]


        if k == 1:
            print(max1)
            return max1
        elif k == 2:
            print(max2)
            return max2
        elif k == 3:
            print(max3)
            return max3
        elif k == 4:
            print(max4)
            return max4
        else:
            return [-2, -2]

    def perform_convolution(self, kernel):
        newImage2 = Image()
        newImage2.copy_image(self)
        for i in range(len(self.image) - 2):
             for j in range(len(self.image[0]) - 2):
                value = self.image[i, j] * kernel[0][0] + self.image[i, j + 1] * kernel[0][1] + self.image[i, j + 2] * kernel[0][2] + self.image[i + 1, j] * kernel[1][0] + self.image[i + 1, j + 1] * kernel[1][1] + self.image[i + 1, j + 2] * kernel[1][2] + self.image[i + 2, j] * kernel[2][0] + self.image[i + 2, j + 1] * kernel[2][1] + self.image[i + 2, j + 2] * kernel[2][2]
                newImage2.image[i + 1, j + 1] = value
        self.image = newImage2.image      


    def perform_5x5_convolution(self, kernel):
        newImage2 = Image()
        newImage2.copy_image(self)
        for i in range(len(self.image) - 4):
             for j in range(len(self.image[0]) - 4):
                value = self.image[i, j] * kernel[0][0] + self.image[i, j + 1] * kernel[0][1] + \
                        self.image[i, j + 2] * kernel[0][2] + self.image[i, j + 3] * kernel[0][3] + \
                        self.image[i, j + 4] * kernel[0][4] + \
                        self.image[i + 1, j] * kernel[1][0] + self.image[i + 1, j + 1] * kernel[1][1] + \
                        self.image[i + 1, j + 2] * kernel[1][2] + self.image[i + 1, j + 3] * kernel[1][3] + \
                        self.image[i + 1, j + 4] * kernel[1][4] + \
                        self.image[i + 2, j] * kernel[2][0] + self.image[i + 2, j + 1] * kernel[2][1] + self.image[i + 2, j + 2] * kernel[2][2] + \
                        self.image[i + 2, j + 3] * kernel[2][3] + self.image[i + 2, j + 4] * kernel[2][4] + \
                        self.image[i + 3, j] * kernel[3][0] + self.image[i + 3, j + 1] * kernel[3][1] + self.image[i + 3, j + 2] * kernel[3][2] + \
                        self.image[i + 3, j + 3] * kernel[3][3] + self.image[i + 3, j + 4] * kernel[3][4] + \
                        self.image[i + 4, j] * kernel[4][0] + self.image[i + 4, j + 1] * kernel[4][1] + self.image[i + 4, j + 2] * kernel[4][2] + \
                        self.image[i + 4, j + 3] * kernel[4][3] + self.image[i + 4, j + 4] * kernel[4][4]

                newImage2.image[i + 1, j + 1] = value
        self.image = newImage2.image      


    def blur(self, function, kernel, repeat):
        for i in range(repeat):
            function(kernel)


def main ():

    img = Image()

    # img.read_image('klor.png')
    img.read_image('GM.jpg')
    # img.read_image('grayscale.jpg')
    # img.read_image('smoothImage')
    # img.read_image('topological.png')
    # img.read_image('check.png')

    img.covert_to_grayscale_avg()

    img.wrap_boundary_copy()
    box_blur_kernel = [[1./9, 1./9, 1./9], [1./9, 1./9, 1./9], [1./9, 1./9, 1./9]]
    box_blur_kernel_5x5 = [[1./25, 1./25, 1./25, 1./25, 1./25], [1./25, 1./25, 1./25, 1./25, 1./25], [1./25, 1./25, 1./25, 1./25, 1./25], [1./25, 1./25, 1./25, 1./25, 1./25], [1./25, 1./25, 1./25, 1./25, 1./25], ]

    gaussian_blur_kernel = [[1./16, 1./8, 1./16], [1./8, 1./4, 1./8], [1./16, 1./8, 1./16]]

    gaussian_blur_kernel_5x5 = [[1./273, 4./273, 7./273, 4./273, 1./273], 
                                [4./273, 16./273, 26./273, 16./273, 4./273], 
                                [7./273, 26./273, 41./273, 26./273, 7./273], 
                                [4./273, 16./273, 26./273, 16./273, 4./273], 
                                [1./273, 4./273, 7./273, 4./273, 1./273]]

    # img.blur(img.perform_5x5_convolution, box_blur_kernel_5x5, 5)
    img.blur(img.perform_5x5_convolution, gaussian_blur_kernel_5x5, 5)
    # img.blur(img.perform_convolution, box_blur_kernel, 15)
    # img.blur(img.perform_convolution, gaussian_blur_kernel, 15)

    img.write_image('smoothImage')



if __name__=="__main__":
    main()

cv2.waitKey(0)
cv2.destroyAllWindows()            
