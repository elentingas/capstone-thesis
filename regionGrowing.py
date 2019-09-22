import numpy as np
import cv2
import math
import queue
import matplotlib.pyplot as plt

class Image:

    def __init__(self):
        self.image = None

    def get_height(self):
        return len(self.image)

    def read_image(self, path):
        self.image = cv2.imread(path, -1)

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

    def regionGrowing(self, x, y):

        threshold = 40
        bg = [99, 30, 233]
        fg = [212, 188, 0]

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

            print('Currently on ',current)

            pixel = (np.int32)(picture.image[current[0], current[1]][0])

            if discovery_matrix[current[0], current[1]] == 0 and current[0] > 1 and current[1] > 1  and current[0] < (self.get_height() - 1) and current[1] < (self.get_width() - 1):

                if abs(pixel - start) < threshold:
                    picture.image[current[0], current[1]] = fg

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
                    if abs(pixel1V - start) < threshold:
                        picture.image[pixel1[0], pixel1[1]] = fg
                        pixels.put(pixel1)           
                else:
                    print('already discovered ', pixel1)

                if discovery_matrix[pixel2[0], pixel2[1]] == 0:
                    if abs(pixel2V - start) < threshold:
                        picture.image[pixel2[0], pixel2[1]] = fg
                        pixels.put(pixel2)         
                else:
                    print('already discovered ', pixel2)         


                if discovery_matrix[pixel3[0], pixel3[1]] == 0:
                    if abs(pixel3V - start) < threshold:
                        picture.image[pixel3[0], pixel3[1]] = fg
                        pixels.put(pixel3)               
                else:
                    print('already discovered ', pixel3)  


                if discovery_matrix[pixel4[0], pixel4[1]] == 0:
                    if abs(pixel4V - start) < threshold:
                        picture.image[pixel4[0], pixel4[1]] = fg
                        pixels.put(pixel4)            
                else:
                    print('already discovered ', pixel4)   


                if discovery_matrix[pixel5[0], pixel5[1]] == 0:
                    if abs(pixel5V - start) < threshold:
                        picture.image[pixel5[0], pixel5[1]] = fg
                        pixels.put(pixel5)             
                else:
                    print('already discovered ', pixel5)   


                if discovery_matrix[pixel6[0], pixel6[1]] == 0:
                    if abs(pixel6V - start) < threshold:
                        picture.image[pixel6[0], pixel6[1]] = fg
                        pixels.put(pixel6)             
                else:
                    print('already discovered ', pixel6) 


                if discovery_matrix[pixel7[0], pixel7[1]] == 0:
                    if abs(pixel7V - start) < threshold:
                        picture.image[pixel7[0], pixel7[1]] = fg
                        pixels.put(pixel7)           
                else:
                    print('already discovered ', pixel7)    


                if discovery_matrix[pixel8[0], pixel8[1]] == 0:
                    if abs(pixel8V - start) < threshold:
                        picture.image[pixel8[0], pixel8[1]] = fg
                        pixels.put(pixel8)            
                else:
                    print('already discovered ', pixel8)    

        print('Region Growing Algorithm Finished')
        return picture          

def main ():

    img = Image()

    img.read_image('image.jpg')

    img.covert_to_grayscale_avg()

    img.wrap_boundary_copy()

    ax = plt.gca()
    fig = plt.gcf()
    implot = ax.imshow(img.image)

    def onclick(event):
        if event.xdata != None and event.ydata != None:

            xx = round(event.xdata)
            yy = round(event.ydata)
            print('Exploring ', xx, yy)

            picture = Image()
            picture = img.regionGrowing(xx, yy)

            plt.imshow(picture.image)
            plt.show()
            picture.write_image('regionGrowing')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


if __name__=="__main__":
    main()

cv2.waitKey(0)
cv2.destroyAllWindows()            
