import numpy as np
import cv2
import math


class Image:

    def __init__(self):
        self.image = None

    def get_height(self):
        return len(self.image)

    def read_image(self, path: str):
        self.image = cv2.imread(path, -1)

    def write_image(self, title: str):
        cv2.imwrite( title + '.jpg', self.image)

    def show_image(self, title: str):
        cv2.imshow(title, self.image)

    def covert_to_grayscale_avg(self):
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                value = self.image[i, j][0] * 1./3 + self.image[i, j][1] * 1./3 + self.image[i, j][2] * 1./3
                self.image[i, j] = [value, value, value]

    def isolate_blue_channel(self):
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                self.image[i, j] = [self.image[i, j][0], 0, 0]

    def isolate_green_channel(self):
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                self.image[i, j] = [0, self.image[i, j][1], 0]

    def isolate_red_channel(self):
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                self.image[i, j] = [0, 0, self.image[i, j][2]]

    def covert_to_grayscale_min(self):
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                value = min(min(self.image[i, j][0], self.image[i, j][1]), self.image[i, j][2])
                self.image[i, j] = [value, value, value]

    def covert_to_grayscale_max(self):
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                value = max(max(self.image[i, j][0], self.image[i, j][1]), self.image[i, j][2])
                self.image[i, j] = [value, value, value]

    def get_width(self):
        return len(self.image[0])

    def perform_convolution(self, kernel):
        newImage2 = Image()
        newImage2.copy_image(self)
        for i in range(len(self.image) - 2):
             for j in range(len(self.image[0]) - 2):
                value = self.image[i, j] * kernel[0][0] + self.image[i, j + 1] * kernel[0][1] + self.image[i, j + 2] * kernel[0][2] + self.image[i + 1, j] * kernel[1][0] + self.image[i + 1, j + 1] * kernel[1][1] + self.image[i + 1, j + 2] * kernel[1][2] + self.image[i + 2, j] * kernel[2][0] + self.image[i + 2, j + 1] * kernel[2][1] + self.image[i + 2, j + 2] * kernel[2][2]
                newImage2.image[i + 1, j + 1] = value
        self.image = newImage2.image      

    def perform_operation(self, kernel):
        newImage2 = Image()
        newImage2.copy_image(self)
        for i in range(len(self.image) - 2):
             for j in range(len(self.image[0]) - 2):
                value = self.image[i, j] * kernel[0][0] + self.image[i, j + 1] * kernel[0][1] + self.image[i, j + 2] * kernel[0][2] + self.image[i + 1, j] * kernel[1][0] + self.image[i + 1, j + 1] * kernel[1][1] + self.image[i + 1, j + 2] * kernel[1][2] + self.image[i + 2, j] * kernel[2][0] + self.image[i + 2, j + 1] * kernel[2][1] + self.image[i + 2, j + 2] * kernel[2][2]
                
                value = value + 127

                if self.max_value < value[0]:
                    self.max_value = value[0]

                if self.min_value > value[0]:
                    self.min_value = value[0]

                if value[0] > 255:
                    value = 255
                    
                if value[0] < 0:
                   value = 0

                newImage2.image[i + 1, j + 1] = value
        self.image = newImage2.image      

    def perform_operation_no_shift(self, kernel):
        newImage2 = Image()
        newImage2.copy_image(self)
        for i in range(len(self.image) - 2):
             for j in range(len(self.image[0]) - 2):
                value = self.image[i, j] * kernel[0][0] + self.image[i, j + 1] * kernel[0][1] + self.image[i, j + 2] * kernel[0][2] + self.image[i + 1, j] * kernel[1][0] + self.image[i + 1, j + 1] * kernel[1][1] + self.image[i + 1, j + 2] * kernel[1][2] + self.image[i + 2, j] * kernel[2][0] + self.image[i + 2, j + 1] * kernel[2][1] + self.image[i + 2, j + 2] * kernel[2][2]
                
                if value[0] < 0:
                    value = -value

                if self.max_value < value[0]:
                    self.max_value = value[0]

                if self.min_value > value[0]:
                    self.min_value = value[0]

                value = value / 127 * 255

                if value[0] > 255:
                    value = 255
                    
                if value[0] < 0:
                   value = 0

                newImage2.image[i + 1, j + 1] = value
        self.image = newImage2.image      

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

    def calculate_gradient_magnitude(self, other_self, another_self):

        newImage3 = Image()
        newImage3.copy_image(another_self)

        for i in range(len(another_self.image)):
             for j in range(len(another_self.image[0])):

                gx = (np.uint32)(another_self.image[i][j][0])
                gy = (np.uint32)(other_self.image[i][j][0])
                valueRes = math.sqrt((gx-127)*(gx-127) + (gy-127)*(gy-127))

                #print(valueRes)

                value = valueRes

                if self.max_value < value:
                    self.max_value = value

                if self.min_value > value:
                    self.min_value = value

                # As the range of gradient magnitude values is [0, 181], we map that range into [0, 255]
                valueRes = valueRes / 181 * 255

                if valueRes > 255:
                    valueRes = 255

                newImage3.image[i][j] = valueRes
                
        self.image = newImage3.image
    
    def calculate_gradient_magnitude_no_shift(self, other_self, another_self):

        newImage3 = Image()
        newImage3.copy_image(another_self)

        for i in range(len(another_self.image)):
             for j in range(len(another_self.image[0])):

                gx = (np.uint32)(another_self.image[i][j][0])
                gy = (np.uint32)(other_self.image[i][j][0])
                valueRes = math.sqrt(gx*gx + gy*gy)

                #print(valueRes)

                value = valueRes

                if self.max_value < value:
                    self.max_value = value

                if self.min_value > value:
                    self.min_value = value

                # As the range of gradient magnitude values is [0, 360], we map that range into [0, 255]
                valueRes = valueRes / 360 * 255

                if valueRes > 255:
                    valueRes = 255

                newImage3.image[i][j] = valueRes
                
        self.image = newImage3.image
        
    def thresholding(self, x):

        for i in range(len(self.image)):
             for j in range(len(self.image[0])):

                value = 0
                if self.image[i, j][0] > x:
                    value = 255

                self.image[i, j] = value
                

def main ():
    img = Image()
    img.read_image('image.jpg')
    img2 = Image()
    img2.copy_image(img)

    identity_kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    box_blur_kernel = [[1./9, 1./9, 1./9], [1./9, 1./9, 1./9], [1./9, 1./9, 1./9]]
    gaussian_blur_kernel = [[1./16, 1./8, 1./16], [1./8, 1./4, 1./8], [1./16, 1./8, 1./16]]

    sobel_kernel_Gx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    sobel_kernel_Gy = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    normalized_sobel_kernel_Gx = [[1./8, 0./8, -1./8], [2./8, 0, -2./8], [1./8, 0, -1./8]]
    normalized_sobel_kernel_Gy = [[1./8, 2./8, 1./8], [0, 0, 0], [-1./8, -2./8, -1./8]]

    prewitt_kernel_Gx = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    prewitt_kernel_Gy = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]

    normalized_prewitt_kernel_Gx = [[1./6, 0, -1./6], [1./6, 0, -1./6], [1./6, 0, -1./6]]
    normalized_prewitt_kernel_Gy = [[1./6, 1./6, 1./6], [0, 0, 0], [-1./6, -1./6, -1./6]]

    img.covert_to_grayscale_avg()
    img.write_image('grayscale')
    img2.covert_to_grayscale_avg()
    
    #img.show_image('image')

    img.wrap_boundary_copy()


    img.max_value = -9999999999999

    img.min_value = 9999999999999


    img.perform_operation(normalized_prewitt_kernel_Gx)

    img2.wrap_boundary_copy()

    img2.max_value = -9999999999999

    img2.min_value = 9999999999999


    img2.perform_operation(normalized_prewitt_kernel_Gy)

    img3 = Image()

    img3.max_value = -9999999999999

    img3.min_value = 9999999999999

    img3.calculate_gradient_magnitude(img, img2)

    #img.show_image('prewitt operator Gx')

    print('\n Gx Max ---> ', img.max_value)
    print('\n Gx Min ---> ', img.min_value)

    #img2.show_image('prewitt operator Gy')

    print('\n Gy Max ---> ', img2.max_value)
    print('\n Gy Min ---> ', img2.min_value)

    #img3.show_image('gradient magnitude')

    print('\n GM Max ---> ', img3.max_value)
    print('\n GM Min ---> ', img3.min_value)

    imgThresh1 = Image()
    imgThresh2 = Image()
    imgThresh3 = Image()

    imgThresh1.copy_image(img3)
    imgThresh2.copy_image(img3)
    imgThresh3.copy_image(img3)

    imgThresh1.thresholding(90)
    #imgThresh1.show_image('thresholding 90')

    imgThresh2.thresholding(50)
    #imgThresh2.show_image('thresholding 50')

    imgThresh3.thresholding(30)
    #imgThresh3.show_image('thresholding 30')

    img.write_image('prewittGxWithShift')
    img2.write_image('prewittGyWithShift')
    img3.write_image('prewittGM')
    img3.write_image('GM')
    #imgThresh1.write_image('boundaryDetection1prewitt')
    imgThresh2.write_image('boundaryDetection1prewitt')
    imgThresh3.write_image('boundaryDetection2prewitt')

if __name__=="__main__":
    main()

cv2.waitKey(0)
cv2.destroyAllWindows()            
