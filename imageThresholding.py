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

                print(valueRes)

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

                print(valueRes)

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
            
    def thresholding_modified(self):

        statistics = [0] * 256

        for i in range(len(self.image)):
             for j in range(len(self.image[0])):

                statistics[self.image[i, j][0]] += 1
      
        # for i in range(256):
        #     print('-> ', i, ' ===> ', statistics[i])

        first_max = -1
        first_max_index = -1
        second_max = -1
        second_max_index = -1

        for i in range(len(statistics)):
            if statistics[i] > first_max:
                first_max = statistics[i]
                first_max_index = i

        statistics[first_max_index] = 0

        for i in range(len(statistics)):
            if statistics[i] > second_max:
                second_max = statistics[i]
                second_max_index = i

        statistics[first_max_index] = first_max

        # not accurate for thresholding
        #print('1st max value is ', first_max, 'with color value ', first_max_index)
        #print('2nd max value is ', second_max, 'with color value ', second_max_index)
        #

        mean = 1./2 * first_max_index + 1./2 * second_max_index
        x = mean

        # interpolate 256 to 5 grayscale colors
        # 0 - 50 maps 25
        # 50 - 100 maps 75
        # 100 - 150 maps 125
        # 150 - 200 maps 175
        # 200 - 256 maps 225

        interpolatedStatistics = [0] * 5

        for i in range(50):
           interpolatedStatistics[0] += statistics[i]

        for i in range(50):
           interpolatedStatistics[1] += statistics[i + 50]

        for i in range(50):
           interpolatedStatistics[2] += statistics[i + 100]

        for i in range(50):
           interpolatedStatistics[3] += statistics[i + 150]

        for i in range(56):
           interpolatedStatistics[4] += statistics[i + 200]

        first_max = -1
        first_max_index = -1
        second_max = -1
        second_max_index = -1

        for i in range(len(interpolatedStatistics)):
            if interpolatedStatistics[i] > first_max:
                first_max = interpolatedStatistics[i]
                first_max_index = i

        interpolatedStatistics[first_max_index] = 0

        for i in range(len(interpolatedStatistics)):
            if interpolatedStatistics[i] > second_max:
                second_max = interpolatedStatistics[i]
                second_max_index = i

        interpolatedStatistics[first_max_index] = first_max

        # 
        print('1st max value is ', first_max, 'with color value ', first_max_index)
        print('2nd max value is ', second_max, 'with color value ', second_max_index)
        #

        first_value = -1
        second_value = -1

        if first_max_index == 4:
            first_value = 225
        elif first_max_index == 3:
            first_value = 175
        elif first_max_index == 2:
            first_value = 125
        elif first_max_index == 1:
            first_value = 75
        elif first_max_index == 0:
            first_value = 25

        if second_max_index == 4:
            second_value = 225
        elif second_max_index == 3:
            second_value = 175
        elif second_max_index == 2:
            second_value = 125
        elif second_max_index == 1:
            second_value = 75
        elif second_max_index == 0:
            second_value = 25
 
        print('1st max value', first_value)
        print('2nd max value', second_value)

        mean = 1./2 * first_value + 1./2 * second_value
        x = mean

        for i in range(len(self.image)):
             for j in range(len(self.image[0])):

                value = 0
                if self.image[i, j][0] > x:
                    value = 255

                self.image[i, j] = value

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

    img.covert_to_grayscale_avg()
    img2.covert_to_grayscale_avg()
    
    #img.show_image('image')

    img.wrap_boundary_copy()

    img2.wrap_boundary_copy()

    img.thresholding_modified()

    #img.show_image('thresholding')
    img.write_image('thresholding')

    imgThresh1 = Image()
    imgThresh2 = Image()


if __name__=="__main__":
    main()

cv2.waitKey(0)
cv2.destroyAllWindows()            
