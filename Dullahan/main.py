# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from skimage import io
import numpy as np

class BinaryThresh:
    def __init__(self, input_path, coloc_files, output_path=""):
        self.input_path = input_path
        self.save_path = output_path
        self.coloc_files = coloc_files

    def imageextracter(self, file_name):
        channel = io.imread(self.input_path + file_name)
        channel_shape = channel.shape
        channel_len = len(channel_shape)
        if(channel_len < 3):
            print("These are not voxels")
            return channel, -1
        else:
            if(channel_shape[-1] == 3):
                #RGB voxels
                return channel, 1
            else:
                #Not RGB voxels
                return channel, 0

    def thresholding(self, image, threshold):
        thresholded_Image = np.copy(image)
        thresholded_Image[thresholded_Image < threshold] = 0
        thresholded_Image[thresholded_Image > threshold] = 255
        return thresholded_Image

    def saveBinarized(self, image, fName):
        io.imsave(self.save_path + fName, image)

    def binarizeImages(self):
        for f in self.coloc_files:
            image, dims = self.imageextracter(f[0])
            if(dims == -1):
                print("Incorrect dimensions")
            else:
                thresholded_image = self.thresholding(image=image, threshold=f[1])
                self.saveBinarized(image=thresholded_image, fName=f[0])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    list_of_files = [["\\N1CCCP_10.tif", 0]]
    save_location = "C:\\RESEARCH\\Mitophagy_data\\N1"

    tester = BinaryThresh(input_path="C:\\RESEARCH\\Mitophagy_data\\N1\\RACC Results", coloc_files=list_of_files, output_path=save_location)
    tester.binarizeImages()