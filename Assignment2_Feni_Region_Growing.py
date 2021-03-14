import sys
import cv2
import numpy as np
import os
import time 
import skimage

SAVE = 's'
ESC_KEY = 27

class Region_Growing():

    def __init__(self, img, threshold, conn=4):
        
        # Initialising all the variables required for region growing
        self.img = img
        self.segmentation = np.zeros(img.shape)
        self.threshold = threshold
        self.seeds = []
        
        # neigbourhood methods
        if conn == 4:
            self.orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        elif conn == 8:
            self.orientations = [(1, 0), (1, 1), (0, 1), (-1, 1),
                                 (-1, 0), (-1, -1), (0, -1), (1, -1)]  # 8 connectivity
        else:
            raise ValueError("(%s) Connectivity type not known (4 or 8 available)!" % (
                sys._getframe().f_code.co_name))

    def set_seeds(self, name="Region Growing"):
        """
        Set seed points using left mouse button.
        """
        self.seeds = []
        self.img = np.array(self.img, dtype=np.uint8)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(name, self.__on_left_click)

        cv2.imshow(name, self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def segment(self):
        """
        Segment the image with the provided user seeds using region growing
        """
        for seed in self.seeds:
            curr_pixel = [seed[1], seed[0]]
            if self.segmentation[curr_pixel[0], curr_pixel[1]] == 255:
                continue  # pixel already explored
            contour = []
            seg_size = 1
            mean_seg_value = (self.img[curr_pixel[0], curr_pixel[1]])
            dist = 0
            while(dist < self.threshold):
                # Include current pixel in segmentation
                self.segmentation[curr_pixel[0], curr_pixel[1]] = 255
                # Explore neighbours of current pixel
                contour = self.__explore_neighbours(contour, curr_pixel)
                # Get the nearest neighbour
                nearest_neighbour_idx, dist = self.__get_nearest_neighbour(
                    contour, mean_seg_value)
                # If no more neighbours to grow, move to the next seed
                if nearest_neighbour_idx == -1:
                    break
                # Update Current pixel to the nearest neighbour and increment size
                curr_pixel = contour[nearest_neighbour_idx]
                seg_size += 1
                # Update Mean pixel value for segmentation
                mean_seg_value = (
                    mean_seg_value*seg_size + float(self.img[curr_pixel[0], curr_pixel[1]]))/(seg_size+1)
                # Delete from contour once the nearest neighbour as chosen as the current node for expansion
                del contour[nearest_neighbour_idx]
        return self.segmentation

    def display_and_resegment(self, name="Region Growing"):
        """
        Display the segmented image and ask for user seeds for further segmentation.
        Press "s" to save the segmented image
        Press "Esc" to cancel further segmentation and exit the program
        """
        self.seeds = []
        # Display original image where segmentation was not done
        result = np.maximum(self.img, self.segmentation)
        result = np.array(result, dtype=np.uint8)
        # Display the result
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        # Activate mouse click on the image window
        cv2.setMouseCallback(name, self.__on_left_click)
        cv2.imshow(name, result)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Press "s" to save the segmented result
        if chr(key) == SAVE:
            cv2.imwrite(name+'.png', result)
        # Press "Esc" to if no more seeds are required and end the program
        if key == ESC_KEY:
            return
        # Resegment the image using new seeds
        self.segment()
        self.display_and_resegment(name=name)

    def __explore_neighbours(self, contour, current_pixel):
        """
        Function is used to detect the nearby pixels and form the cluster
        """
        for orientation in self.orientations:
            neighbour = self.__get_neighbouring_pixel(
                current_pixel, orientation, self.img.shape)
            if neighbour is None:
                continue
            if self.segmentation[neighbour[0], neighbour[1]] == 0:
                contour.append(neighbour)
                self.segmentation[neighbour[0], neighbour[1]] = 150
        return contour

    def __get_neighbouring_pixel(self, current_pixel, orient, img_shape):
        neighbour = (current_pixel[0]+orient[0], current_pixel[1]+orient[1])
        if self.is_pixel_inside_image(pixel=neighbour, img_shape=img_shape):
            return neighbour
        else:
            return None

    def __get_nearest_neighbour(self, contour, mean_seg_value):
        """
        Calculates the nearby pixels which are under the threshold
        """
    
        dist_list = [abs(int(self.img[pixel[0], pixel[1]]) - int(mean_seg_value))for pixel in contour]
        if len(dist_list) == 0:
            return -1, 1000
        min_dist = min(dist_list)
        index = dist_list.index(min_dist)
        return index, min_dist

    def is_pixel_inside_image(self, pixel, img_shape):
        """
        Checking if the pixel is inside the frame of image
        """
        return 0 <= pixel[0] < img_shape[0] and 0 <= pixel[1] < img_shape[1]

    def __on_left_click(self, event, x, y, flags, param):
        """
        Function automatically called by opencv2 when having mouse events on a
        displayed frame. In here, we are searching for a left mouse click
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.seeds.append((x, y))
            print(f"Seed:{(x,y)}")


if __name__ == "__main__":
    
    try:
        start = time.perf_counter()
        # Reading the Image data
        image_path="apple.png"
        image_name, ext = os.path.splitext(image_path)
        image=cv2.imread(image_path, 0)
        
        row,col= image.shape
        mean = 0  # Mean of Gaussian Distribution
        var = 400  # Variance of Gaussian Distribution
        sigma = var**0.5  # Standard Deviation
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        
        # Noisy Image
        noisy_g = image + gauss
        noisy_g_clip = np.clip(noisy_g, 0, 255)
        
        noisy_s_p=skimage.util.random_noise(image, mode='s&p', seed=None, clip=True)*255
        image_data = cv2.resize(image, (300,300), interpolation = cv2.INTER_AREA)
    
        # Resizing the image
        if image_data.shape[0] > 1000:
        		image_data = cv2.resize(image_data, (0, 0), fx=0.25, fy=0.25)
        if image_data.shape[0] > 500:
            image_data = cv2.resize(image_data, (0,0), fx=0.5, fy=0.5)
        
        # Defining the parameters for region growing algorithm
        neighbours = 4
        segmentation_name = image_name+ " segmentation"
        threshold = 50
        
        
        # Calling the Region Growing Class which was defined earlier
        region_growing = Region_Growing(image_data, threshold=threshold, conn=neighbours)
        # Set Seeds
        region_growing.set_seeds()
        # Segmentation
        region_growing.segment()
        # Display Segmentation
        print(f"Total time: {int(time.perf_counter() - start)} s ")
        region_growing.display_and_resegment(name=segmentation_name)
        
        #print(f"Total time: {time.perf_counter() - start} ")
    except:
        print("An error occured as you pressed the wrong key!")
