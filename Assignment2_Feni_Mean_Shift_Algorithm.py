import cv2
import numpy as np
import matplotlib.pyplot as plt

class Mean_Shift():

    def __init__(self, img, windowRadius = 25):

        # Initializes variables for mean shift algorithm.
        self.img = img
        self.imShape = img.shape
        self.segmentedImg = np.zeros(img.shape)
        self.windowRadius = windowRadius
        self.histogram = np.zeros(256, dtype="uint")
        self.intensityArray = np.arange(256)
        self.seed = []

    def generateHistogram(self):
        """
        Generates histogram of stored image by counting intensities of each pixel.
        """
        for x in range(self.imShape[0]):
            for y in range(self.imShape[1]):
                self.histogram[self.img[x, y]] += 1

    def calcMeanValue(self, seedIndex):
        """
        Calculates the mean intensity of a given seed point and the window radius around it,
        returning the mean seed point.
        """
        lowerBound = self.seed[seedIndex] - self.windowRadius
        upperBound = self.seed[seedIndex] + self.windowRadius

        # Verifies valid index entries to histogram.
        if(lowerBound < 0): 
            lowerBound = 0
        if(upperBound > 256 ): 
            upperBound = 256

        # Verifies that the histogram is nonempty for mean shift calculation to proceed. If empty, returns a randomized seed for future iterations.
        if(np.sum(self.histogram[lowerBound:upperBound]) == 0):
            print("Empty Histogram. Re-randomizing seed.")
            return np.random.randint(low=0, high=255)

        # Performs mean shift calculation and casts to integer to work as array index.
        weightedIntensity = np.sum(np.multiply(self.histogram[lowerBound:upperBound], self.intensityArray[lowerBound:upperBound]))
        meanIntensity = int(weightedIntensity/np.sum(self.histogram[lowerBound:upperBound]))
    
        return meanIntensity

    def convergeMeanShift(self, seedIndex = -1, iterationLimit = 1000):
        """
        Sets stop condition and repeats mean shift calculation until stop condition is met. Stop condition is either iteration limit or 5 consecutive repeated outputs.
        """
        previousSeed = -1
        stopIterator = 0

        # If no seed index is provided, the algorithm creates a new random seed and adds it to the list.
        if(seedIndex == -1):
            self.seed.append(np.random.randint(low=0, high=256))
            seedIndex = len(self.seed) - 1
    
        # Iterates calcMeanValue until timeout or convergence.
        for i in range(iterationLimit):
            self.seed[seedIndex] = self.calcMeanValue(seedIndex)

            # If seed is same as previous iteration, counts up to 5 then breaks loop.
            if(previousSeed == self.seed[seedIndex]):
                stopIterator += 1
                if(stopIterator >= 5):
                    break

            previousSeed = self.seed[seedIndex]

    def filterRedundantSeeds(self):
        """
        Purges duplicate converged seeds from array and sorts the remaining unique seeds.
        """
        self.seed = np.unique(self.seed)

    def segmentImage(self):
        """
        For each seed in the list, checks image for pixels within the window and assigns them to the seed. Unassigned/unsegmented pixels remain as intensity 0.
        """
        for mode, intensity in enumerate(self.seed):
            lowerBound = self.seed[mode] - self.windowRadius
            upperBound = self.seed[mode] + self.windowRadius

            # Verifies valid index entries to histogram.
            if(lowerBound < 0): 
                lowerBound = 0
            if(upperBound > 256 ): 
                upperBound = 256
            
            # Assigns each pixel in the array to its respective mode. Unassigned pixels remain as 0.
            self.segmentedImg[(lowerBound <= self.img) & (self.img <= upperBound)] = intensity

if __name__ == "__main__":
    np.random.seed = 42 # Set seed for consistent randomness.
    image = cv2.imread("sheep.jfif", cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread("apple.png", cv2.IMREAD_GRAYSCALE)

    # Initialize Mean Shift class.
    meanShift = Mean_Shift(image, windowRadius=15)

    # Generate histogram based on image stored within class.
    meanShift.generateHistogram()

    # Generate a seed for each level of intensity. How many seeds generated is up to user desire. Iterate converging the mean shift algorithm through each seed.
    meanShift.seed = list(range(256))
    for iteration in range(256):
        meanShift.convergeMeanShift(seedIndex = iteration)

    # Remove redundant seeds and print remaining seeds to console.
    meanShift.filterRedundantSeeds()
    print(meanShift.seed)

    # Segment the image with the currently stored list of seeds.
    meanShift.segmentImage()

    # Display results.
    plt.imshow(meanShift.segmentedImg, cmap="gray")
    plt.show()
    plt.imshow(meanShift.img, cmap="gray")
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(meanShift.histogram)
    plt.show()