import numpy as np
import struct
import matplotlib.pyplot as plt
import math

def load_mnist(image_file, label_file):
    """
    Load MNIST data from the binary files
    """
    # Load labels
    with open(label_file, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    
    # Load images
    with open(image_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    
    return images, labels

# Define paths to the dataset files
train_images_file = 'data/train-images.idx3-ubyte__'
train_labels_file = 'data/train-labels.idx1-ubyte__'
test_images_file = 'data/t10k-images.idx3-ubyte__'
test_labels_file = 'data/t10k-labels.idx1-ubyte__'

# Load the datasets
train_images, train_labels = load_mnist(train_images_file, train_labels_file)
test_images, test_labels = load_mnist(test_images_file, test_labels_file)

print(f"Training set: {train_images.shape[0]} images")
print(f"Test set: {test_images.shape[0]} images")


class NaiveBayes:
    def __init__(self, mode=0):
        """
        Initialize Naive Bayes classifier
        mode: 0 for discrete, 1 for continuous
        """
        self.mode = mode
        self.num_classes = 10  # digits 0-9
        self.image_size = 28 * 28
        self.class_priors = None
        
        # For discrete mode
        self.bin_count = 32
        self.pixel_bin_probs = None
        
        # For continuous mode
        self.pixel_means = None
        self.pixel_variances = None
    
    def bin_pixel_value(self, pixel_value):
        """Convert pixel value (0-255) to bin index (0-31)"""
        return pixel_value // 8  # 256 / 32 = 8
    
    def train_discrete(self, images, labels):
        """Train the model using discrete binning of pixel values"""
        num_samples = images.shape[0]
        
        # Calculate class priors
        self.class_priors = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            self.class_priors[i] = np.sum(labels == i) / num_samples
        
        # Initialize bin counters for each pixel and class
        # shape: (class, pixel, bin)
        self.pixel_bin_probs = np.zeros((self.num_classes, self.image_size, self.bin_count))
        
        # Count occurrences in each bin
        for i in range(num_samples):
            digit_class = labels[i]
            for pixel_idx in range(self.image_size):
                pixel_value = images[i, pixel_idx]
                bin_idx = self.bin_pixel_value(pixel_value)
                self.pixel_bin_probs[digit_class, pixel_idx, bin_idx] += 1
        
        # Add pseudocount (Laplace smoothing) and normalize to get probabilities
        pseudocount = 1
        for digit_class in range(self.num_classes):
            class_samples = np.sum(labels == digit_class)
            for pixel_idx in range(self.image_size):
                for bin_idx in range(self.bin_count):
                    # Add pseudocount to avoid zero probabilities
                    self.pixel_bin_probs[digit_class, pixel_idx, bin_idx] += pseudocount
                    # Normalize
                    self.pixel_bin_probs[digit_class, pixel_idx, bin_idx] /= (class_samples + self.bin_count * pseudocount)

    def train_continuous(self, images, labels):
        """Train the model using continuous Gaussian distribution"""
        num_samples = images.shape[0]
        
        # Calculate class priors
        self.class_priors = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            self.class_priors[i] = np.sum(labels == i) / num_samples
        
        # Initialize arrays for means and variances
        self.pixel_means = np.zeros((self.num_classes, self.image_size))
        self.pixel_variances = np.zeros((self.num_classes, self.image_size))
        
        # Calculate means
        for digit_class in range(self.num_classes):
            class_samples = images[labels == digit_class]
            self.pixel_means[digit_class] = np.mean(class_samples, axis=0)
            
            # Calculate variances
            self.pixel_variances[digit_class] = np.var(class_samples, axis=0) + 1e-10  # Add small epsilon to avoid division by zero
    
    def train(self, images, labels):
        """Train the Naive Bayes classifier"""
        if self.mode == 0:  # Discrete
            self.train_discrete(images, labels)
        else:  # Continuous
            self.train_continuous(images, labels)


    def predict_discrete(self, image):
        """Predict class for an image using discrete binning"""
        log_posteriors = np.zeros(self.num_classes)
        
        # Start with log prior probabilities
        for digit_class in range(self.num_classes):
            log_posteriors[digit_class] = np.log(self.class_priors[digit_class])
            
            # Add log likelihoods for each pixel
            for pixel_idx in range(self.image_size):
                pixel_value = image[pixel_idx]
                bin_idx = self.bin_pixel_value(pixel_value)
                log_posteriors[digit_class] += np.log(self.pixel_bin_probs[digit_class, pixel_idx, bin_idx])
        
        return log_posteriors
    
    def predict_continuous(self, image):
        """Predict class for an image using continuous features"""
        log_posteriors = np.zeros(self.num_classes)
        
        # Start with log prior probabilities
        for digit_class in range(self.num_classes):
            log_posteriors[digit_class] = np.log(self.class_priors[digit_class])
            
            # Add log likelihoods for each pixel using Gaussian PDF
            for pixel_idx in range(self.image_size):
                pixel_value = image[pixel_idx]
                mean = self.pixel_means[digit_class, pixel_idx]
                var = self.pixel_variances[digit_class, pixel_idx]
                
                # Log of Gaussian PDF: -0.5 * log(2π * var) - 0.5 * (x - mean)^2 / var
                log_likelihood = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((pixel_value - mean) ** 2) / (2 * var) 
                log_posteriors[digit_class] += log_likelihood
        
        return log_posteriors
    
    def predict(self, image):
        """Predict class for an image based on current mode"""
        if self.mode == 0:  # Discrete
            log_posteriors = self.predict_discrete(image)
        else:  # Continuous
            log_posteriors = self.predict_continuous(image)
        
        # Return both the predicted class and the log posteriors
        return np.argmax(log_posteriors), log_posteriors

    def visualize_imagination(self):
        """Visualize the imagination of numbers in the classifier"""
        imagination = np.zeros((self.num_classes, 28, 28), dtype=np.uint8)
        
        if self.mode == 0:  # Discrete
            # For discrete mode, we need to determine the most likely bin for each pixel
            for digit_class in range(self.num_classes):
                for pixel_idx in range(self.image_size):
                    # Find the bin with the highest probability for this pixel and class
                    max_prob_bin = np.argmax(self.pixel_bin_probs[digit_class, pixel_idx])
                    # Mark as 1 if the bin corresponds to value >= 128, otherwise 0
                    if max_prob_bin >= 16:  # 16 * 8 = 128
                        imagination[digit_class, pixel_idx // 28, pixel_idx % 28] = 1
        
        else:  # Continuous
            # For continuous mode, we use the mean values
            for digit_class in range(self.num_classes):
                for pixel_idx in range(self.image_size):
                    mean_value = self.pixel_means[digit_class, pixel_idx]
                    if mean_value >= 128:
                        imagination[digit_class, pixel_idx // 28, pixel_idx % 28] = 1
        
        return imagination
    
def main():
    # Parse command line arguments
    import sys
    mode = 0  # Default to discrete mode
    if len(sys.argv) > 1:
        mode = int(sys.argv[1])
    
    # Load the datasets
    train_images_file = 'data/train-images.idx3-ubyte__'
    train_labels_file = 'data/train-labels.idx1-ubyte__'
    test_images_file = 'data/t10k-images.idx3-ubyte__'
    test_labels_file = 'data/t10k-labels.idx1-ubyte__'
    
    train_images, train_labels = load_mnist(train_images_file, train_labels_file)
    test_images, test_labels = load_mnist(test_images_file, test_labels_file)
    
    print(f"Training Naive Bayes classifier in {'discrete' if mode == 0 else 'continuous'} mode")
    
    # Initialize and train the classifier
    classifier = NaiveBayes(mode=mode)
    classifier.train(train_images, train_labels)
    
    # Test the classifier
    num_correct = 0
    num_test = test_images.shape[0]
    
    for i in range(num_test):
        predicted_class, log_posteriors = classifier.predict(test_images[i])
        true_class = test_labels[i]
        
        # Print posteriors for each test image
        print(f"Test image {i+1}:")
        log_posteriors /= np.sum(log_posteriors)
        
        for j in range(10):
            print(f"Posterior for digit {j}: {log_posteriors[j]:.17f}")
        
        print(f"Prediction: {predicted_class}, True label: {true_class}")
        print()
        
        if predicted_class == true_class:
            num_correct += 1
    
    error_rate = 1 - (num_correct / num_test)
    print(f"Error rate: {error_rate:.6f}")
    
    # Visualize the imagination of numbers
    imagination = classifier.visualize_imagination()
    
    for digit_class in range(10):
        print(f"\nImagination of digit {digit_class}:")
        for row in imagination[digit_class]:
            print(''.join(['1' if pixel else '0' for pixel in row]))

if __name__ == "__main__":
    main() 

# discrete
# python naive_bayes.py 0

# condinuous
# python naive_bayes.py 1