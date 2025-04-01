import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_gaussian(mean, variance):
    """
    Generate a random sample from a Gaussian (normal) distribution
    with given mean and variance using the Box-Muller transform.
    """
    # Box-Muller transform:
    # 1. Generate two independent uniform random variables
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)
    
    # 2. Transform to standard normal distribution (mean=0, variance=1)
    X = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V)
    
    # 3. Scale and shift to get the desired mean and variance
    x = mean + np.sqrt(variance) * X
    
    return x

def main():
    """Main function to handle user input and generate samples"""
    try:
        # Get user input
        mean = float(sys.argv[1])
        variance = float(sys.argv[2])
        
        if variance <= 0:
            print("Error: Variance must be positive")
            return
        
        # Generate and display a sample
        sample = generate_gaussian(mean, variance)
        print(f"\nGenerated data point from N({mean}, {variance}): {sample:.6f}")
            
    except ValueError:
        print("Error: Please enter valid numbers for mean and variance")

if __name__ == "__main__":
    main()