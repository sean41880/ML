import numpy as np

def generate_gaussian(mean, variance):
    """
    Generate a random sample from a Gaussian (normal) distribution
    with given mean and variance using the Box-Muller transform.
    
    Args:
        mean (float): True mean of the distribution
        variance (float): True variance of the distribution
        
    Returns:
        float: A random sample from N(mean, variance)
    """
    # Box-Muller transform:
    # 1. Generate two independent uniform random variables
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    
    # 2. Transform to standard normal distribution (mean=0, variance=1)
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    
    # 3. Scale and shift to get the desired mean and variance
    x = mean + np.sqrt(variance) * z
    
    return x

def main():
    """
    Main function to run the sequential estimator
    """
    try:
        # Get the true parameters from user
        true_mean = float(input("Enter the true mean (m): "))
        true_variance = float(input("Enter the true variance (s): "))
        
        if true_variance <= 0:
            print("Error: Variance must be positive")
            return
        
        # Set convergence criteria
        mean_threshold = 0.01  # Convergence threshold for mean
        variance_threshold = 0.01  # Convergence threshold for variance
        
        # Initialize estimates
        count = 0
        mean_estimate = 0
        variance_estimate = 0
        
        # Previous estimates for checking convergence
        prev_mean_estimate = float('inf')
        prev_variance_estimate = float('inf')
        
        # Continue until both estimates converge
        while (abs(mean_estimate - prev_mean_estimate) > mean_threshold or
               abs(variance_estimate - prev_variance_estimate) > variance_threshold or
               count < 2):  # Need at least 2 points for variance
            
            # Get a new data point from the Gaussian distribution
            x = generate_gaussian(true_mean, true_variance)
            count += 1
            
            # Save previous estimates
            prev_mean_estimate = mean_estimate
            prev_variance_estimate = variance_estimate
            
            # Update mean estimate (online algorithm)
            delta = x - mean_estimate
            mean_estimate = mean_estimate + delta / count
            
            # Update variance estimate (online algorithm)
            if count > 1:
                # Welford's algorithm for online variance computation
                delta2 = x - mean_estimate
                variance_estimate = variance_estimate + (delta * delta2 - variance_estimate) / count
            
            # Print current iteration
            print("Add data point: ", x)
            print("Mean = ", mean_estimate, "Variance = ", variance_estimate)
            
        print("\nEstimation converged!")
        print(f"Final mean estimate: {mean_estimate:.6f} (true mean: {true_mean})")
        print(f"Final variance estimate: {variance_estimate:.6f} (true variance: {true_variance})")
        
    except ValueError:
        print("Error: Please enter valid numbers for mean and variance")

if __name__ == "__main__":
    main()