import numpy as np

def generate_data_point(n, a, w):
    """
    Generate a single data point (x,y) from a polynomial basis linear model.
    """
    # Generate x uniformly from (-1, 1)
    x = np.random.uniform(-1.0, 1.0)
    
    # Compute polynomial basis values [1, x, x², ..., xⁿ⁻¹]
    phi = np.array([x**i for i in range(n)])
    
    # Compute true function value: w₀ + w₁x + w₂x² + ... + wₙ₋₁xⁿ⁻¹
    f_x = np.dot(w, phi)
    
    # Add Gaussian noise with mean 0, variance a²
    noise = generate_gaussian(0, a**2)
    
    # Final y value
    y = f_x + noise
    
    return x, y

def generate_gaussian(mean, variance):
    """
    Generate a random sample from a Gaussian (normal) distribution
    using the Box-Muller transform.
    """
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    
    # Transform to standard normal distribution
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    
    # Scale and shift to get desired mean and variance
    return mean + np.sqrt(variance) * z

def main():
    """Main function to handle user input and generate a data point"""
    try:
        # Get user input
        n = int(input("Enter the number of basis functions (n): "))
        a = float(input("Enter the noise standard deviation (a): "))
        
        print(f"Enter {n} weights for the polynomial terms:")
        w = []
        for i in range(n):
            w_i = float(input(f"w_{i}: "))
            w.append(w_i)
        
        # Generate and display a data point
        x, y = generate_data_point(n, a, w)
        print(f"\nGenerated data point: (x, y) = ({x:.6f}, {y:.6f})")
            
    except ValueError:
        print("Error: Please enter valid numbers for the parameters")

if __name__ == "__main__":
    main()