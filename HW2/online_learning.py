import numpy as np
import sys
import math

def binomial_likelihood(n, k, p):
    """
    Calculate the binomial probability mass function P(X=k|n,p)
    """
    # Calculate binomial coefficient (n choose k)
    coef = math.comb(n, k)
    # Calculate probability
    return coef * (p**k) * ((1-p)**(n-k))

def beta_binomial_update(data_line, prior_a, prior_b, case_num):
    """
    Process a single line of binary data and update Beta parameters
    
    Args:
        data_line: String of 0s and 1s
        prior_a: Alpha parameter of the Beta prior
        prior_b: Beta parameter of the Beta prior
        case_num: Case number for output
        
    Returns:
        Updated prior parameters
    """
    # Clean the data (remove any non-binary characters)
    binary_data = [int(char) for char in data_line if char in '01']
    
    # Count successes (1s) and total trials
    n = len(binary_data)
    k = sum(binary_data)
    
    # Calculate MLE estimate
    p_mle = k/n if n > 0 else 0
    
    # Calculate likelihood
    likelihood = binomial_likelihood(n, k, p_mle)
    
    # Calculate posterior parameters
    posterior_a = prior_a + k
    posterior_b = prior_b + (n - k)
    
    # Print in the required format
    print(f"case {case_num}: {data_line}")
    print(f"Likelihood: {likelihood}")
    print(f"Beta prior: a = {prior_a} b = {prior_b}")
    print(f"Beta posterior: a = {posterior_a} b = {posterior_b} \n")
    
    return posterior_a, posterior_b

def main():
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python online_learning.py <prior_a> <prior_b>")
        print("Example: python online_learning.py 0 0")
        sys.exit(1)
    
    # Get initial prior parameters from command line
    try:
        prior_a = float(sys.argv[1])
        prior_b = float(sys.argv[2])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # File path
    file_path = 'data/online_data.txt'
    
    # Process each line of the file
    try:
        with open(file_path, 'r') as file:
            # Skip any possible header lines starting with '//'
            lines = [line.strip() for line in file if not line.strip().startswith('//')]
            
            # Current prior parameters
            current_a = prior_a
            current_b = prior_b
            
            for i, line in enumerate(lines):
                if not line:  # Skip empty lines
                    continue
                    
                # Process the line and update parameters
                current_a, current_b = beta_binomial_update(line, current_a, current_b, i+1)
                
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)

if __name__ == "__main__":
    main()