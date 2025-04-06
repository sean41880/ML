import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

def generate_gaussian(mean, variance):
    """
    Generate a random sample from a Gaussian distribution using Box-Muller transform.
    """
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mean + np.sqrt(variance) * z

def generate_data_point(n, a, w):
    """
    Generate a single data point from a polynomial basis linear model.
    """
    # Generate x uniformly from (-1, 1)
    x = np.random.uniform(-1.0, 1.0)
    
    # Compute polynomial basis values [1, x, x², ..., xⁿ⁻¹]
    phi = np.array([x**i for i in range(n)])
    
    # Compute true function value
    f_x = np.dot(w, phi)
    
    # Add Gaussian noise
    noise = generate_gaussian(0, a**2)
    y = f_x + noise
    
    return x, y, phi

def compute_predictive_distribution(x, posterior_mean, posterior_cov, a):
    """
    Compute predictive distribution parameters for a given input x.
    """
    # Compute phi(x) for the input
    phi = np.array([x**i for i in range(len(posterior_mean))])
    
    # Predictive mean
    pred_mean = np.dot(posterior_mean, phi)
    
    # Predictive variance
    pred_var = a**2 + np.dot(np.dot(phi, posterior_cov), phi)
    
    return pred_mean, pred_var

def visualize_results(ax, n, a, w, data_points, posterior_mean=None, posterior_cov=None, title=None, show_variance=True):
    """
    Visualize the regression results on a given subplot.
    
    Args:
        ax: The matplotlib axis to plot on
        n: Number of basis functions
        a: Noise standard deviation
        w: Ground truth weights
        data_points: List of (x,y) data points
        posterior_mean: Posterior mean weights (None for ground truth only)
        posterior_cov: Posterior covariance matrix (None for ground truth only)
        title: Plot title
        show_variance: Whether to show variance bands
    """
    # Generate x values for plotting
    x_plot = np.linspace(-2, 2, 100)
    
    # True function values
    true_y = [np.dot(w, [x**i for i in range(n)]) for x in x_plot]
    
    # Plot the true function
    ax.plot(x_plot, true_y, 'g-', label='Ground Truth')
    
    # If we have data points, plot them
    if data_points:
        x_data, y_data = zip(*data_points)
        ax.scatter(x_data, y_data, c='blue', s=30, alpha=0.5, label='Data Points')
    
    # If we have posterior parameters, plot predictive distribution
    if posterior_mean is not None and posterior_cov is not None:
        # Predictive distribution
        pred_means = []
        pred_std = []
        
        for x in x_plot:
            phi = np.array([x**i for i in range(n)])
            mean = np.dot(posterior_mean, phi)
            var = a**2 + np.dot(np.dot(phi, posterior_cov), phi)
            std = np.sqrt(var)
            
            pred_means.append(mean)
            pred_std.append(std)
        
        # Plot the predictive mean
        ax.plot(x_plot, pred_means, 'k-', label='Predictive Mean')
        
        # Plot the predictive variance bands
        if show_variance:
            ax.plot(x_plot, np.array(pred_means) + np.array(pred_std), 'r-', label='Mean ± 1 Std Dev')
            ax.plot(x_plot, np.array(pred_means) - np.array(pred_std), 'r-')
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    
    # Only add the legend to the first subplot to avoid redundancy
    if posterior_mean is None:
        ax.legend(loc='best')


def main():
    # Get user inputs
    b = float(input("Enter the precision parameter (b) for prior: "))
    n = int(input("Enter the number of basis functions (n): "))
    a = float(input("Enter the noise standard deviation (a): "))
    
    print(f"Enter the {n} weights for the ground truth function:")
    w = [float(input(f"w_{i}: ")) for i in range(n)]
    
    # Initialize parameters 
    posterior_mean = np.zeros(n)
    # b = precision, covariance = 1/b * I
    posterior_cov = (1.0/b) * np.eye(n) 
    
    # Initialize previous values for the first iteration
    prev_posterior_mean = np.zeros(n)
    prev_posterior_cov = posterior_cov.copy()
    
    # Store data points
    data_points = []
    
    # Set convergence criteria
    max_iter = 1000
    conv_threshold = 1e-5
    
    # For visualization at specified points
    points_10_mean = None
    points_10_cov = None
    points_50_mean = None
    points_50_cov = None
    
    # Main loop
    for iter_count in range(1, max_iter + 1):
        # Generate new data point
        x, y, phi = generate_data_point(n, a, w)
        data_points.append((x, y))
        
        # Update posterior
        phi_col = phi.reshape(-1, 1)
        phi_row = phi.reshape(1, -1)
        
        # Precision matrix update
        precision_matrix = inv(posterior_cov) + (1.0/a**2) * np.dot(phi_col, phi_row)
        # Posterior covariance
        posterior_cov = inv(precision_matrix)
        
        # Posterior mean
        # For first iteration, only use the data term since prior mean is 0
        if iter_count == 1:
            posterior_mean = posterior_cov @ ((1.0/a**2) * y * phi)
        else:
            posterior_mean = posterior_cov @ (inv(prev_posterior_cov) @ prev_posterior_mean + (1.0/a**2) * y * phi)
        
        # Compute predictive distribution at x
        pred_mean, pred_var = compute_predictive_distribution(x, posterior_mean, posterior_cov, a)
        
        # Print current state
        print(f"\nAdd data point ({x:.5f}, {y:.5f}):")
        print("\nPosterior mean:")
        for i in range(n):
            print(f"{posterior_mean[i]:.10f}")
        
        print("\nPosterior variance:")
        for i in range(n):
            for j in range(n):
                print(f"{posterior_cov[i, j]:.10f}", end=", " if j < n-1 else "\n")
        
        print(f"\nPredictive distribution ~ N({pred_mean:.5f}, {pred_var:.5f})")
        
        # Save state at specified points
        if iter_count == 10:
            points_10_mean = posterior_mean.copy()
            points_10_cov = posterior_cov.copy()
        elif iter_count == 50:
            points_50_mean = posterior_mean.copy()
            points_50_cov = posterior_cov.copy()
        
        # Check convergence
        mean_diff = np.linalg.norm(posterior_mean - prev_posterior_mean)
        if mean_diff < conv_threshold and iter_count > 10:
            print(f"\nConverged after {iter_count} iterations!")
            break
        
        # Update for next iteration
        prev_posterior_mean = posterior_mean.copy()
        prev_posterior_cov = posterior_cov.copy()
    
    # Visualizations
    print("\nGenerating combined visualization...")
    
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top-left: Ground truth
    visualize_results(axs[0, 0], n, a, w, [], title="Ground Truth Function", 
                     posterior_mean=None, posterior_cov=None, show_variance=False)
    
    # Top-right: Final prediction 
    visualize_results(axs[0, 1], n, a, w, data_points, posterior_mean, posterior_cov,
                     title=f"Final Result (after {iter_count} points)")
    
    # Bottom-left: After 10 points
    if points_10_mean is not None:
        visualize_results(axs[1, 0], n, a, w, data_points[:10], points_10_mean, points_10_cov,
                         title="After 10 Data Points")
    else:
        axs[1, 0].text(0.5, 0.5, "Not enough data points", 
                      horizontalalignment='center', verticalalignment='center')
    
    # Bottom-right: After 50 points
    if points_50_mean is not None and iter_count >= 50:
        visualize_results(axs[1, 1], n, a, w, data_points[:50], points_50_mean, points_50_cov,
                         title="After 50 Data Points")
    else:
        axs[1, 1].text(0.5, 0.5, "Not enough data points", 
                      horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig("bayesian_combined.png")
    plt.show()

if __name__ == "__main__":
    main()