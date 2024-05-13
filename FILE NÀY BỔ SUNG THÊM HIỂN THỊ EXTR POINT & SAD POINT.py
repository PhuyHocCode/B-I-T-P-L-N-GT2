import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gradient(f, x, y, h=0.01):
    # Compute the gradient of the function at point (x, y)
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

def hessian(f, x, y, h=0.01):
    # Compute the Hessian matrix of the function at point (x, y)
    df_dxdx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    df_dydy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    df_dxdy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return np.array([[df_dxdx, df_dxdy], [df_dxdy, df_dydy]])

def extrema(f, x_range, y_range, threshold=1e-5):
    # Find extrema of the function within the given ranges
    extrema = []
    for x in np.arange(x_range[0], x_range[1], 0.1):
        for y in np.arange(y_range[0], y_range[1], 0.1):
            grad = gradient(f, x, y)
            hess = hessian(f, x, y)
            eigvals, _ = np.linalg.eig(hess)
            if all(eigval > 0 for eigval in eigvals):
                if np.linalg.norm(grad) < threshold:
                    extrema.append((x, y, "Minimum"))
            elif all(eigval < 0 for eigval in eigvals):
                if np.linalg.norm(grad) < threshold:
                    extrema.append((x, y, "Maximum"))
            else:
                extrema.append((x, y, "Saddle"))
    return extrema

def plot_surface(f, x_range, y_range):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(x_range[0], x_range[1], 0.1)
    Y = np.arange(y_range[0], y_range[1], 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    
    # Find extrema
    extrema_points = extrema(f, x_range, y_range)
    for point in extrema_points:
        x, y, type_extrema = point
        if type_extrema == "Minimum":
            ax.scatter([x], [y], [f(x, y)], color='g', s=50, label=f'Minimum ({x:.2f}, {y:.2f}, {f(x, y):.2f})')
        elif type_extrema == "Maximum":
            ax.scatter([x], [y], [f(x, y)], color='r', s=50, label=f'Maximum ({x:.2f}, {y:.2f}, {f(x, y):.2f})')
        else:
            ax.scatter([x], [y], [f(x, y)], color='b', s=50, label=f'Saddle ({x:.2f}, {y:.2f}, {f(x, y):.2f})')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Define the function equation
    equation = input("Enter the function equation in terms of x and y: ")
    custom_function = lambda x, y: eval(equation)
    
    # Define the range of x and y values to search for extrema
    x_range = (-5, 5)
    y_range = (-5, 5)
    
    # Plot the surface with extrema
    plot_surface(custom_function, x_range, y_range)
