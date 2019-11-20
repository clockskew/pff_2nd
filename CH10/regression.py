import numpy as np
from pylab import plt, mpl
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return np.sin(x) + 0.5 * x


def create_plot(x, y, styles, labels, axlabels):
    plt.figure(figsize=(10, 6))
    for i in range(len(x)):
        plt.plot(x[i], y[i], styles[i], label=labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
    plt.legend(loc=0)
    plt.show()


def fm(p):
    x, y = p
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2


if __name__ == '__main__':
    x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    res = np.polyfit(x, f(x), deg=1, full=True)
    ry = np.polyval(res[0], x)
    matrix = np.zeros((4 + 1, len(x)))
    matrix[4, :] = np.sin(x)
    # matrix[4, :] = x ** 5
    # matrix[4, :] = x ** 4
    matrix[3, :] = x ** 3
    matrix[2, :] = x ** 2
    matrix[1, :] = x
    matrix[0, :] = 1

    reg = np.linalg.lstsq(matrix.T, f(x), rcond=None)[0]
    reg.round(4)
    ry = np.dot(reg, matrix)

    create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', 'regression'], ['x', 'f(x)'])

    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    X, Y = np.meshgrid(x, y)
    Z = fm((X, Y))
    x = X.flatten()
    y = Y.flatten()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap='coolwarm', linewidth=0.5, antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()