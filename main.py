import matplotlib.pyplot as plt
import skfuzzy as sk
import numpy as np

x = np.arange(-10, 10)

def PrintPlot(y, t):
    plt.plot(x, y)
    plt.title(t)
    plt.show()

# (1) ----------------------------------
def Standard():
    PrintPlot(sk.trimf(x, [-5, 3, 6]), "Triangular: [-5 : 3 : 6]")
    PrintPlot(sk.trapmf(x, [-8, -4, 1, 6]), "Trapezoidal: [-8 : -4 : 1 : 6]")

# (2) ----------------------------------
def Gauss():
    PrintPlot(sk.gaussmf(x, -2, 3), "Gauss: [-2 mean, 3 sigma]")
    PrintPlot(sk.gauss2mf(x, -4, 2, 3, 3), "2-combined Gaussians: [-4, 2, 3, 3]")

# (3) ----------------------------------
def Bell():  # a_width b_slope c_center : y(x) = 1 / (1 + abs([x - c] / a) ** [2 * b])
    PrintPlot(sk.gbellmf(x, 5, 2, 0), "Generalized Bell: [width 5, slope 2, center 0]")

# (4) ----------------------------------
def Sigmas(): # y = 1 / (1. + exp[- c * (x - b)])  c_width(+-> l->0 && r->1) (if c==0 -> line)
    PrintPlot(sk.sigmf(x, 2, -1), "Sigmoid unilaterial: [offset 2, slope 3]")
    PrintPlot(sk.psigmf(x, -3, 1, 3, 1), "Sigmoids' product: [-3, 1, 3, 1]")
    PrintPlot(sk.dsigmf(x, -4, 2, 5, 0.5), "Sigmoids' difference: [-4, 2, 5, 0.5]")

# (5) ----------------------------------
def Polynomial():
    PrintPlot(sk.zmf(x, -3, 4), "Z-shaped: [ceiling -3, floor 4]")
    PrintPlot(sk.smf(x, -7, 7.4), "S-shaped: [ceiling -7, floor 7.4]")
    PrintPlot(sk.pimf(x, -9.2, -5, -3, 7), "PI-shaped: [-9.2, -5, -3, 7]")

# (6) ----------------------------------
def Minimax():
    y1 = sk.gaussmf(x, 5, 3)
    y2 = sk.gaussmf(x, 7, 3)
    z1, z2 = sk.fuzzy_and(x, y1, x, y2)
    z3, z4 = sk.fuzzy_or(x, y1, x, y2)

    plt.plot(z1, z2, color='red')
    plt.plot(z3, z4, color='black', linestyle='--')
    plt.title("Min")
    plt.show()

    plt.plot(z3, z4, color='red')
    plt.plot(z1, z2, color='black', linestyle='--')
    plt.title("Max")
    plt.show()

# (7) ----------------------------------
#def ConAndDis():


# (8) ----------------------------------

# Standard()
# Gauss()
# Bell()
# Sigmas()
# Polynomial()
# Minimax()
# ConAndDis()

