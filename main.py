import matplotlib.pyplot as plt
import skfuzzy as sk
import numpy as np

x = np.arange(-10, 10)


def print_plot(t, y = np.array([])):
    if y.size != 0:
        plt.plot(x, y)
    plt.title(t)
    plt.show()


# (1) ----------------------------------
def standard():
    print_plot("Triangular", sk.trimf(x, [-5, 3, 6]))
    print_plot("Trapezoidal", sk.trapmf(x, [-8, -4, 1, 6]))


# (2) ----------------------------------
def gauss():
    print_plot("Gauss", sk.gaussmf(x, -2, 3))
    print_plot("2-combined Gaussians", sk.gauss2mf(x, -4, 2, 3, 3))


# (3) ----------------------------------
def bell():  # a_width b_slope c_center : y(x) = 1 / (1 + abs([x - c] / a) ** [2 * b])
    print_plot("Generalized Bell", sk.gbellmf(x, 5, 2, 0))


# (4) ----------------------------------
def sigmas(): # y = 1 / (1. + exp[- c * (x - b)])  c_width(+-> l->0 && r->1) (if c==0 -> line)
    print_plot("Sigmoid unilaterial: [offset 2, slope 3]", sk.sigmf(x, 2, -1))
    print_plot("Sigmoids' product", sk.psigmf(x, -3, 1, 3, 1))
    print_plot("Sigmoids' difference", sk.dsigmf(x, -4, 2, 5, 0.5))


# (5) ----------------------------------
def polynomial():
    print_plot("Z-shaped", sk.zmf(x, -3, 4))
    print_plot("S-shaped", sk.smf(x, -7, 7.4))
    print_plot("PI-shaped", sk.pimf(x, -9.2, -5, -3, 7))


# (6) ----------------------------------
y1 = sk.gaussmf(x, 5, 3)
y2 = sk.gaussmf(x, 7, 3)
z1, z2 = sk.fuzzy_and(x, y1, x, y2)
z3, z4 = sk.fuzzy_or(x, y1, x, y2)


def min_max():
    plt.plot(z1, z2, color='red')
    plt.plot(z3, z4, color='black', linestyle='--')
    print_plot("Min")

    plt.plot(z3, z4, color='red')
    plt.plot(z1, z2, color='black', linestyle='--')
    print_plot("Max")


# (7) ----------------------------------
def set_con_and_dis(y, t):
    plt.plot(x, y, color='red')
    plt.plot(x, y1, linestyle='--', color='black')
    plt.plot(x, y2, linestyle='--', color='grey')
    print_plot(t)


def con_and_dis():
    set_con_and_dis(y1 * y2, "Min interpretation")
    set_con_and_dis(y1 + y2 - (y1 * y2), "Max interpretation")


# (8) ----------------------------------
def negation():
    y = sk.fuzzy_not(y2)
    plt.plot(x, y, color='red')
    plt.plot(x, y2, color='black')
    print_plot("Not")


standard()
gauss()
bell()
sigmas()
polynomial()
min_max()
con_and_dis()
negation()
