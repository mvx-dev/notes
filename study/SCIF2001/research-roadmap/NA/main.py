import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import numpy as np


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def fit(df):
    angle = df.iloc[:, 0]
    power = df.iloc[:, 1]

    popt, pcov = sc.optimize.curve_fit(gaussian, power, angle)

    return popt, pcov


if __name__ == "__main__":
    df = pd.read_csv("data.csv", delimiter="\t")

    popt, pcov = fit(df)
    print(popt)
    print(pcov)

    plt.errorbar(df.iloc[:, 0], df.iloc[:, 1],
                 xerr=df.iloc[:, 2], yerr=df.iloc[:, 3], fmt=".")

    plt.plot(df.iloc[:, 0], gaussian(df.iloc[:, 0], *popt))

    plt.title("Output Power vs Input Angle")
    plt.xlabel(r"Input Angle ($\theta$)")
    plt.ylabel(r"Output Power ($\mu W$)")
    plt.savefig("output.png")
    plt.plot()
