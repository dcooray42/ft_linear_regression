from matplotlib import pyplot as plt
from train import MyLinearRegression, zscore, reverse_zscore
import numpy as np
import pickle

def display_mse(x, y, xlabel, ylabel) :
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def display_graph(x, y, y_hat, marker, yscatterlabel, y_hatscatterlabel, xlabel, ylabel, loc) :
    plt.figure()
    plt.scatter(x, y, label=yscatterlabel)
    plt.scatter(x, y_hat, marker=marker, label=y_hatscatterlabel)
    plt.plot(x, y_hat, color="orange")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    plt.grid()
    plt.show()

def main() :
    try :
        arr = np.loadtxt("data.csv", delimiter=",", dtype=str)
        x = np.array(arr[1:, 0]).astype(float)
        y = np.array(arr[1:, 1]).astype(float)
        X = zscore(x)
        with open("thetas.pkl", "rb") as f :
            best = pickle.load(f)
    except :
        return
    lr = MyLinearRegression(best["thetas"])
    y_predict = lr.predict_(X)
    y_predict = reverse_zscore(y_predict, y)
    display_mse(np.arange(len(best["mse_values"])), np.array(best["mse_values"]), "Iteration", "MSE")
    display_graph(x, y, y_predict, ".", "True values", "Predicted values", "Km", "Price", "lower right")

if __name__ == "__main__" :
    main()