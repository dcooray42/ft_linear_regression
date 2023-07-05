from matplotlib import pyplot as plt
from train import MyLinearRegression, zscore, reverse_zscore
import numpy as np
import pickle

def display_metrics(x, y, xlabel, ylabel) :
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

def reverse_zscore_one(value, x_ori) :   
    if not isinstance(value, float) or type(x_ori).__module__ != np.__name__ :
        print(type(value))
        return None
    if x_ori.size <= 0 :
        return None
    X_ori = np.squeeze(x_ori)
    if X_ori.ndim != 1 :
        return None
    return (value * np.std(X_ori)) + np.mean(X_ori)

def zscore_one(value, x) :   
    if type(x).__module__ != np.__name__ :
            return None
    if x.size <= 0 :
        return None
    X = np.squeeze(x).astype(float)
    if X.ndim != 1 :
        return None
    return (value - np.mean(X)) / np.std(X)

def main() :
    try :
        arr = np.loadtxt("data.csv", delimiter=",", dtype=str)
        x = np.array(arr[1:, 0]).astype(float)
        y = np.array(arr[1:, 1]).astype(float)
        X = zscore(x)
        with open("thetas.pkl", "rb") as f :
            best = pickle.load(f)
            thetas = best["thetas"]
    except :
        thetas = np.array([0, 0])
    try :
        value = zscore_one(int(input("Enter a mileage : ")), x)
    except :
        print("Please enter a valid number")
        return
    lr = MyLinearRegression(thetas)
    y_predict = lr.predict_(X)
    y_predict = reverse_zscore(y_predict, y)
    estimatedPrice = float(thetas[0] + thetas[1] * value)
    print(f"estimatedPrice({value}) = {float(thetas[0])} + {float(thetas[1])} * {value} = {estimatedPrice}")
    if thetas[0] != 0 and thetas[1] != 0 :
        print(f"reverse_zscore(estimatedPrice({value})) = {reverse_zscore_one(estimatedPrice, y)}")
        display_metrics(np.arange(len(best["mse_values"])), np.array(best["mse_values"]), "Iteration", "MSE")
        display_metrics(np.arange(len(best["rmse_values"])), np.array(best["rmse_values"]), "Iteration", "RMSE")
        display_metrics(np.arange(len(best["mae_values"])), np.array(best["mae_values"]), "Iteration", "MAE")
        display_metrics(np.arange(len(best["r2_values"])), np.array(best["r2_values"]), "Iteration", "R2 Score")
        display_graph(x, y, y_predict, ".", "True values", "Predicted values", "Km", "Price", "lower right")
    print(f"MSE : {lr.mse_(y, y_predict)}")
    print(f"RMSE : {lr.rmse_(y, y_predict)}")
    print(f"MAE : {lr.mae_(y, y_predict)}")
    print(f"R2 Score : {lr.r2score_(y, y_predict)}")

if __name__ == "__main__" :
    main()