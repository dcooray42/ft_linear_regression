import numpy as np
import pickle

class MyLinearRegression():
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if type(thetas).__module__ != np.__name__ or not isinstance(alpha, float) or not isinstance(max_iter, int) :
            return None
        if thetas.size <= 0 :
            return None
        if alpha < 0 or alpha > 1 or max_iter < 0 :
            return None
        Thetas = thetas.squeeze().astype(float) if thetas.shape != (1, 1) else thetas.flatten().astype(float)
        if Thetas.ndim != 1 :
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = Thetas
        self.mse_values = []

    def fit_(self, x, y, y_ori) :

        def gradient(x, y, y_ori, theta) :

            def add_intercept(x) :
                if type(x).__module__ != np.__name__ :
                    return None
                if x.size <= 0 :
                    return None
                X = np.copy(x).astype(float, copy=False)
                return (np.insert(X, 0, 1, axis=1)
                        if X.ndim != 1
                        else np.insert(np.transpose(np.expand_dims(X, axis=0)), 0, 1, axis=1))

            if type(x).__module__ != np.__name__ or type(y).__module__ != np.__name__ or type(y_ori).__module__ != np.__name__ or type(theta).__module__ != np.__name__ :
                return None
            if x.size <= 0 or y.size <= 0 or y_ori.size <= 0 or theta.size <= 0 :
                return None
            X = add_intercept(x.astype(float))
            Y = y.squeeze().astype(float) if y.shape != (1, 1) else y.flatten().astype(float)
            Y_ori = y_ori.squeeze().astype(float) if y_ori.shape != (1, 1) else y_ori.flatten().astype(float)
            Theta = theta.squeeze().astype(float) if theta.shape != (1, 1) else theta.flatten().astype(float)
            if Y.ndim != 1 or Theta.ndim != 1 or X.shape[0] != Y.shape[0] or Y.shape[0] != Y_ori.shape[0] or X.shape[1] != Theta.shape[0] :
                return None
            self.mse_values.append(self.mse_(reverse_zscore(Y, Y_ori), reverse_zscore(X.dot(Theta) - Y, Y_ori)))
            return X.T.dot(X.dot(Theta) - Y).reshape(-1, 1) / Y.shape[0]

        if type(x).__module__ != np.__name__ or type(y).__module__ != np.__name__ or type(y_ori).__module__ != np.__name__ or type(self.thetas).__module__ != np.__name__ :
            return None
        if x.size <= 0 or y.size <= 0 or y_ori.size <= 0 or self.thetas.size <= 0 :
            return None
        if self.alpha < 0 or self.alpha > 1 :
            return None
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            return None
        X = x.reshape(-1, 1) if x.ndim == 1 else x.astype(float)
        Y = y.squeeze().astype(float) if y.shape != (1, 1) else y.flatten().astype(float)
        Y_ori = y_ori.squeeze().astype(float) if y_ori.shape != (1, 1) else y_ori.flatten().astype(float)
        Theta = self.thetas.squeeze().astype(float) if self.thetas.shape != (1, 1) else self.theta.flatten().astype(float)
        if Y.ndim != 1 or Theta.ndim != 1 or X.shape[0] != Y.shape[0] or Y.shape[0] != Y_ori.shape[0] or X.shape[1] + 1 != Theta.shape[0] :
            return None
        for _ in range(self.max_iter) :
            gradient_descent = gradient(X, Y, Y_ori, Theta)
            Theta -= self.alpha * gradient_descent.squeeze()
        self.thetas = Theta.reshape(-1, 1)
        return self.thetas
    
    def predict_(self, x) :

        def add_intercept(x) :
            if type(x).__module__ != np.__name__ :
                return None
            if x.size <= 0 :
                return None
            X = np.copy(x).astype(float, copy=False)
            return (np.insert(X, 0, 1, axis=1)
                    if X.ndim != 1
                    else np.insert(np.transpose(np.expand_dims(X, axis=0)), 0, 1, axis=1))

        if type(x).__module__ != np.__name__ or type(self.thetas).__module__ != np.__name__ :
            return None
        if x.size <= 0 or self.thetas.size <= 0 :
            return None
        X = add_intercept(x.astype(float))
        Theta = self.thetas.squeeze().astype(float) if self.thetas.shape != (1, 1) else self.thetas.flatten().astype(float)
        if Theta.ndim != 1 or X.shape[1] != Theta.shape[0] :
            return None
        return X.dot(Theta).reshape(-1, 1)

    def loss_elem_(self, y, y_hat) :
        if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ :
            return None
        if y.size <= 0 or y_hat.size <= 0 :
            return None
        Y = y.squeeze().astype(float) if y.shape != (1, 1) else y.flatten().astype(float)
        Y_hat = y_hat.squeeze().astype(float) if y_hat.shape != (1, 1) else y_hat.flatten().astype(float)
        if Y.ndim != 1 or Y_hat.ndim != 1 :
            return None
        return np.array(list([(Y_hat[i] - value) ** 2] for i, value in enumerate(Y)))

    def loss_(self, y, y_hat) :
        if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ :
            return None
        if y.size <= 0 or y_hat.size <= 0 :
            return None
        Y = y.squeeze().astype(float) if y.shape != (1, 1) else y.flatten().astype(float)
        Y_hat = y_hat.squeeze().astype(float) if y_hat.shape != (1, 1) else y_hat.flatten().astype(float)
        if Y.ndim != 1 or Y_hat.ndim != 1 or Y.shape != Y_hat.shape :
            return None
        return sum(self.loss_elem_(y, Y_hat))[0] / (2 * Y.shape[0])
    
    def mse_(self, y, y_hat) :
        if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ :
            return None
        if y.size <= 0 or y_hat.size <= 0 :
            return None
        Y = y.squeeze().astype(float) if y.shape != (1, 1) else y.flatten().astype(float)
        Y_hat = y_hat.squeeze().astype(float) if y_hat.shape != (1, 1) else y_hat.flatten().astype(float)
        if Y.ndim != 1 or Y_hat.ndim != 1 or Y.shape != Y_hat.shape :
            return None
        return (Y_hat - Y).dot(Y_hat - Y) / Y.shape[0]
    
def data_spliter(x, y, proportion) :
    if type(x).__module__ != np.__name__ or type(y).__module__ != np.__name__ :
        return None
    if x.size <= 0 or y.size <= 0 :
        return None
    if not isinstance(proportion, float) :
        return None
    if proportion < 0 or proportion > 1 :
        return None
    X = x.astype(float)
    Y = y.squeeze().astype(float) if y.shape != (1, 1) else y.flatten().astype(float)
    if Y.ndim != 1 or X.shape[0] != Y.shape[0] :
        return None
    r_indexes = np.arange(X.shape[0])
    np.random.shuffle(r_indexes)
    X = X[r_indexes]
    Y = Y[r_indexes]
    split_num = int(X.shape[0] * proportion)
    return (X[:split_num, :] if X.ndim != 1 else X[:split_num],
            X[split_num:, :] if X.ndim != 1 else X[split_num:],
            Y[:split_num].reshape(-1, 1),
            Y[split_num:].reshape(-1, 1))

def zscore(x) :   
    if type(x).__module__ != np.__name__ :
            return None
    if x.size <= 0 :
        return None
    X = np.squeeze(x).astype(float)
    if X.ndim != 1 :
        return None
    return (X - np.mean(X)) / np.std(X)

def reverse_zscore(x, x_ori) :   
    if type(x).__module__ != np.__name__ and type(x_ori).__module__ != np.__name__ :
            return None
    if x.size <= 0 and x_ori.size <= 0 :
        return None
    X = np.squeeze(x).astype(float)
    X_ori = np.squeeze(x_ori)
    if X.ndim != 1 and X_ori.ndim != 1 :
        return None
    return (X * np.std(X_ori)) + np.mean(X_ori)
    
def main() :
    try :
        arr = np.loadtxt("data.csv", delimiter=",", dtype=str)
        x = np.array(arr[1:, 0]).astype(float)
        y = np.array(arr[1:, 1]).astype(float)
        x = zscore(x)
        Y = zscore(y)
    except :
        return
    lr = MyLinearRegression(np.array([0, 0]), 1e-5, 1000000)
    thetas = lr.fit_(x, Y, y)
    with open("thetas.pkl", "wb") as f:
        pickle.dump({"thetas" : thetas, "mse_values" : lr.mse_values}, f)

if __name__ == "__main__" :
    main()