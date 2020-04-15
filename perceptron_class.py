##### Update this Class #####
class Perceptron(object):
    def __init__(self, rate = 0.01, niter = 10):
        self.rate = rate
        self.niter = niter
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def __sigmoid_derivative(self, x):
        sx = sigmoid(x)
        return sx * (1-sx)
    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        """
        # Initialize Weights
        self.weight = np.zeros(X.shape[1])
        self.bias = np.ones(1 + X.shape[0])
#         X = np.concatenate((X, self.bias), axis=1)
        # Number of misclassifications
        self.errors = []
        for i in range(self.niter):
            err = 0
            for xi, target in zip(X, y):
                delta_w = self.rate * (target - self.predict(xi))
                # Activate
                self.weight[:] += delta_w * xi
                self.weight[0] += delta_w 
                #calculate error
                err += int(delta_w != 0.0)
            self.errors.append(err)
        return self
    def net_input(self, X):
        return np.dot(X, self.weight) 
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.5, 1, 0)