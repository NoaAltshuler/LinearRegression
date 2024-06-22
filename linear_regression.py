import numpy as np

class LinearRegression:
    def __init__(self):
        weights_ = None
    def fit(self, X, y):
        """
                Fits the linear regression model to the training data.
        """
        try:
            self.weights_ = np.linalg.inv(np.transpose(X) @ X+ np.eye(X.shape[1])) @ np.transpose(X) @ y
        except np.linalg.LinAlgError:
            raise Exception("The matrix is not invertible.")
    def predict(self,X):
        """
             Predicts the target values for the input features.
        """
        return np.dot(X,self.weights_)
    def score(self,X,y):
        """
             Calculates the R^2 score of the model.
        """
        result = self.predict(X)
        mean = y.mean()
        ssr = sum((y-result)**2)
        sst= sum((y-mean)**2)
        return 1-ssr/sst








