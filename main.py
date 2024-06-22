
import matplotlib.pyplot as plt
from linear_regression import LinearRegression as lr
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

def question3():
    print("Question 3:")
    df = pd.read_csv('simple_regression.csv', delimiter=',')
    X_train,X_test, y_train,y_test = prepData(df)
    q3 = lr()
    q3.fit(X_train, y_train)
    print("weights are", *q3.weights_)
    print("coefficient of determination is: ", round(q3.score(X_test, y_test),2))

def question4():
    print("\nQuestion 4")
    """
     preparing data for fit
    """
    df = fetch_california_housing(as_frame=True)
    X = df.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ones_column = np.ones((X_scaled.shape[0], 1))  # Column of ones
    X_scaled = np.concatenate((X_scaled, ones_column), axis=1)
    y= df.target
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    q4 = lr()
    q4.fit(X_train,y_train)
    print("weights are", pd.DataFrame(q4.weights_),2)
    print("coefficient of determination is: ", round(q4.score(X_test, y_test),2))

def question5():
    print("\nquestion 5:")
    """ 
    preapring data
    """
    df = pd.read_csv('Students_on_Mars.csv', delimiter=',')
    X, y = splitToFeaturesAndAnswers(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """"
    Note: From the 4th polynomial degree, the prediction starts to diverge,all values bigger then 1.0
    """
    error_rate = list()
    degrees = range(1,6)
    for i in range(1,6):
        q5 = lr()
        poly = PolynomialFeatures(i)
        X_poly_train = poly.fit_transform(X_train)
        q5.fit(X_poly_train, y_train)
        X_poly_test = poly.transform(X_test)
        res = 1-q5.score(X_poly_test,y_test)
        """
        when the error rate is above 2 the exect value is not
        relevant because it is bigger than 1.0
        """
        if res >2:
            error_rate.append(2)
        else:
            error_rate.append(res)
        print('degree ', {i}, 'error rate ', {res})
    plt.plot(degrees, error_rate, marker='o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error rate')
    plt.title('Error rate vs Polynomial Degree')
    plt.grid(True)
    plt.show()


def splitToFeaturesAndAnswers(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X,y
def prepData(df):
    X, y = splitToFeaturesAndAnswers(df)
    ones_column = np.ones((X.shape[0], 1))  # Column of ones
    X = np.concatenate((X, ones_column), axis=1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

question3()
question4()
question5()


