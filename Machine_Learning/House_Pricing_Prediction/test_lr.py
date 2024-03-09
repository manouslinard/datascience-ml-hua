import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from linear_regression import LinearRegression

def exercise_3_1(seed: int = 42):
    """Answer to exercise 3_1."""
    dataset = datasets.fetch_california_housing()

    # test size is 30 %:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=seed)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    _, MSE = lr.evaluate(X_test, y_test)

    RMSE = np.sqrt(MSE)
    return RMSE

def exercise_3_2():
    """Answer to exercise 3_2."""
    l = []
    for _ in range (20):
        l.append(exercise_3_1(seed=None))   # no seed
    print("Average RMSE:", np.mean(l))
    print("Standard Deviation of RMSE:", np.std(l))

def exercise_3_3():
    """Answer to exercise 3_3."""
    dataset = datasets.fetch_california_housing()
    l = []
    for _ in range(20):
        lr = linear_model.LinearRegression()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.data, dataset.target, test_size=0.3)
        lr.fit(X_train, y_train)
        yhat = lr.predict(X_test)

        mse_ols = np.mean((y_test - yhat) ** 2)
        RMSE = np.sqrt(mse_ols)
        l.append(RMSE)
    print("Average RMSE:", np.mean(l))
    print("Standard Deviation of RMSE:", np.std(l))


if __name__ == "__main__":
    print("Exercise 3_1 Answer ======== :")
    print("RMSE is:", exercise_3_1())
    print("======================")

    print("Exercise 3_2 Answer ======== :")
    exercise_3_2()
    print("======================")

    print("Exercise 3_3 Answer ======== :")
    print("Sklearn linear regression:")
    exercise_3_3()
    print("======================")
