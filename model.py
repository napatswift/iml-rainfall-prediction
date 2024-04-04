import numpy as np
import math

# Linear regression in one dimension
class MyLinearRegression:
    
    # Class constructor
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.w = None
        self.w_dim = None
        self.losses = None
        

    def fit(self, x: np.ndarray, y: np.ndarray, iteration=1500):
        """
        Fit the linear regression model to the given data.

        Parameters:
        x (numpy.ndarray): The input features.
        y (numpy.ndarray): The target values.
        iteration (int): The number of iterations for training. Default is 1500.

        Returns:
        self (MyLinearRegression): The fitted linear regression model.
        """

        # Set the target values
        self.y = y

        # Determine the dimension of the weight vector
        self.w_dim = x.shape[-1] + 1  # plus one for bias

        # Initialize the weight vector with zeros
        self.w = np.zeros(self.w_dim)

        # Append a column of ones to the input features for the bias term
        self.x = np.append(x, np.ones((x.shape[0], 1)), axis=1)

        # Initialize an array to store the losses for each iteration
        self.losses = np.ones(iteration)

        # Perform gradient descent for the specified number of iterations
        for i in range(iteration):
            self.losses[i] = self._make_one_update()

        return self
        

    def _make_one_update(self):
        """
        Perform one update step of gradient descent.

        Returns:
        float: The loss after the update step.
        """
        # Make a copy of the current weight vector
        w_current = self.w.copy()

        # Compute the step size using the learning rate and the gradient
        step = -1 * self.alpha * self._compute_gradient(w_current)

        # Update the weight vector
        w_update = w_current + step

        # Calculate the loss after the update
        update_loss = self.sq_loss(w_update)

        # Update the weight vector with the new values
        self.w = w_update

        # Return the loss after the update
        return update_loss
    
    def _compute_gradient(self, w_current: np.ndarray):
        """
        Compute the gradient of the loss function with respect to the weight vector.

        Parameters:
        w_current (numpy.ndarray): The current weight vector.

        Returns:
        numpy.ndarray: The computed gradient.
        """

        grad_v = np.zeros(self.w_dim)

        predictions = self.x @ w_current
        errors = self.y - predictions

        grad_v[:-1] = -2 / self.x.shape[0] * np.sum(self.x[:, :-1] * errors[:, None], axis=0)

        grad_v[-1] = -2 / self.x.shape[0] * np.sum(predictions)

        return grad_v / math.sqrt(np.inner(grad_v, grad_v))

    def sq_loss(self, w: np.ndarray) -> float:
        """
        Calculates the squared loss for linear regression.

        Parameters:
        w (numpy.ndarray): the weight vector

        Returns:
        float: the mean squared loss
        """
        return np.mean((self.y - self.x @ w)**2)

    def predict(self, X:  np.ndarray):
        """
        Predicts using linear regression model

        Parameters:
        X (numpy.ndarray): features

        Returns:
        numpy.ndarray: predicted values
        """

        x = np.append(X.copy(), np.ones((X.shape[0], 1)), axis=1)

        return x @ self.w