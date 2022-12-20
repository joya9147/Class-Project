# Utilities for INFO 5604
# Fall 2022
# Lab 6 solution version

# Imports
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings 
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import mean_squared_error


def sigmoid(z):
    """
    Sigmoid (or logit) function. From a value to a probability.
    """
    return 1.0 / (1.0 + np.exp(-z))

def inv_sigmoid(p):
    """
    Inverse of the logit functon. From a probabiity to the associated value.
    """
    return - np.log((1/p) - 1)


def print_accuracies(classifier, X_train, y_train, X_test, y_test):
    """
    Print the training and test accuracy of a classifier. It also returns the two values
    in case they are needed for further processing.
    
    Arguments
    classifier
    X_train
    y_train
    X_test
    y_test
    
    """
    
    train_acc = classifier.score(X_train, y_train)
    test_acc = classifier.score(X_test, y_test)
    
    print(f'Train accuracy {train_acc:.3f}\tTest accuracy {test_acc:.3f}')
    
    return [train_acc, test_acc]

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """
    Plots 2D decision regions. From _Python Machine Learning 3rd ed._ by Raschka and Mirjalili
    """

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore", category=UserWarning, message=".*You passed a edgecolor/edgecolors.*")

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], 
                        y=X[y == cl, 1],
                        alpha=0.8, 
                        color=colors[idx],
                        marker=markers[idx], 
                        label=cl, 
                        edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black',
                    alpha=1.0, linewidth=1,
                    marker='o', s=100, label='test set') 
        
def print_errors(regressor, X_train, y_train, X_test, y_test):
    
    train_error = mean_squared_error(y_train, regressor.predict(X_train))
    test_error  = mean_squared_error(y_test, regressor.predict(X_test))
    
    print(f'Train error {train_error:.3f}\tTest error {test_error:.3f}')
    
    return [train_error, test_error]
        
def plot_learning_curve(train_fractions, train_scores, test_scores, score_name='Accuracy'):
    ### Implement in lab
    pass
    