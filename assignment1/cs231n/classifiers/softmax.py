import numpy as np
from math import log, exp
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) # (N, C)
    max_score = np.max(scores)
    scores_exp = np.exp(scores - max_score)
    loss += -log(scores_exp[y[i]]) + log(np.sum(scores_exp))
    for j in xrange(num_classes):
        Sj = scores_exp[j] / np.sum(scores_exp)
        if j == y[i]:
            dW[:, j] += X[i] * (Sj - 1)
        else:
            dW[:, j] += X[i] * Sj
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  scores = X.dot(W) # (N, C)
  max_scores = np.amax(scores, axis=1) # (N,)
  scores_exp = np.exp(scores - max_scores[:,None])
  loss += -np.sum(np.log(scores_exp[range(num_train), y])) + np.sum(np.log(np.sum(scores_exp, axis=1)))

  S = scores_exp / np.sum(scores_exp, axis=1)[:,None] # (N, C)
  S_corr = S - np.equal(np.arange(num_classes), y[:,None]) # (N, C)
  dW += np.dot(X.T, S_corr) # (D, C)

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
