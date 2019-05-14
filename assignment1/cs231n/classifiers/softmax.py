import numpy as np
from random import shuffle

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

  for i in range(X.shape[0]):

    denominator_loss = np.array([np.dot(W[:, j], X[i]) for j in range(W.shape[1])])
    numerator_loss = np.dot(W[:, y[i]], X[i])

    #Avoid numerical issues
    maxf = np.max(denominator_loss)
    numerator_loss -= maxf
    denominator_loss -= maxf

    #Compute loss
    loss = loss - np.log(np.exp(numerator_loss) / np.sum(np.exp(denominator_loss)))

    #Compute gradient
    for j in range(W.shape[1]):
      dW[:, j] = dW[:, j] + (np.exp(np.dot(W[:, j], X[i]) - maxf) / np.sum(np.exp(denominator_loss)) - (j == y[i])) * X[i] 

  loss /= X.shape[0]
  dW /= X.shape[0]


  #Adding regularization to loss and to dW
  loss += reg * np.sum(W * W)  
  dW += 2 * reg * W


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
  num_train = X.shape[0]

  #Cross-entropy loss function
  W_yi = W[:, y]

  f_i = np.sum(X * W_yi.T, axis=1, keepdims=True)
  f_j = np.dot(X, W)

  maxes = np.max(f_j, axis=1, keepdims=True)
  f_i -= maxes
  f_j -= maxes

  loss = -np.sum(np.log(np.exp(f_i) / np.sum(np.exp(f_j), axis=1, keepdims=True)), axis=0)
  loss /= num_train
  loss += reg * np.sum(W * W)


  #Softmax cross-entropy loss gradient
  XW = np.dot(X, W)
  maxes = np.max(XW, axis=1, keepdims=True)
  XW -= maxes

  exw = np.exp(XW)

  frac = exw / np.sum(exw, axis=1, keepdims=True)
  
  sub = np.zeros(frac.shape)
  sub[np.arange(num_train), y] = 1

  frac = frac - sub

  dW = np.dot(X.T, frac)
  dW /= num_train
  dW += 2 * reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

