import numpy as np
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    # Interate through training samples
    # Calculate scores of each sample, find the score of the correct class
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        # Iterate through predicted scores
        # Skip the score of the correct one,
        # sum the loss across all the other scores if it is greater than correct_class_score + 1
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1

            # Meaning that the margin is not big enough, need to update derivative term here
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]

        dW[:, y[i]] -= dW.dot(np.ones([num_classes]))

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)   # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO:
    # Implement a vectorized version of the structured SVM loss, storing the
    # result in loss.
    #############################################################################

    # Calculate all loss score for each element
    scores = X.dot(W)

    # Get the score of the correct class in each sample
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)

    # Calculate margin for each sample
    margins = scores - correct_class_scores + 1  # note delta = 1

    # Replace all correct margin to zero so that we do not count them
    margins[np.arange(num_train), y] = 0
    margins_mask = margins > 0

    # Set loss equal to sum of all elements in margins and average them.
    loss = np.sum(margins[margins_mask]) / num_train
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    # TODO:
    # Implement a vectorized version of the gradient for the structured SVM
    # loss, storing the result in dW.
    #
    # Hint: Instead of computing the gradient from scratch, it may be easier
    # to reuse some of the intermediate values that you used to compute the
    # loss.
    #############################################################################

    # Convert margins greater than zero to 1
    margin_check = margins_mask.copy().astype(int)

    # For margins of correct classes of each sample, count the number of classes to be updated
    margin_check[np.arange(num_train), y] = -margins_mask.dot(np.ones(num_classes))

    # Update gradient with all samples at once with np.dot() and average them
    dW = X.T.dot(margin_check) / num_train
    dW += reg * W

    return loss, dW
