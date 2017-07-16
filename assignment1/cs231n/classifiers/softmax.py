import numpy as np


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
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)

        # Normalization tricks
        scores -= np.max(scores)

        # Compute loss
        # Li = -log(e^s(yi) / sigma(e^s(j)))
        sum_scores = 0
        for j in range(num_classes):
            # Store the score of the correct class
            if j == y[i]:
                score_y = np.e**scores[j]

            # Accumulate the scores across all categories
            sum_scores += np.e**scores[j]

        # Compute gradient
        for j in range(num_classes):
            score_j = np.e**scores[j]
            dW[:, j] += (score_j / sum_scores - (j == y[i])) * X[i]

        loss += -np.log(score_y / sum_scores)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    # Calculate all loss score for each element
    scores = X.dot(W)

    # Normalization tricks
    scores -= np.max(scores, axis=1).reshape(num_train, 1)

    # Compute loss
    exp_scores = np.e ** scores

    correct_class_scores = exp_scores[np.arange(num_train), y].reshape(num_train, 1)
    sum_class_scores = np.sum(exp_scores, axis=1).reshape(num_train, 1)

    loss = -np.log(correct_class_scores / sum_class_scores)
    loss = loss.mean() + 0.5 * reg * np.sum(W * W)

    # Compute gradient
    y_table = np.indices(exp_scores.shape)[1]
    y_table = y_table == y.reshape(num_train, 1)

    dW = np.dot(X.T, (exp_scores / sum_class_scores - y_table))
    dW /= num_train
    dW += reg * W

    return loss, dW
