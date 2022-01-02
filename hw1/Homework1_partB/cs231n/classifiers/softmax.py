from builtins import range
import numpy as np
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
    num_train=X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        scores = X[i].dot(W)
        scores  -= np.max(scores)  #trick：保证指数后不大
        loss -= np.log(np.exp( scores[y[i]]  )/np.sum(np.exp(scores))  )  #对应的label的loss值
        
        #
        for j in range(W.shape[1]):
            dW[:,j] +=  (np.exp(scores[ j  ]  ) /np.sum(np.exp(scores)) ) * X [i]
        
        #真实值列还需要减X
        dW[:,y[i]] -= X[i]
        
    
    pass
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    
    #对于L2regulation
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train=X.shape[0]
    result = X.dot(W)
    
    result -= result.max()
    result = np.exp(result)
    
    real_label_loss = result [ range(num_train) ,y  ] / np.sum(result ,axis= 1)
    
    loss += (-np.sum(np.log(real_label_loss))/num_train + reg* np.sum(W*W))
    
    #而grad则根据reallabel不同
    left_dw =  result  / np.expand_dims( np.sum(result ,axis= 1) ,axis=  W.shape[1] )
    
    left_dw[ range(num_train) ,y  ] -= 1 
    left_dw = X.T.dot(left_dw)
    left_dw /= num_train
    
    dW += (    left_dw    +2*reg*W)
    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
