from builtins import range
import numpy as np
from random import shuffle
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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)   #W是十个分类器的参数,10*3073
        correct_class_score = scores[y[i]] #y_pred
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                
                #同时也存在dw:y[i]对应的类,一维的可以不转置加减,real_label
                dW[:, y[i] ] -= X[ i ,: ] 
                
                #
                dW[:, j ] += X[ i, :]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    
    #对于L2regulation
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    #向量化实现
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train=(X.shape[0])
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    result = X.dot(W)
    sub = result[ range(num_train), y ] .reshape( num_train ,1  )
    result -= sub - 1  #delta=1
    #对应real——label列设置为0
    result[ range(num_train) , y]=0 
    
    result = np.maximum(result,0)  #保存哪些会发生更新，后续梯度使用
    
    loss = np.sum(result)/num_train
    
    #reg
    penality = reg* np.sum( W*W )
    
    loss += penality
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    '''
    1.根据上面的result 4900 * 10 判断哪些w需要更新
    2.那些real_label对应的梯度需要更新多次
    3.X.T.dot(X_dw)指的是对于每一个分类器每一个w都需要看它的更新情况
    4.其他需要被处理的
    '''
    
    X_dw = np.zeros(result.shape)
    #会发生更新的，对应的result矩阵中都大于0
    X_dw [ result > 0 ] = 1
    
    #真实的laebl每次都要发生 -1的更新
    X_dw [ list(range(num_train))  , y  ] = - np.sum(X_dw ,axis=1)
    
    dW = X.T.dot( X_dw )
    dW /= num_train 
    #向量格式
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
