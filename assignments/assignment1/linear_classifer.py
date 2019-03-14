import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    c = np.max(predictions, keepdims=True, axis=1)
    soft_max = np.exp(predictions - c)/np.sum(np.exp(predictions - c), keepdims=True, axis=1)
    #print("SOFTMAX", soft_max)
    return soft_max


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    #print(probs)
    # print("CALL CES")
    # print("probs inside ces", probs)
    # print("target inside ces", target_index)
    px = np.zeros_like(probs)
    #print("PX", probs)
    for row in range(px.shape[0]):
        px[row,target_index[row]] = 1.
    #px[:,target_index] = 1.
    #print(px)
    qx = probs


    ces = np.zeros_like(target_index, dtype=np.float)
    for ix in range(0, target_index.shape[0]):
        ces[ix]= -1. * np.sum(px[ix] * np.log(qx[ix]))
    #print(target_index.shape)
    #print("CES", ces.reshape((target_index.shape[0],1)))
    return  ces.reshape((target_index.shape[0],1))



# def softmax_with_cross_entropy_no_batch(predictions, target_index):
#     '''
#     Computes softmax and cross-entropy loss for model predictions,
#     including the gradient
#
#     Arguments:
#       predictions, np array, shape is either (N) or (batch_size, N) -
#         classifier output
#       target_index: np array of int, shape is (1) or (batch_size) -
#         index of the true class for given sample(s)
#
#     Returns:
#       loss, single value - cross-entropy loss
#       dprediction, np array same shape as predictions - gradient of predictions by loss value
#     '''
#     # TODO implement softmax with cross-entropy
#     sft_max = softmax(predictions)
#     real1 = np.zeros_like(predictions)
#     real1[target_index] = 1
#     der = sft_max - real1
#     loss = cross_entropy_loss(sft_max, target_index)
#     print("LOSS", loss)
#     return loss, der/loss[0]

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy minibatch
    sft_max = softmax(predictions)
    real1 = np.zeros_like(predictions)
    #print("REAL1 BEFORE ",real1)
    #print("TARGET", target_index)
    for row in range(real1.shape[0]):
        real1[row,target_index[row]] = 1.
    #print("REAL1 APPLIED",real1)

    der = sft_max - real1

    #print("DER", der)
    loss = cross_entropy_loss(sft_max, target_index)
    #print("LOSS", loss)
    return np.mean(loss), der/loss.shape[0]

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    loss = reg_strength*np.sum(W**2)
    #print("LOSS", loss)
    grad = 2*reg_strength*W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    #predictions += np.ones((1, 2))
    #print("PRED", predictions)
    #print("PREDICTIONS", predictions, "TARGET", target_index)
    # TODO implement prediction and gradient over W

    loss, dW = softmax_with_cross_entropy(predictions, target_index)
    #print("LOSS RESULT", loss, "dW", dW)
    return loss, np.dot(X.T, dW)


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''
        print("X", X.shape)
        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss = 0.

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            #print("SECTIONS", sections)
            #print("BATCH",batches_indices[0])
            for batch_i in batches_indices:
                loss = 0.
                X_batch = X[batch_i]
                y_batch = y[batch_i]
                loss, grad = linear_softmax(X_batch, self.W, y_batch)
                l2, grad2 = l2_regularization(self.W, reg_strength=reg)
                self.W += (-1)*(grad + grad2) * learning_rate
                loss = loss + l2
                loss_history.append(loss)
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!

            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        y_pred = np.argmax(X.dot(self.W), axis=1)



        return y_pred



                
                                                          

            

                
