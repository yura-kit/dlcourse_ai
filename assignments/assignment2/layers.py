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
    c = np.max(predictions, keepdims=True, axis=1)
    soft_max = np.exp(predictions - c)/np.sum(np.exp(predictions - c), keepdims=True, axis=1)
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
    px = np.zeros_like(probs)
    for row in range(px.shape[0]):
        px[row,target_index[row]] = 1.
    qx = probs

    ces = np.zeros_like(target_index, dtype=np.float)
    for ix in range(0, target_index.shape[0]):
        ces[ix]= -1. * np.sum(px[ix] * np.log(qx[ix]))
    return  ces.reshape((target_index.shape[0],1))

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength*np.sum(W**2)
    grad = 2*reg_strength*W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
    sft_max = softmax(predictions)
    real1 = np.zeros_like(predictions)
    for row in range(real1.shape[0]):
        real1[row,target_index[row]] = 1.

    der = sft_max - real1

    loss = cross_entropy_loss(sft_max, target_index)
    return np.mean(loss), der/loss.shape[0]


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.cache = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        out = np.maximum(0, X)
        self.cache = X
        #print("ReLU forward out:", out.shape)
        return out

    def backward(self, d_out):
        """
        Backward pass
        the gradient = 0 for x < 0 and 1 for x > 0.
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result, X = d_out, self.cache
        d_result = d_out*(X > 0)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}

#Affine
class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        out = X.dot(self.W.value) + self.B.value
        self.X = X
        #print("FC forward out:", out.shape)
        return out

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        #db = np.sum(d_out, axis=0)
        #self.B.grad = db.reshape(self.B.grad.shape)
        self.B.grad = np.dot(np.ones([1, d_out.shape[0]]), d_out)

        dw = self.X.T.dot(d_out)
        self.W.grad = dw

        dx = d_out.dot(self.W.value.T)
        return dx

    def params(self):
        return {'W': self.W, 'B': self.B}
