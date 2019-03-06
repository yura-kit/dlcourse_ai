import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
                #dists[i_test][i_train] = np.sqrt(np.sum((X[i_test] - self.train_X[i_train])**2))
                dists[i_test][i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]), axis=0)
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        The logic is simple — for each test vector I subtract it from the entire training matrix.
        I use a trick called numpy addition broadcasting. As long as the dimensions match up,
        numpy knows to do a row-wise subtraction if the element on the right is one-dimensional.
        After this subtraction, I simply element-wise square and sum along the column dimension
        to get a single row of the distance matrix for test vector i.

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        #print(dists.shape)
        #print(self.train_X.shape)
        for i_test in range(num_test):
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops
            # dists[i_test][i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]), axis=0)
            dists[i_test, :] = np.sum(np.abs(X[i_test, :] - self.train_X), axis=1)
            #print(dists[i_test].shape)
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Fully vectorizes the calculations
        NUMPY
        https://habr.com/en/post/415373/

        https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        -------------------------------------------------------------------------------------------
        I need to vectorize more. How? After some Googling and old-fashioned pen and paper
        I figured out better way. The “trick” is to expand the l2 distance formula.
        For each vector x and y, the l2 distance between them can be expressed as:
        (x - y)^2 = x^2 + y^2 - 2xy

        To vectorize efficiently, we need to express this operation for ALL the vectors at once in numpy.
        The first two terms are easy — just take the l2 norm of every row in the matrices X and X_train.
        The last term can be expressed as a matrix multiply between X and transpose(X_train).
        Numpy can do all of these things super efficiently.
        -------------------------------------------------------------------------------------------
        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        #print('required shape', dists.shape)
        # TODO: Implement computing all distances with no loops!
        #print('new shape of X', X[:,np.newaxis].shape )
        #print('new shape of train_X', self.train_X[np.newaxis,:].shape)

        dists = np.abs(X[:,np.newaxis] - self.train_X[np.newaxis,:])
        dists = np.apply_along_axis(np.sum, 2, dists)
        #print('X',X.shape)
        #print('train X',self.train_X.shape)
        #print('result',dists.shape)
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        print('dist shape', dists.shape)
        #print(pred.shape)
        #print(pred)
        print('y shape',self.train_y.shape)
        print('result shape', pred.shape)
        print('k=',self.k)
        print(pred)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            #print(dists[i])
            #print(np.argsort(dists[i, :])[:self.k])
            closest_y = self.train_y[np.argsort(dists[i, :])[:self.k]]# sort by value then get indexes by slicing till k (first k in each row)

            u, indices = np.unique(closest_y, return_inverse=True)
            #print('U',u, indices)
            #print(u[np.argmax(np.bincount(indices))])
            pred[i] = u[np.argmax(np.bincount(indices))]
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            closest_y = self.train_y[np.argsort(dists[i, :])[:self.k]]
            # sort by value then get indexes by slicing till k (first k in each row)
            print((closest_y))
            u, indices = np.unique(closest_y, return_inverse=True)
            print('U',u, indices)
            print(u[np.argmax(np.bincount(indices))])
            pred[i] = u[np.argmax(np.bincount(indices))]
        return pred
