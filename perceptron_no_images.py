import numpy
import time
import sys


class Perceptron(object):
    """ Perceptron Learning Algorithm implementation
    """

    def __init__(self, N):
        """ generates the training set; the pla will try to aproximate self.V
        using this training set
        """
        # define two random points
        xA, yA, xB, yB = self._generate_random_numbers(4)
        # define a line given the two points
        self.V = numpy.array([xB * yA - xA * yB, yB - yA, xA - xB])
        # generate a training set using the line
        self.X = self._generate_training_set(N)

    def _generate_random_numbers(self, N):
        """ return N random numbers within [-1, 1]
        """
        return [numpy.random.uniform(-1, 1) for i in range(N)]

    def _generate_training_set(self, N):
        """ generates N random points within [-1, 1] x [-1, 1] and evaluates
        them with self.V
        """
        X = []
        for i in range(N):
            xA, yA = self._generate_random_numbers(2)
            x = numpy.array([1, xA, yA])
            s = int(numpy.sign(self.V.T.dot(x)))
            X.append((x, s))
        return X

    def _has_misclassified_points(self, vec):
        """ returns the fraction of misclassified points
        """
        X, N = self.X, len(self.X)
        for x, s in X:
            if int(numpy.sign(vec.T.dot(x))) != s:
                return True
        return False
 
    def _choose_miscl_point(self, vec):
        """ returns the first misclassified point it finds
        """
        X = self.X
        for x, s in X:
            if int(numpy.sign(vec.T.dot(x))) != s:
                return (x, s)
 
    def pla(self):
        """ Perceptron Learnin Algorithm
        """
        # initialize the weigths to zeros
        X, N = self.X, len(self.X)
        w = numpy.zeros(3)
        iteration = 0
        # iterate until all points are correctly classified
        while self._has_misclassified_points(w):
            iteration += 1
            # pick random misclassified point
            x, s = self._choose_miscl_point(w)
            # update weights
            w += s * x
        self.w = w
        return iteration


if __name__ == "__main__":
    try:
        N = int(sys.argv[1])
    except:
        print "** ERROR ** You should pass a number as an argument, indicating the length of the dataset you want to use"
        sys.exit(2)

    start = time.time()
    p = Perceptron(N)
    iterations = p.pla()
    end = time.time()

    print 'N = %s; Iterations = %s; Time = %s seconds' % (str(N), str(iterations), str(end - start))
