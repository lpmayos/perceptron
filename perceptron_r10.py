#-*- encoding: utf8 -*-
import numpy
import time
import sys


class Perceptron(object):
    """ Perceptron Learning Algorithm implementation
    """

    def __init__(self, N, D):
        """ generates the training set; the pla will try to aproximate self.hyperplane
        using this training set
        """
        # generate d-dimensional hyperplane
        self.hyperplane = self._generate_hyperplane(D)
        # generate a training set using the line
        self.X = self._generate_training_set(N, D)

    def _generate_hyperplane(self, D):
        """ generates an hyperplane of 10 dimensions
        """
        # coeficientes A..J
        coeficientes = [numpy.random.uniform(-1, 1) for i in range(D)]
        #calcular coeficiente K para aseguraros que fijamos una de las coordenadas
        # por ejemplo, asignamos 0.5 a todas las incógnitas (en el plano sería Ax + By + Cz + D = 0 --> D = -(A·0.5 + B·0.5 + C·0.5))
        K = 0
        for i in range(D):
            K += 0.5 * coeficientes[i]
        coeficientes.append(-K)
        return numpy.array(coeficientes)

    def _generate_random_numbers(self, N):
        """ return N random numbers within [-1, 1]
        """
        return [numpy.random.uniform(-1, 1) for i in range(N)]

    def _generate_training_set(self, N, D):
        """ generates N random points within [-1, 1] x [-1, 1] and evaluates
        them with self.hyperplane
        """
        X = []
        for i in range(N):
            # hyperplane: A, B, ..., K
            # point x1, x2, ..., x10
            point = numpy.array(self._generate_random_numbers(D) + [1])
            s = int(numpy.sign(self.hyperplane.T.dot(point)))
            X.append((point, s))
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
 
    def pla(self, D):
        """ Perceptron Learnin Algorithm
        """
        # initialize the weigths to zeros
        X, N = self.X, len(self.X)
        w = numpy.zeros(D + 1)
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

    try:
        D = int(sys.argv[2])
    except:
        print "** ERROR ** You should pass a number as an argument, indicating the number of dimensions you want to use"
        sys.exit(2)

    start = time.time()
    p = Perceptron(N, D)
    iterations = p.pla(D)
    end = time.time()

    print 'D = %s; N = %s; Iterations = %s; Time = %s seconds' % (str(D), str(N), str(iterations), str(end - start))
