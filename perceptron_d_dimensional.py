#-*- encoding: utf8 -*-
import numpy
import time
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
from images2gif import writeGif


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
            point = numpy.array([1] + self._generate_random_numbers(D))
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
 
    def _plot(self, mispts=None, vec=None, save=False):
        """
        """
        plt.figure()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        V = self.hyperplane
        a, b = -V[1] / V[2], -V[0] / V[2]
        l = numpy.linspace(-1, 1)
        plt.plot(l, a * l + b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x, s in self.X:
            plt.plot(x[1], x[2], cols[s] + 'o')
        if mispts:
            for x, s in mispts:
                plt.plot(x[1], x[2], cols[s] + '.')
        if vec is not None:
            aa, bb = -vec[1] / vec[2], -vec[0] / vec[2]
            plt.plot(l, aa * l + bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' % (str(len(self.X)), str(len(mispts))))
            plt.savefig('p_N%s' % (str(len(self.X))), dpi=200, bbox_inches='tight')
        # plt.show()

    def pla(self, D, save):
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

            if save:
                self._plot(vec=w)
                plt.title('N = %s, Iteration %s\n' % (str(N), str(iteration)))
                plt.savefig('p_N%s_it%s' % (str(N), str(iteration)), dpi=200, bbox_inches='tight')

        self.w = w
        return iteration


def create_animated_gif(filename, delete_files=False):
    """ creates an animated gif with all png images of the current directory
    """
    file_names = sorted((fn for fn in os.listdir('.') if fn.endswith('.png')))
    images = [Image.open(fn) for fn in file_names]
    size = (1390, 1097)
    for image in images:
        image.thumbnail(size, Image.ANTIALIAS)

    if delete_files:
        for fn in file_names:
            os.remove(fn)

    print writeGif.__doc__

    writeGif(filename, images, duration=0.4)


if __name__ == "__main__":
    try:
        D = int(sys.argv[1])
    except:
        print "** ERROR ** You should pass a number as an argument, indicating the length of the dataset you want to use"
        sys.exit(2)

    try:
        N = int(sys.argv[2])
    except:
        print "** ERROR ** You should pass a number as an argument, indicating the number of dimensions you want to use"
        sys.exit(2)

    start = time.time()
    p = Perceptron(N, D)
    save = (D == 2)
    iterations = p.pla(D, save)
    end = time.time()

    if save:
        create_animated_gif('p_D%s_N%s_%s.gif' % (str(D), str(N), str(time.time())), delete_files=True)

    print 'D = %s; N = %s; Iterations = %s; Time = %s seconds' % (str(D), str(N), str(iterations), str(end - start))
