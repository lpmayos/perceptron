import numpy
import matplotlib.pyplot as plt
from images2gif import writeGif
from PIL import Image
import os
import time


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

    def plot(self, mispts=None, vec=None, save=False):
        """
        """
        plt.figure()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        V = self.V
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
 
    def pla(self, save=False):
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
            if save:
                self.plot(vec=w)
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
    start = time.time()
    N = 20
    p = Perceptron(N)
    iterations = p.pla(save=True)
    end = time.time()
    print 'N = %s; Iterations = %s; Time = %s seconds' % (str(N), str(iterations), str(end - start))

    create_animated_gif('p_N%s_%s.gif' % (str(N), str(time.time())), delete_files=True)
