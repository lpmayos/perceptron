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
        # generate D-dimensional hyperplane (if D==2 it is a plane)
        self.hyperplane = self._generate_hyperplane(D)

        # generate a training set using the line
        self.X = self._generate_training_set(N, D)

    def _generate_hyperplane(self, D):
        """ generates an hyperplane of D dimensions (if D==2 it is a plane)
        """
        # coefficients A..J
        coefficients = [numpy.random.uniform(-1, 1) for i in range(D)]
        
        # we calculate last coefficient to make sure that we know one of the coordinates
        # Example: in the plane, Ax + By + Cz + D, if we assign 0.5 to all variables then D = -(A·0.5 + B·0.5 + C·0.5))
        last_coefficient = 0
        for i in range(D):
            last_coefficient += 0.5 * coefficients[i]
        coefficients.append(-last_coefficient)
        
        return numpy.array(coefficients)

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
 
    def _plot(self, vec):
        """ plots the plane, the points and the hypothesis that the algorithm is checking
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

        aa, bb = -vec[1] / vec[2], -vec[0] / vec[2]
        plt.plot(l, aa * l + bb, 'g-', lw=2)

    def pla(self, D, generate_images):
        """ Perceptron Learning Algorithm
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

            # generate this iteration image if it was so configured
            if generate_images:
                self._plot(w)
                plt.title('N = %s, Iteration %s\n' % (str(N), str(iteration)))
                plt.savefig('p_N%s_it%s' % (str(N), str(iteration)), dpi=200, bbox_inches='tight')

        self.w = w
        return iteration


def create_animated_gif(filename, keep_itermediate_images):
    """ creates an animated gif with all png images of the current directory
    """
    file_names = sorted((fn for fn in os.listdir('.') if fn.endswith('.png')))
    images = [Image.open(fn) for fn in file_names]
    size = (1390, 1097)
    for image in images:
        image.thumbnail(size, Image.ANTIALIAS)

    if not keep_itermediate_images:
        for fn in file_names:
            os.remove(fn)

    print writeGif.__doc__

    writeGif(filename, images, duration=0.4)


def parse_arguments():
    """ This module expects at least two arguments: a number D indicating the number of dimensions and a number N indication the length of the sample to use.
    If D == 2, it expects one more argument True or False indicating if we want to generate images for each iteration.
    If we choose to generate images, it expects one more argument True or False indicating of we want to generate animated gif.
    If we choose to generate animated gif, it expects one more argument indicating if we want to keep the intermediate files.
    """
    try:
        D = int(sys.argv[1])
    except:
        print "** ERROR ** You should pass a number as first argument, indicating the number of dimensions you want to use"
        sys.exit(2)

    try:
        N = int(sys.argv[2])
    except:
        print "** ERROR ** You should pass a number as second argument, indicating the length of the dataset you want to use"
        sys.exit(2)

    if D == 2:
        try:
            generate_images = sys.argv[3] == 'True'
        except:
            print "** ERROR ** You should pass True or False as third argument, indicating if you want to generate an image for each iteration"
            sys.exit(2)

        if generate_images:
            try:
                generate_animated_gif = sys.argv[4] == 'True'
            except:
                print "** ERROR ** You should pass True or False as fourth argument, indicating if you want to generate an animated gif with all the executed iterations"
                sys.exit(2)

            if generate_animated_gif:
                try:
                    keep_itermediate_images = sys.argv[5] == 'True'
                except:
                    print "** ERROR ** You should pass True or False as fifth argument, indicating if you want to keep the files generated for each iteartion or you prefer to delete them"
                    sys.exit(2)
            else:
                keep_itermediate_images = True

        else:
            keep_itermediate_images = False
            generate_animated_gif = False
    else:
        generate_images = False
        keep_itermediate_images = False
        generate_animated_gif = False

    return D, N, generate_images, generate_animated_gif, keep_itermediate_images


if __name__ == "__main__":

    # parse arguments with the desired configuration
    D, N, generate_images, generate_animated_gif, keep_itermediate_images = parse_arguments()

    # initialize perceptron an execute Perceptron Learning Algorithm
    start = time.time()
    p = Perceptron(N, D)
    iterations = p.pla(D, generate_images)
    end = time.time()

    # generate animated gif in case the user selected that option
    if generate_animated_gif:
        create_animated_gif('p_D%s_N%s_%s.gif' % (str(D), str(N), str(time.time())), keep_itermediate_images=keep_itermediate_images)

    # print results on screen
    print 'D = %s; N = %s; Iterations = %s; Time = %s seconds' % (str(D), str(N), str(iterations), str(end - start))
