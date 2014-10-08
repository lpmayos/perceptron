Perceptron Learning Algorithm
=============================

Problem 1.4 of Machine Leraning course of MIIS at UPF.


Aknowledgement
--------------

This is heavily inspired (if not copied) from [http://datasciencelab.wordpress.com/2014/01/10/machine-learning-classics-the-perceptron/](http://datasciencelab.wordpress.com/2014/01/10/machine-learning-classics-the-perceptron/).


execution
---------

Clone and execute the different files from the command line with:

* perceptron.py: two dimensional perceptron, with image generation
    - <blockquote>$ python perceptron.py 20</blockquote>

* perceptron_no_images.py: two dimensional perceptron, without image generation
    - <blockquote>$ python perceptron_no_images.py 20</blockquote>

* perceptron_d_dimensional.py: d-dimensional perceptron, with image generation for d=2
    - <blockquote>$ python perceptron_d_dimensional.py 2 20</blockquote>


requirements
------------

Install (with pip) Pillow and images2gif


important
---------

As explained here: [http://stackoverflow.com/questions/19149643/error-in-images2gif-py-with-globalpalette](http://stackoverflow.com/questions/19149643/error-in-images2gif-py-with-globalpalette) I had to modify the line 426 of images2gif.py fot this code to work.