# Automatic Differentiation

This package and series of lessons arose out of my own curiosity of what automatic differentiation means, 
and how one might implement it.  

It is not intended to be used for scalable or production grade deep learning.  For any serious application, one 
should use tools like [Tensorflow](https://www.tensorflow.org/) or [Pytorch](https://pytorch.org/), which provide much 
better implementations.   

Also, the approach I take here is admittedly naive.  I approached it from the point of view of combining freshman 
calculus with commutative ring theory. See `examples/Ring Theoretic Approach to Automatic Differentiation.ipynb`.
(And as I'm writing this README, I'm finally making sense of the motivation for all that exterior algebra stuff 
from my graduate math days.)

After having made some headway, I decided to look at the literature (`https://en.wikipedia.org/wiki/Automatic_differentiation`) 
to see how the experts think about automatic differentiation.  Surprisingly, I got a lot of it right.

However, the approach I took turns out to follow the *forward accumulation* paradigm, which turns out to be grossly inefficient for the 
standard case of a scalar valued function of many variables. 

My approach is also further inefficient because I have made no attempt to cache intermediate values of gradients.  Indeed 
(without even bothering to look at a profiler), it is very clear to me that many values are being computed multiple times, and 
that this will be a crippling blow against being able to use this at any degree of scale.  

