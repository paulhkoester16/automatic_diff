# automatic_diff

This package and series of lessons arose out of my own curiosity of what automatic differentiation means, 
and how one might implement it.  

It is not intended to be used for scalable or production grade deep learning.  For any serious application, one 
should use tools like `Tensorflow` or `Pytorch`, which provide much better implementations.   

Also, the approach I take here is admittedly naive.  I approached it from the point of view of combining freshman 
calculus with commutative ring theory.  (And as I'm writing this README, I'm finally making sense of the motivation
for all that exterior algebra stuff from my graduate math days.)

The ring theoretic approach suggests itself from the observation that the product rule arises from multiplication 
of formal polynomials.  In other words, consider polynomial expressions in the formal variable $dx$:
$$
    a_0 + a_1 dx
$$
and
$$
    b_0 + b_1 dx
$$
Then multiplying these are formal polynomials in $dx$, we get
$$
    a_0 b_0 + (a_1 b_0 + a_0 b_1) dx + a_1 b_1 (dx)^2 
$$
Now if we map a function, derivative pair $(f(x), f'(x))$ to the formal expression $f(x) + f'(x) dx$, and if we mod out $(dx)^2 = 0$, we get 
$$
  (f(x), f'(x)) \cdot (g(x), g'(x)) = (f(x)g(x), f'(x)g(x) + f(x)g'(x))
$$

The ring theoretic approach also gave me an excuse to more deeply explore operator overloading in python.  I had been aware 
that one could overload the `+`, `*`, `/`, `**` operators for your own custom classes, but I had never had a practical excuse
for doing so.  


After having made some headway, I decided to look at the literature (`https://en.wikipedia.org/wiki/Automatic_differentiation`) 
to see how the experts think about automatic differentiation.  Surprisingly, I got a lot of it right.

However, the approach I took turns out to follow the *forward accumulation* paradigm, which turns out to be grossly inefficient for the 
standard case of a scalar valued function of many variables. 

My approach is also grossly inefficient because I have made no attempt to cache intermediate values.  For example, 
$$
    h(x, y) = \cos{(xy)}
$$
would be treated as a composition of 
$$
    f(z) = \cos{(z)}
$$
and 
$$
    g(x, y) = x \cdot y 
$$
giving the gradient
$$
    \nabla h = [\frac{\partial f}{\partial z} \frac{\partial h}{\partial x}, \frac{\partial f}{\partial z} \frac{\partial h}{\partial x}]
$$
My implementation requires 4 derivative evaluations, making no attempt to exploit the fact that $\frac{\partial f}{\partial z}$ could be
computed only once. 


