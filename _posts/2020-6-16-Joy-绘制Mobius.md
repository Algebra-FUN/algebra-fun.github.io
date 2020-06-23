---
layout:     post
title:      Draw a Mobius Strip
subtitle:   在Python中绘制莫比乌斯环
date:       2020-6-16
author:     Algebra-FUN
header-img: img/post-bg-2020-6-16.jpg
catalog: true
tags:
    - Math
    - Joy
    - Topology
---

# Draw a Mobius Strip with Python

First of all, we need to figure out the Math Analytic Expression of the Mobius Strip.

## Math Analytic Expression
You can easily get this expression if you have a good command of **Analytic Geometry**.

$$
\begin{cases}
x=(r+\omega\cos(\beta))\cos(\theta)\\
y=(r+\omega\cos(\beta))\sin(\theta)\\
z=\omega\sin(\beta)
\end{cases}
s.t.
\begin{cases}
\theta\in[0,2\pi]\\
\omega\in[-\frac W2,\frac W2]
\end{cases}
$$

> If you don't know how to get the expression, you can search on google.

When it is classic Mobius strip, then $$\beta=\frac{\theta}2$$.

Substitute the $$\beta$$ in the expression, then you can get this expression:

$$
\begin{cases}
x=(r+\omega\cos(\frac{\theta}2))\cos(\theta)\\
y=(r+\omega\cos(\frac{\theta}2))\sin(\theta)\\
z=\omega\sin(\frac{\theta}2)
\end{cases}
s.t.
\begin{cases}
\theta\in[0,2\pi]\\
\omega\in[-\frac W2,\frac W2]
\end{cases}
$$

## Draw with Matplotlib

We are going to draw the Mobius Strip with Numpy and Matplotlib.

* import the dependencies and set the parameters.

```python
import numpy as np
from numpy import pi,cos,sin
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

R,W=10,10
a,b=24,20
```

* define the Analytic Expression.

```python
def x(u,v):
    return (R+v*cos(u/2))*cos(u)
def y(u,v):
    return (R+v*cos(u/2))*sin(u)
def z(u,v):
    return v*sin(u/2)
```

> where u,v represent of $$\theta$$ and $$\omega$$

* calculate the Meshgrid of the graph.

```python
u_range = np.linspace(0,2*pi,a)
v_range = np.linspace(-W/2,W/2,b)

uv_meshgrid = np.meshgrid(u_range,v_range) # build the meshgrid of parameter

X,Y,Z = map(lambda f: f(*uv_meshgrid),(x,y,z)) # calculate the meshgrid of the mobius strip
```

* draw with matplotlib.

```python
fig = plt.figure() 
ax = fig.gca(projection='3d') # instance of 3d canvas

ax.plot_wireframe(X,Y,Z) # plot the wireframe of the mobius strip

plt.axis('off')
ax.set_zlim(-.8*W,.8*W)
plt.show()
```

the output:

![](https://github.com/Algebra-FUN/My-Math-Model/blob/master/Mobius/Figure_1.png?raw=true)

## Draw with Sympy

We are going to draw the Mobius Strip with Sympy.

> **Sympy** is a **Powerful** Python library for **symbolic** mathematics.

Let's coding

```python
from sympy import *
from sympy.abc import r, theta, beta, omega
from math import pi
from sympy.plotting import plot3d_parametric_surface

x = (r+omega*cos(beta))*cos(theta)
y = (r+omega*cos(beta))*sin(theta)
z = omega*sin(beta)

R, W = 10, 10

x, y, z = map(lambda expr: expr.subs(
    [(beta, theta/2), (r, R)]), (x, y, z))

plot3d_parametric_surface(x, y, z,
                          (theta, 0, 2*pi), (omega, -W/2, W/2))
```

the output:

![](https://github.com/Algebra-FUN/My-Math-Model/blob/master/Mobius/Figure_2.png?raw=true)

> You see the code with sympy is more compact and symbolic than using mumpy and matplotlib

## For FUN

You can try to modify the expression of $$\beta=\frac{\theta}2$$.

When  $$\beta=\theta$$

the output will be:

![](https://github.com/Algebra-FUN/My-Math-Model/blob/master/Mobius/Figure_3.png?raw=true)