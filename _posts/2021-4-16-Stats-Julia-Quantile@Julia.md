---
layout:     post
title:      Quantile @ Julia
subtitle:   使用Julia计算分位数
date:       2021-4-16
author:     Algebra-FUN
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - Julia
    - Statistics
    - Math
---


# Quantile in Julia

In Statistics field, when we do the hypothesis testing, we have to get the quantile of a specific distribution so we go to find its value in the table. This method is out of dated, we should embrace the modern technology and exploit it. 

Julia is a powerful programming language designed for technical computing. As it own saying that `Julia: A fresh approach to technical computing`. 

Today, we are going to teach you how to get the quantile of a specific distribution in short time with only line of code.

## Preparation

1. You should have Julia installed on your device.
2. Install the package `Distribution`
	```julia
	using Pkg
	Pkg.add("Distribution")
	```
3. Import the package `Distribution`
	```julia
	using Distribution
	```
	

## Common Distribution

First of all, we should get the `Distribution` object.

For example, if we need to use `Standard Normal Distribution`,

```julia
N = Normal(0,1)
```

then you get it.

Common Distribution you may use.

```julia
# N(μ,σ):	the Normal distribution
Normal(μ,σ)
# χ²(n):	the ChiSquare distribution
Chisq(n)
# t(n):		the T distribution
TDist(n)
```

## Calculate Quantile

Now, we discuss how to get the `Quantile` in Julia.

It is so easy as you only need to call function `quantile` to approach your goal.

```julia
quantile(d::UnivariateDistribution, q::Real)
```

Above, it is the basic grammar of the function `quantile`.

Maybe it seem abstract to you, so let see some simple example to help you understand it.

1. $u_{0.975}$:  The $0.975$ quantile of Standard Normal Distribution $N(0,1)$
	```julia
	U=Normal()
	quantile(U,0.975)
	# return 1.9599639845400576
	```
2. $t_{0.95}(15)$ :The $0.95$ quantile of $t(15)$: T Distribution with $15$ freedom
	```julia
   T₍₁₅₎=TDist(15)
   quantile(T₍₁₅₎,0.95)
   # return 1.7530503556925727
   ```
3. $\chi_{0.025}^2(24)$： The $0.025$ quantile of $\chi^2(24)$ : `ChiSquare` Distribution with $24$ freedom
	```julia
	χ²₍₂₄₎=Chisq(24)
	quantile(χ²₍₂₄₎,0.025)
	# return 12.401150217444433
	```

