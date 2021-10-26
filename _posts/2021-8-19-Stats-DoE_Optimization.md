---
layout:     post
title:      The best parameter choosing of DoE under specific constrains
subtitle:   DoE 模型在特定情况下的最优参数选择
date:       2021-8-15
author:     Algebra-FUN
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - Statistics
    - Math
---



# The best parameter choosing of DoE under specific constrains.

## Problem statement

Given factors $$\{x_i\}_{i=1}^n$$ and response $$\{y_j\}_{j=1}^s$$, then we use given dataset to get a serise of models

$$
y_j=f_j(x_1,x_2,\cdots,x_n)+\epsilon_j
$$

where $$\epsilon_j\sim N(0,\sigma_j^2)$$ and $$\forall p\neq q,Var(\epsilon_p,\epsilon_q)=0$$, which means those random variables are independent.

> This isn't a common assumptions. They may not be independent.

According the practical production requirement, the responses $$y_j$$ have their own restrictions

Constraints:

$$
y_j^{l}\leq y_j\leq y_j^{h}
$$

Thus, the problem is that what parameters $$x_i$$ can satisfy the constrains of $$y_j $$? And how to find the best parameters $$x_i^*$$? 

## Problem transform

Before we solve this problem, first we should considerate one real situation and one sub-problem of this problem.

**The one real situation:**

In practice, most of factors $$x_i$$ which we can't control precisely, they have some random error. Thus, we should considerate the random error of input.

Let, $$x_i\sim N(\hat{x_i},\delta_i^2)$$and $$\forall p\neq q,Var(x_p,x_q)=0$$ , where $$\hat{x_i}$$ is the wanted input of the factor $$x_i$$

**The one sub-problem**:

The sub-problem is that how can we measure the "betterness" of parameters $$x_i$$ ?

This sub-problem is very important, which we should address at the begining.

Since, we considerate the random errors of responses and inputs of factors, we can measure the "betterness" in probability perspective.

$$
P(y_1^{l}\leq y_1\leq y_1^{h},y_2^{l}\leq y_2\leq y_2^{h},\cdots,y_n^{l}\leq y_n\leq y_n^{h})
$$

which is the probability that all response $$y_i$$ got in constraints, we use this probability to measure the "betterness" of parameter $$x_i$$ and after that how to find the optimal solution $$x_i^*$$ is the problem of optimization.

In this way, our problem becomes how to calculate $$P(y_1^{l}\leq y_1\leq y_1^{h},y_2^{l}\leq y_2\leq y_2^{h},\cdots,y_n^{l}\leq y_n\leq y_n^{h})$$ and how to find to the optimal solution of this?

## Approach of the problem

### Calculate the probability

First, we considerate only one response situation.

#### Single response situation

Model:

$$
y=f(x_1,x_2,\cdots,x_n)+\epsilon,\epsilon \sim N(0,\sigma^2)
$$

Constraints:

$$
y^{l}\leq y\leq y^{h}
$$

Inputs of factors:

$$
x_i\sim N(\hat{x_i},\delta_i^2),\forall p\neq q,Var(x_p,x_q)=0
$$

According to the assumption,

$$
y\sim N(f(x_1,x_2,\cdots,x_n),\sigma^2)
$$

$$
p(y,x_1,x_2,\cdots,x_n)=p(y|x_1,x_2,\cdots,x_n)p(x_1,x_2,\cdots,x_n)
$$

Since $$x_i$$ are independent to each others,

$$
p(x_1,x_2,\cdots,x_n)=\prod_{i=1}^np(x_i)
$$

Thus,

$$
p(y,x_1,x_2,\cdots,x_n)=p(y|x_1,x_2,\cdots,x_n)\prod_{i=1}^np(x_i)
$$

$$
p(y)=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\cdots\int_{-\infty}^{\infty}p(y|x_1,x_2,\cdots,x_n)\prod_{i=1}^np(x_i)dx_1dx_2\cdots dx_n
$$

Since,

$$
y\sim N(f(x_1,x_2,\cdots,x_n),\sigma^2),x_i\sim N(\hat{x_i},\delta_i^2)
$$
Thus,

$$
\frac{y-f(x_1,x_2,\cdots,x_n)}{\sigma}\sim N(0,1),\frac{x_i-\hat{x_i}}{\delta_i}\sim N(0,1)
$$

$$
p(y|x_1,x_2,\cdots,x_n)=\frac1{\sigma}\phi\left(\frac{y-f(x_1,x_2,\cdots,x_n)}{\sigma}\right),p(x_i)=\frac1{\delta_i}\phi\left(\frac{x_i-\hat{x_i}}{\delta_i}\right)
$$

where $$\phi(t)$$ is the density function of standard normal distribution.

Thus,

$$
p(y)=\left(\sigma\prod_{i=1}^n\delta_i\right)^{-1}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\cdots\int_{-\infty}^{\infty}\phi\left(\frac{y-f(x_1,x_2,\cdots,x_n)}{\sigma}\right)\prod_{i=1}^n\phi\left(\frac{x_i-\hat{x_i}}{\delta_i}\right)dx_1dx_2\cdots dx_n
$$

Thus,

$$
P(y^l\leq y\leq y^h)
=\int_{y^l}^{y^h}p(y)dy\\
=\left(\sigma\prod_{i=1}^n\delta_i\right)^{-1}\int_{y^l}^{y^h}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\cdots\int_{-\infty}^{\infty}\phi\left(\frac{y-f(x_1,x_2,\cdots,x_n)}{\sigma}\right)\prod_{i=1}^n\phi\left(\frac{x_i-\hat{x_i}}{\delta_i}\right)dx_1dx_2\cdots dx_ndy
$$

#### Multi-responses situations

Denote factors $$\boldsymbol x \in \mathbb{R}^n$$, responses $$\boldsymbol y \in\mathbb{R}^s$$,$$f \in C(\mathbb{R}^n\rightarrow\mathbb{R}^s)$$

Model:

$$
\boldsymbol y =f(\boldsymbol x)+\boldsymbol\epsilon,\boldsymbol\epsilon\sim N(0,\Sigma)
$$

where $$\Sigma$$ is covarience matrix of $$\boldsymbol \epsilon$$

Given input $$\boldsymbol x \sim N(\boldsymbol{\hat x},\Delta)$$
where $$\Delta$$ is covarience matrix of $$\boldsymbol x$$

Constraints:

$$
\boldsymbol{y^l}\leq\boldsymbol y\leq \boldsymbol{y^h}
$$

Denote that

$$
C^{\boldsymbol y} = \prod_{j=1}^s\left[y_j^l,y_j^h\right]
$$
According to the assumption,

$$
p(\boldsymbol y,\boldsymbol x;\boldsymbol{\hat x})=p(\boldsymbol y|\boldsymbol x)p(\boldsymbol x;\boldsymbol{\hat x})
$$

$$
\Sigma^{-1}(\boldsymbol y-f(\boldsymbol x))\sim N(0,I_s),\Delta^{-1}(\boldsymbol x-\boldsymbol{\hat x}) \sim N(0,I_n)
$$

$$
p(\boldsymbol y,\boldsymbol x)=|\Sigma|^{-\frac12}\phi_s\left(\Sigma^{-\frac12}(\boldsymbol y-f(\boldsymbol x))\right),p(\boldsymbol x;\boldsymbol{\hat x})=|\Delta|^{-\frac12}\phi_n\left(\Delta^{-\frac12}(\boldsymbol x-\boldsymbol{\hat x})\right)
$$

where $$\phi_s,\phi_n$$ are density functions of $$s,n$$ dimension standard normal distribution. 

$$
\begin{align}
p(\boldsymbol y;\boldsymbol{\hat x})
&=\int_{\mathbb{R}^n}p(\boldsymbol y,\boldsymbol x;\boldsymbol{\hat x})d\boldsymbol x\\
&=\int_{\mathbb{R}^n}p(\boldsymbol y|\boldsymbol x)p(\boldsymbol x;\boldsymbol{\hat x})d\boldsymbol x\\
\end{align}
$$

Thus,

$$
\begin{align}
P(\boldsymbol y\in C^{\boldsymbol y};\boldsymbol{\hat x})
&=\int_{C^{\boldsymbol y}}p(\boldsymbol y;\boldsymbol{\hat x})d\boldsymbol y\\
&=\int_{C^{\boldsymbol y}}\int_{\mathbb{R}^n}p(\boldsymbol y|\boldsymbol x)p(\boldsymbol x;\boldsymbol{\hat x})d\boldsymbol xd\boldsymbol y\\
&=|\Sigma|^{-\frac12}|\Delta|^{-\frac12}\int_{C^{\boldsymbol y}}\int_{\mathbb{R}^n}\phi_s\left(\Sigma^{-\frac12}(\boldsymbol y-f(\boldsymbol x))\right)\phi_n\left(\Delta^{-\frac12}(\boldsymbol x-\boldsymbol{\hat x})\right)d\boldsymbol xd\boldsymbol y
\end{align}
$$

Denote

$$
I(\boldsymbol{\hat x})=P(\boldsymbol y\in C^{\boldsymbol y};\boldsymbol{\hat x})
$$

### Optimization

**Statement**:

$$
\begin{align}
&\max_{\boldsymbol{\hat x}}\quad I(\boldsymbol{\hat x})\\
&s.t.\quad\boldsymbol{\hat x}\in\mathbb{D}
\end{align}
$$

> It seems that it is some kind of difficult to solve....

## Simple situation

Assume that input $$\boldsymbol x$$ are precise.

$$
p(\boldsymbol y;\boldsymbol x)=|\Sigma|^{-\frac12}\phi_s\left(\Sigma^{-\frac12}(\boldsymbol y-f(\boldsymbol x))\right)
$$

Assume that responses $$\boldsymbol y$$ are independent,

$$
\begin{align}
P(\boldsymbol y \in C^{\boldsymbol y};\boldsymbol x)
&=\prod_{j=1}^s\int_{y_j^l}^{y_j^h}\frac1{\sigma_j}\phi\left(\frac{y_j-f_j(\boldsymbol x)}{\sigma_j}\right)dy_j\\
&=\prod_{j=1}^s\int_{y_j^l}^{y_j^h}\phi\left(\frac{y_j-f_j(\boldsymbol x)}{\sigma_j}\right)d\left(\frac{y_j-f_j(\boldsymbol x)}{\sigma_j}\right)\\
&=\prod_{j=1}^s\int_{\frac{y_j^l-f_j(\boldsymbol x)}{\sigma_j}}^{\frac{y_j^h-f_j(\boldsymbol x)}{\sigma_j}}\phi(t)dt\\
&=\prod_{j=1}^s\left[\Phi\left(\frac{y_j^h-f_j(\boldsymbol x)}{\sigma_j}\right)-\Phi\left(\frac{y_j^l-f_j(\boldsymbol x)}{\sigma_j}\right)\right]\\
\end{align}
$$

