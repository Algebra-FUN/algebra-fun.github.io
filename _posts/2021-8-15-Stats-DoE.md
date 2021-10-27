---
layout:     post
title:      Research on DoE model
subtitle:   DoE 模型的研究
date:       2021-8-15
author:     Algebra-FUN
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - Statistics
    - Math
---



# Idea of enhancement of `DoE` model

## DoE derivation

### Preset

Given $$n$$ factors $$X_k$$ and a response $$\widetilde{Y}$$,

where $$X_k$$ satisfy `DoE` layout.

First of all, we transform $$X_k$$ to $$\widetilde{X_k}$$

$$
\widetilde{X_k} = 2\frac{X_k-\bar X_k}{X_k^H-X_k^L}
$$

where $$X_k^H,X_k^L$$ are high level and low level of factor $$X_k$$ and $$\bar X_k=\frac{X_k^H+X_k^L}2$$ is the middle level(mid point)

In the `DoE`, $$\widetilde{X_k}=-1,1$$, 

Thus here is an advantage of `DoE`

$$Cov(X_i,X_j)=0$$, which means $$X_i,X_j$$ are independent.

### The model

$$
\widetilde{Y} =  \sum_k^n\widetilde{\alpha_k}\widetilde{X_k}+\sum_{s<t}^n\widetilde{\beta_{st}}\widetilde{X_s}\widetilde{X_t}+\sum_k^n\widetilde{\gamma_k}\widetilde{X_k}^2+\epsilon
$$

where $$\epsilon^{(k)} \sim N(0,\sigma^2),Var(\epsilon^{(i)},\epsilon^{(j)})=0$$

### Estimate $$\widetilde{\alpha_k}$$

We mainly discuss factors $$\widetilde{X_u}$$

$$
\widetilde{Y} =  \epsilon+\widetilde{\alpha_u}\widetilde{X_u}+\widetilde{\gamma_u}\widetilde{X_u}^2+\chi_u
$$

where,

$$
\chi_u=\sum_{k\neq u}^n\widetilde{\alpha_k}\widetilde{X_k}+\sum_{s<t}^n\widetilde{\beta_{st}}\widetilde{X_s}\widetilde{X_t}+\sum_{k\neq u}^n\widetilde{\gamma_k}\widetilde{X_k}^2
$$

$$
\begin{align}
\sum \widetilde{Y}_{\widetilde{X_u}=1}
&:=\sum_{s_i} \widetilde{Y}^{(s_i)} \\
&= \sum_{s_i}\epsilon^{(s_i)}+\frac m2\widetilde{\alpha_u}+\frac m2\widetilde{\gamma_u}+\sum_{s_i}\chi_u^{(s_i)} \\
&=\sum_{s_i}\epsilon^{(s_i)}+\frac m2\widetilde{\alpha_u}+\frac m2\widetilde{\gamma_u}
\end{align}
$$

where $$s_i \in \{s_i|\widetilde{X_u}^{(s_i)}=1\}$$

same,

$$
\begin{align}
\sum \widetilde{Y}_{\widetilde{X_u}=-1}
&:=\sum_{s_j} \widetilde{Y}^{(s_j)} \\
&= \sum_{s_j}\epsilon^{(s_j)}-\frac m2\widetilde{\alpha_u}+\frac m2\widetilde{\gamma_u}+\sum_{s_j}\chi_u^{(s_j)} \\
&=\sum_{s_j}\epsilon^{(s_j)}-\frac m2\widetilde{\alpha_u}+\frac m2\widetilde{\gamma_u}
\end{align}
$$

where $$s_j \in \{s_j|\widetilde{X_u}^{(s_j)}=-1\}$$

$$
\sum_{s_i} \widetilde{Y}^{(s_i)} - \sum_{s_j} \widetilde{Y}^{(s_j)} = m\widetilde{\alpha_u} + \sum_s^m \epsilon^{(s)}
$$

$$
\sum \widetilde{Y}_{\widetilde{X_u}=1} - \sum \widetilde{Y}_{\widetilde{X_u}=-1} = m\widetilde{\alpha_u} + \sum_s^m \epsilon^{(s)}\\
\widetilde{\alpha_u}=\frac{\sum \widetilde{Y}_{\widetilde{X_u}=1} - \sum \widetilde{Y}_{\widetilde{X_u}=-1}}m+\frac1m\sum_s^m \epsilon^{(s)}
$$

$$\widetilde{\alpha_u}$$ 's $$1-\alpha$$ confident interval:

$$
\widetilde{\alpha_u}=\left[\frac{\sum \widetilde{Y}_{\widetilde{X_u}=1} - \sum \widetilde{Y}_{\widetilde{X_u}=-1}}m \pm\frac{\sigma}mu_{1-\frac{\alpha}2}\right]
$$

where $$u_{1-\frac{\alpha}2}$$ is $$1-\frac{\alpha}2$$ quantile number of standard normal distribution.

This coefficient  $$\widetilde{\alpha_k}$$ is called the main effect of $$\widetilde{X_u}$$

> Attention that: $$\widetilde{\alpha_k} \neq 0 \nLeftrightarrow \alpha_k \neq 0$$ 

### Estimate $$\widetilde{\beta_{uv}}$$

We mainly discuss factors $$\widetilde{X_u},\widetilde{X_v}$$

The derivation progress is same as above one.

$$\widetilde{\beta_{uv}}$$ 's $$1-\alpha$$ confident interval:

$$
\widetilde{\beta_{uv}}=\left[\frac{\sum \widetilde{Y}_{\widetilde{X_u}\widetilde{X_v}=1} - \sum \widetilde{Y}_{\widetilde{X_u}\widetilde{X_v}=-1}}m \pm\frac{\sigma}mu_{1-\frac{\alpha}2}\right]
$$

where $$u_{1-\frac{\alpha}2}$$ is $$1-\frac{\alpha}2$$ quantile number of standard normal distribution.

### Estimate $$\widetilde{\gamma_k}$$

Since in high level and low level points, $$\widetilde{X_k}^2=1$$, we can't estimate $$\widetilde{\gamma_k}$$

$$
\sum_s^m \widetilde{Y}^{(s)} = m\sum_k^n\widetilde{\gamma_k}+\sum_s^m\epsilon^{(s)}
$$

we can only get

$$
\sum_k^n\widetilde{\gamma_k}=\frac1m\sum_s^m \widetilde{Y}^{(s)}+\frac1m\sum_s^m\epsilon^{(s)}
$$

without separation.

How to solve this problem? We use center points to estimate $$\epsilon$$. So, can we modify the center points to both estimate $$\epsilon$$ and $$\widetilde{\gamma_k}$$?

The Answer is Yes. Here is my approach, which is the key difference from the classical-DoE approach.

In classical-DoE approach, we use center points to estimate $$\epsilon$$, which seems some kind of waste, since we can estimate it in linear regression progress after choosing features of model.

Thus, we should modify the center point in order to estimate $$\widetilde{\gamma_k}$$ at the same time.

Experiment $$C_i:(\widetilde{X_1},\widetilde{X_2},\cdots,\widetilde{X_n},\widetilde{Y}^{(i)})$$, $$\widetilde{X_k}=\delta,\forall k\neq i,\widetilde{X_i}=0,0<\delta<1$$

Since we have estimated $$\widetilde{\alpha_k}$$,$$\widetilde{\beta_{uv}}$$, we can calculate the value:

$$
\delta^2\sum_{k\neq i}^n\widetilde{\gamma_k}=c_i+\epsilon^{(i)}
$$

$$
\widetilde{\gamma_k}=\frac1m\sum_s^m \widetilde{Y}^{(s)}-\frac1{\delta^2}c_i+\frac1m\sum_s^m\epsilon^{(s)}+\frac1{\delta^2}\epsilon^{(i)}
$$

where $$c_i$$

$$
\widetilde{Y}^{(i)} =  \delta\sum_{k\neq i}^n\widetilde{\alpha_k}+\delta^2\sum_{s<t\neq i}^n\widetilde{\beta_{st}}+\delta^2\sum_{k\neq i}^n\widetilde{\gamma_k}+\epsilon^{(i)}
$$

$$
c_i:=\widetilde{Y}^{(i)}- \delta\sum_{k\neq i}^n\widetilde{\alpha_k}-\delta^2\sum_{s<t\neq i}^n\widetilde{\beta_{st}}
$$



## The Model

If we have $$n$$ features

$$
y=x^T\Beta x+\epsilon
$$

where $$x \in \mathbb{R}^{n+1}, \Beta\in\mathbb{R}^{(n+1)\times (n+1)},y\in\mathbb{R}$$

$$x=[x_0,x_1,x_2,\cdots,x_n]^T=[1,x_1,x_2,\cdots,x_n]^T$$, where $$x_0\equiv1$$ is called `dumb input` and $$x_i$$ is the $$i$$ th `factor`

$$\Beta$$ is the parametric matrix and it is a symmetric matrix($$\Beta^T=\Beta$$), $$y$$ is the `response`(single)

$$\epsilon$$ is a random variable for depicting the random error in the experiment.

## Estimation of $$\Beta $$

### The loss function

If we have $$m$$ samples, $$(x^{(i)},y^{(i)})$$ indicates for the $$i$$ th sample within the dataset.

To define the loss function, we use `MSE` loss function here.

$$
l(\Beta)=\sum_{i=1}^n(y^{(i)}-{x^{(i)}}^T\Beta x^{(i)})^2
$$

### Minimize the loss function to find the optimal solution

First of all, calculate the differential:

$$
\begin{align}
\frac{\part l}{\part B}&=2\sum_{i=1}^n({x^{(i)}}^T\Beta x^{(i)}-y^{(i)})(x^{(i)}{x^{(i)}}^T)\\
&=2\sum_{i=1}^n({x^{(i)}}^T\Beta x^{(i)})(x^{(i)}{x^{(i)}}^T)-2\sum_{i=1}^ny^{(i)}(x^{(i)}{x^{(i)}}^T)\\
&=2\sum_{i=1}^nx^{(i)}({x^{(i)}}^T\Beta x^{(i)}){x^{(i)}}^T-2\sum_{i=1}^ny^{(i)}(x^{(i)}{x^{(i)}}^T)\\
&=2\sum_{i=1}^n(x^{(i)}{x^{(i)}}^T)\Beta (x^{(i)}{x^{(i)}}^T)-2\sum_{i=1}^ny^{(i)}(x^{(i)}{x^{(i)}}^T)\\
\end{align}
$$

Let us denote:

$$
A_i:=x^{(i)}{x^{(i)}}^T,C:=\sum_{i=1}^ny^{(i)}A_i
$$

and $$A_i,C \in \mathbb{R}^{(n+1)\times (n+1)}$$ are all symmetric matrix, which means $${A_i}^T=A_i,C^T=C$$ 

thus,

$$
\frac{\part l}{\part B}=2\left(\sum_{i=1}^nA_i\Beta A_i-C\right)
$$

If let $$\frac{\part l}{\part B} = 0$$, then

$$
\sum_{i=1}^nA_i\Beta A_i=C
$$

when $$m\geq\frac{(n+1)^2+(n+1)}2$$ and

$$
r\left[\sum_{i=1}^m(A_i\otimes A_i)\right]=\frac{(n+1)^2+(n+1)}2
$$

we can get the unique solution of symmetric matrix $$\Beta $$.

$$
\sum_{i=1}^nA_i\Beta A_i=C \Leftrightarrow \left[\sum_{i=1}^m(A_i\otimes A_i)\right]b=\gamma
$$

where 

$$
b=
\begin{bmatrix}
col_1 \Beta\\
col_2 \Beta\\
\vdots\\
col_{n+1}\Beta
\end{bmatrix},
\gamma=
\begin{bmatrix}
col_1 C\\
col_2 C\\
\vdots\\
col_{n+1}C
\end{bmatrix}
$$

$$
\hat\Beta=
\begin{bmatrix}
col_1 \Beta&col_2 \Beta &\cdots &col_{n+1}\Beta
\end{bmatrix}
\Leftrightarrow \hat b=
\begin{bmatrix}
col_1 \Beta\\
col_2 \Beta\\
\vdots\\
col_{n+1}\Beta
\end{bmatrix}
$$

So, we can get the optimal solution $$\hat\Beta$$ and the prediction of  response $$\hat y$$ will be:

$$
\hat y = x^T\hat\Beta x
$$

## The forward error analysis

Since we know, in the production environment, the factor we set can have some random error which we can't control precisely or eliminate completely. So we should think about the influence of the error of factor.

Denote $$\Delta x \sim N(0,diag(\sigma_0^2,\sigma_1^2,\cdots,\sigma_{n+1}^2))$$ as $$n+1$$ dimensional random vector. In conventional linear model:

$$
y=x^T\beta+\epsilon
$$

We can modify it to this form:

$$
y+\Delta y=(x+\Delta x)^T\beta
$$

where we consider $$\epsilon=\beta_0{\Delta x}_0$$,

$$
y+\Delta y=x^T\beta+{\Delta x}^T\beta\\
\Rightarrow\Delta y=\sum_{i=1}^n\beta_i{\Delta x}_i
$$

$$
\Delta y\sim N(0,\sum_{i=1}^n \beta_i^2\sigma_i^2)
$$

So, in the estimation of $$\sigma^2$$ of $$\epsilon$$, we count the error of input into $$\epsilon$$, which have some influence on the precise of $$\beta$$ estimation. However, this influence in the linear model is not significant since the model is linear, the error of input is only magnify $$\beta$$ which is a constant.

But in our `DoE` model, we should consider about the effect of `interaction` and `curvature` of factors. The model is actually a `quadratic polynomial regression`, the influence in this model won't be linear. So, let us see what will happen.

We have the `DoE` model, we write it in Matrix form:

$$
y=x^T\Beta x+\epsilon
$$

When we add the error of input:

$$
y+\Delta y=(x+\Delta x)^T\Beta(x+\Delta x)
$$

Since $${\Delta x}_0$$ can not affect other real factors,

Denote $$x=[x_0,\widetilde x],\Delta x=[{\Delta x}_0,\Delta \widetilde x]$$,

$$
\Beta=\begin{bmatrix}
b&{\widetilde \beta}^T\\
\widetilde \beta&\widetilde \Beta
\end{bmatrix}
$$

$$
y=b+{\widetilde x}^T\widetilde \beta+{\widetilde x}^T\widetilde \Beta\widetilde x +\epsilon
$$

$$
\Delta y={\Delta\widetilde x}^T\widetilde \beta+2{\widetilde x}^T\widetilde \Beta\Delta\widetilde x+{\Delta\widetilde x}^T\widetilde \Beta\Delta\widetilde x
$$

Since $$||\Delta\widetilde x||^2$$ is too small, so

$$
\Delta y\approx{\Delta\widetilde x}^T\widetilde \beta+2{\widetilde x}^T\widetilde \Beta\Delta\widetilde x
$$

In the next step, we should estimate the variance of $$\Delta y$$

$$
{\Delta\widetilde x}^T\widetilde \beta \sim N(0,\sum_{i=1}^n {\Beta_{0i}}^2\sigma_i^2)
$$

$$
{\widetilde x}^T\widetilde \Beta\Delta\widetilde x \sim N(0,\sum_{i\leq j}{\Beta_{ij}}^2x_i^2\sigma_j^2)
$$

So,

$$
Var(\Delta y)\approx\sum_{i=1}^n {\Beta_{0i}}^2\sigma_i^2+\sum_{i\leq j}{\Beta_{ij}}^2x_i^2\sigma_j^2
$$

As you can see, when the factor $$x_i$$ becomes large, then the variance of $$\Delta y$$ will be magnified to be large too.

## Features Standardization

For the Auto-Quad-DoE Model, we can standardize the feature, the model can be expressed in following formula:

$$
y = \beta_0 + \sum_{i=1}^n \beta_i x_i + \epsilon
$$

where $$\epsilon \sim N(0,\sigma^2)$$

Then we standardize the input features $$x_i$$ and the output response $$y$$

$$
x_i^* = \frac{x_i-\mu_i}{\sigma_i},y^* = \frac{y-\mu_y}{\sigma_y}
$$

where $$\mu_i$$ is the mean of feature $$x_i$$, $$\sigma_i $$ is the std of feature $$x_i$$

$$
y = \beta_0 + \sum_{i=1}^n \beta_i x_i + \epsilon
$$

$$
\sigma_y y^* + \mu_y = \beta_0 + \sum_{i=1}^n (\beta_i \sigma_i) x_i^* + \sum_{i=1}^n\beta_i \mu_i + \epsilon
$$

$$
\sigma_y y^* = \sum_{i=1}^n (\beta_i \sigma_i) x_i^* + \left(\beta_0 + \sum_{i=1}^n\beta_i \mu_i  - \mu_y \right) +  \epsilon
$$

$$
y^* = \sum_{i=1}^n \frac{\beta_i \sigma_i}{\sigma_y} x_i^* + \frac{\beta_0 + \sum_{i=1}^n\beta_i \mu_i  - \mu_y}{{\sigma_y}} +  \epsilon^*
$$

$$
\Rightarrow \beta_i^* = \frac{\sigma_i}{\sigma_y}\beta_i,\epsilon^* \sim N(0,\frac{\sigma^2}{\sigma_y^2}),\beta_0^* =\frac{\beta_0 + \sum_{i=1}^n\beta_i \mu_i  - \mu_y}{{\sigma_y}}
$$

$$
y^*=\beta_0^*+\sum_{i=1}^n\beta_i^*x_i^*+\epsilon^*
$$

Why we need to calculate $$\beta_i^*$$ ? When we make `coef-plot`, since each $$x_i$$ has different dimension size which has greatly impact on $$\beta_i$$. For example, a feature $$x_a$$ may be big ($$>10^3$$), the corresponding coefficient $$\beta_a$$ will be small ($$<10^{-3}$$) and another feature $$x_b$$ is small($$<10^{-3}$$), the corresponding coefficient $$\beta_b$$ will be big ($$>10^3$$). Thus when we see the `coef-plot` without standardization, you can see that $$\beta_b>>\beta_a$$ and assert that feature $$x_b$$ is more important than feature $$x_a$$. However, the truth isn't it. If $$x_a^*\approx x_b^*$$, which means the $$x_a$$ is as important as $$x_b$$ approximately. Hence, when we want to make a `coef-plot`, we should calculate the $$\beta_i^* $$ after standardized. 

