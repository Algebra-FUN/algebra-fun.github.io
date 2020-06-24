---
layout:     post
title:      布朗-SIR-元胞自动机
subtitle:   Plague-Model-Simulation
date:       2020-2-11
author:     Algebra-FUN
header-img: img/post-bg-2020-2-11.jpg
catalog: true
tags:
    - Math
    - Modeling
    - Simulation
---

# 布朗-SIR-元胞自动机

> 项目地址：https://github.com/Algebra-FUN/B-CA-Infection-Simulation

## 关于

这是一种传染病模型，基于SIR，元胞自动机（CA）原理

## 实验效果

![](https://github.com/Algebra-FUN/B-CA-Infection-Simulation/tree/master/img/50days.gif?raw=true)

![](https://github.com/Algebra-FUN/B-CA-Infection-Simulation/tree/master/img/100days.gif?raw=true)

## 模型原理

### SIR

参考SIR模型，将人群分为3个组成部分

* S(Susceptible)-易感人群
* I(Infected)-感染人群
* R(Removed)-移除人群

##### 性质
1. 感染者（I）具有感染易感者（S）的能力
2. 感染者（I）具有转化为移除者（R）的趋势
3. 移除者（R）将不再具备传染能力

### 布朗-元胞自动机（B-CA）

基于元胞自动机（CA）模型，增加元胞（Cell）在空间内做布朗运动，以模拟人口区域范围流动性

### 元胞空间

#### 参数

##### 布朗运动速度（v）

* 代表布朗运动强度
* 反应`人口流动强度`

##### 空间大小（D）

* 代表元胞所在空间的尺度
* 反应`人口密度`

### 运行机制

分为3个步骤：

1. move-元胞运动
2. infect-传染
3. remove-移除

#### move-元胞运动

$$
(\frac{dx}{dt})^2 + (\frac{dy}{dt})^2 = v
$$

$$
\theta = random[0,2\pi)
$$

#### infect-传染

感染者（I）感染易感者（S）的过程

感染概率$$\alpha$$与感染者（I）和易感者（S）之间的距离$$d$$成正比

第$$i$$个易感者$$S_i$$被第$$j$$个感染者$$I_j$$感染概率$$\alpha$$可表示为

$$
\alpha(S_i,I_j)= \large{e^{-\kappa\cdot d(S_i,I_j)}}
$$

$$
\kappa>0,d\geq0 \Rightarrow -\kappa\cdot d\leq0\Rightarrow 0<e^{-\kappa\cdot d(S_i,I_j)} \leq1\Rightarrow \alpha\in(0,1]
$$

其中$$\kappa$$为传染概率系数，$$d(S_i,I_j)$$表示第$$i$$个易感者$$S_i$$被第$$j$$个感染者$$I_j$$之间的距离

则第$$i$$个易感者$$S_i$$被感染的概率为

$$
\alpha(S_i)=1-\prod_j\left(1-\alpha(S_i,I_j)\right)
$$

#### remove-移除

每天感染者（I）的被移除概率$$\beta$$为一常数

则感染者在被感染后第$$k$$天被移除的概率$$\gamma(k)$$为

$$
\gamma(k)=\beta(1-\beta)^{k-1}
$$

## Python Dependency

1. numpy
2. pandas
3. matplotlib
4. imageio


