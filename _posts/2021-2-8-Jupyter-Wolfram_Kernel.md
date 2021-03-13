---
layout:     post
title:      Wolfram with Jupyter
subtitle:   在Jupyter中使用Wolfram
date:       2021-2-8
author:     Algebra-FUN
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - Wolfram
    - IDE
    - Jupyter
---

# Wolfram with Jupyter

> Use Mathematica in Jupyter  Lab

## What do we need

* [Jupyter Lab](https://jupyter.org/install.html)
* [Wolfram Engine](https://www.wolfram.com/engine/)
* [Wolfram Kernel](https://github.com/WolframResearch/WolframLanguageForJupyter)

> I think the reader should have the experience about using Jupyter Notebook or Lab, so I won't introduce how to install Jupyter Lab Here.

## Wolfram Engine

Wolfram Engine is Free and Strong. It's is similar to Python Interpreter. As consequence, we can use a kernel to transfer it to Jupyter.

Download it in this link: [https://brilliant.org/wiki/beta-function/](https://www.wolfram.com/engine/)

Download and install and activate it, as the web page follow.

It's easy to install.

## Wolfram Kernel

### Clone the repository

```shell
git clone https://github.com/WolframResearch/WolframLanguageForJupyter.git
```

### Install the kernel

Run the following command in your shell to make the Wolfram Language engine available to Jupyter:

```shell
./configure-jupyter.wls add
```

**Attention here!**: the command won't work on windows, you should use `cd` command first

For windows user:

```shell
cd WolframLanguageForJupyter
configure-jupyter.wls add
```

Waiting for installing...

However, maybe there will raise a `network error`, so you can try in other way to install following the [`Method 2: Using Wolfram Language`](https://github.com/WolframResearch/WolframLanguageForJupyter#method-2-using-wolfram-language)

## Checking

See if the kernel is installed.

```shell
jupyter kernelspec list
```

And also you can start Jupyter Lab to see the performance.
