---
layout:     post
title:      Julia with Jupyter
subtitle:   在Jupyter中使用Julia
date:       2021-3-27
author:     Algebra-FUN
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - Julia
    - IDE
    - Jupyter
---

# Julia with Jupyter

> Use Julia in Jupyter  Lab

## What do we need

* [Jupyter Lab](https://jupyter.org/install.html)
* IJulia

> I think the reader should have the experience about using Jupyter Notebook or Lab, so I won't introduce how to install Jupyter Lab Here.

## Install IJuila

```julia
using Pkg
Pkg.add("IJulia")
```

This process will install julia kernel to jupyter automatically

> We don't recommend cmd install pkg

## Checking

See if the kernel is installed.

```shell
jupyter kernelspec list
```

If you can see julia here, indicating it has been installed successfully. And also you can start Jupyter Lab to see the performance.

### If failed

However, may be you can see the julia kernel in the list, you can use following script to solve this problem manually.

```Julia
using IJulia
installkernel("Julia", "--depwarn=no")
```

And check again.

## 