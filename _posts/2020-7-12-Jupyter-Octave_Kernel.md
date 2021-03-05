---
layout:     post
title:      Jupyter + Octave = Simple Free Matlab
subtitle:   Jupyter + Octave = 简单免费的Matlab
date:       2020-7-12
author:     Algebra-FUN
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - Octave
    - IDE
    - Jupyter
---

# Jupyter + Octave = Simple Free Matlab
As we known, Matlab is a powerful Matrix computation program language with specific intergrated development environment. However Matlab is commercial software you should pay for it and too cumbersome about 10G. Naturally, we want a free and tiny software with the same power. The solution is HERE: Jupyter + Octave.

To configure this environment, you should install three component: **Jupyter**,**Octave**,**Octave Kernel**

## Jupyter 
The Jupyter Notebook is an open-source web application that allows you to create and share documents that > contain live code, equations, visualizations and narrative text.

> We highly recommend Jupyter Lab

### Install
```shell
pip install jupyter
```

### Launch
```shell
jupyter notebook
```

## Octave

* Octave is **free** software licensed under the [GNU General Public License (GPL)](https://www.gnu.org/software/octave/license.html). 
* The Octave syntax is largely compatible with Matlab.

### Download and Install

Follow the link to download and install. http://wiki.octave.org/Octave_for_Microsoft_Windows

> DON'T download the latest version, use the stable version(5.2.0 is the version I use presently)

### Validation
```shell
octave --version
```

### Configure Environment Variables

Add location to `PATH`

## Octave Kernel

Run the following command directly.

```shell
pip install metakernel
pip install octave_kernel
python -m octave_kernel install
```

Finally, you can use `octave` in Jupyter.