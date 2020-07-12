---
layout:     post
title:      Jupyter Octave kernel installation
subtitle:   在Jupyter中使用Octave
date:       2020-7-12
author:     Algebra-FUN
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - Octave
    - IDE
    - Jupyter
---

# Jupyter Octave kernel installation

> 在Jupyter中使用Octave

## Jupyter 
### Install
```shell
pip install jupyter
```

### Launch
```shell
jupyter notebook
```

## Octave
### Download
http://wiki.octave.org/Octave_for_Microsoft_Windows

### Config Environment Variables

add location to `PATH`

## Octave Kernel
```shell
pip install metakernel
pip install octave_kernel
python -m octave_kernel install
```

Finally, you can use `octave` in jupyter.