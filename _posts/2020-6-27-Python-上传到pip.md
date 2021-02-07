---
layout:     post
title:      Upload my package to pip
subtitle:   把自己的Python包上传到Pypipng
date:       2020-6-27
author:     Algebra-FUN
header-img: img/post-bg-2020-6-27.png
catalog: true
tags:
    - Python
    - Experience
---

# Upload my  package to pip

### 创建项目，项目结构如下

```text
project
 |--package
 |   |--sth.py
 |   |--__init__.py
 |--setup.py
```

> 注意`setup.py`要与要实际上传的文件夹平行

### 编写setup.py

For example,

```python
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    # pip install nnn
    name="WeReadScan",
    version="0.6.5",
    keywords=("weread", "scan", "pdf", "convert", "selenium"),
    description="WeRead PDF Scanner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # 协议
    license="GPL Licence",

    url="https://github.com/Algebra-FUN/WeReadScan",
    author="Algebra-FUN",
    author_email="2593991307@qq.com",

    # 自动查询所有"__init__.py"
    packages=find_packages(),
    include_package_data=True,
    platforms="window",
    # 提示前置包
    install_requires=['pillow', 'numpy', 'matplotlib',
                      'img2pdf', 'opencv-python', 'selenium'],
    python_requires='>=3.6'
)
```

> 按照格式填写相关信息即可

### 打包

```shell
python setup.py sdist bdist_wheel
```

### 上传

```text
twine upload dist/*
```

### 这个步骤遇到的相关问题：

#### twine安装

```
pip install twine
```

#### python setup.py bdist_wheel 报错的处理办法

若报错如下图：

```shell
usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
   or: setup.py --help [cmd1 cmd2 ...]
   or: setup.py --help-commands
   or: setup.py cmd --help

error: invalid command 'bdist_wheel'
```

多半是setuptools版本不正确或者你的环境中没有安装wheel， 请使用一下命令升级：

```shell
pip install wheel
pip install --upgrade setuptools
```


#### Pypi注册

前往https://pypi.org/注册即可

#### .pypirc配置

在用户根目录下新建文件，用以上传自动身份验证

```ini
[distutils]
index-servers = pypi
 
[pypi]
username:xxx
password:xxx
```

> 也可以不设置，每次都输入username&password
>
> p.s. 在VSCode的Terminal中输入password会是黑色的（第一次会以为没输入上……）

### 安装验证

```text
 pip install <your_package> -i https://pypi.python.org/simple/
```

> 一般镜像源同步的很慢……
>
> Pypi官网也需要耐心等待……