---
title: Vector Fitting
description: Summary Of Vector Fitting
slug: vector-fitting
date: 2025-11-20 00:00:00+0000
image: cover.jpg
categories:
    - algorithm
tags:
    - algorithm
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

> 正在施工中！

<!-- more -->

## 开始
Vector Fitting算法算是我接触的第一个算法了，也是我第一次使用`纸笔`和`numpy`推导实现的。
这个算法我学到了很多，并且在`sckit-rk`库的基础上进一步完善。

## 引入
为什么我们需要vector fitting 算法。

仿真是及其消耗时间和算力的行为，在进行设计的时候应该减少仿真的次数，用尽可能少的频点获取带宽内的响应。

数据点极少的情况下，ADS通常会进行内插。这对于一个物理可实现的系统来说是毁灭性打击。

不论是使用线性内插还是,cubic内插，都会破坏数据的连续性，尤其是两个频点之间存在极点的情况下。极点可能直接被跳过。此时响应会在史密斯圆图上走出一条不连续的曲线，结果也就失去了物理意义。

vector fitting的意义是，使用有限多个频点，建模无源网络的频率响应。使得响应的各个点连续，具有物理意义。

## vector fitting原理

### 选什么模型去拟合
已知一个无源网络的频率响应往往可以使用有理函数进行表示
$$
\begin{align}
H(s)=\frac{N(s)}{D(s)}=\frac{\sum_{n=0}^{N} a_ns^n}{\sum_{n=0}^{N} b_ns^n}
\end{align}
$$
N(s)和D(s)都是幂函数，因此首先选用多项式基$s^N$作为拟合的基底


## 算法设计优化
### 目标（奖励项）
既然已经得到了需要拟合的模型，那么直接与仿真数据做对比，一般使用l2范数，方便做最小二乘计算最优参数

$$
\begin{align}
e^2=\frac{1}{K}\sum^{K}_{k=1}|H_k-\widetilde{H}(jw_k)|^2
\end{align}
$$

最小二乘的目标是最小化$e^2$，使得拟合结果尽量贴近真实值

#### 将非线性问题转化为线性问题
(3)式展开后明显是一个非线性最小化问题
$$
\begin{align}
e^2 = \frac{1}{k}\sum_{k=1}^{\bar{k}} \left|H_k \frac{\sum_{n=0}^{\bar{n}} b_n(j\omega_k)^n - \sum_{n=0}^{\bar{n}} a_n(j\omega_k)^n}{\sum_{n=0}^{\bar{n}} b_n(j\omega_k)^n}\right|^2
\end{align}
$$
在实际计算中需要将其转换为线性问题，才能高效的使用QR分解等方法求解。

最简单的方法便是levy迭代，直接忽略分母，保留分子作为优化目标

$$
\begin{align}
\left(e_L\right)^2 = \frac{1}{k}\sum_{k=1}^{\bar{k}} \left|H_k \sum_{n=0}^{\bar{n}} b_n(j\omega_k)^n - \sum_{n=0}^{\bar{n}} a_n(j\omega_k)^n\right|^2
\end{align}
$$

使用levy迭代，虽然将非线性问题转化为了线性问题。但是，引入了诸多缺陷。

首先是删去分母之后，解的精度下降。

其次，也是最重要的一点，分子中的$(jw_k)^n$项不再有约束，且随频率的增大急剧增大。如果将其写作一个矩阵，这个矩阵叫做范德蒙德病态矩阵。高阶项的权重远远大于低阶项，导致在计算最小二乘的时候，低阶项的系数被忽略，迭代后不收敛。

为了解决病态问题，需要引入SK迭代，来归一化每一项的系数

#### 权重问题

$$
\begin{align}
\left(e_{SK}^{(i)}\right)^2 = \frac{1}{K} \sum_{k=1}^{K} \left| \frac{H_k \sum_{n=0}^{N} b_n^{(i)}(j\omega_k)^n - \sum_{n=0}^{N} a_n^{(i)}(j\omega_k)^n}{\sum_{n=0}^{N} b_n^{(i-1)}(j\omega_k)^n} \right|^2
\end{align}
$$

前一项的系数$b_n^{(i-1)}$是常数，所以（5）式的分母就是常数，使用前一项的分母归一化此时的每一项，经过多次迭代，便可以将每一项的最小二乘权重归一化到`1`

#### 变基
虽然我们已经使用了sk迭代，但是$s^n$受频率的影响非常大，我们又选择它作为基，所以不论怎么改善，各个系数之间的权重永远是不一样的。这导致在公式写为矩阵后，仍然包含范德蒙德块。进而导致系数不收敛。

此时便需要考虑更换基，使用与频率无关或者关系小的基。

通过对原始公式的变换，可以将多项式基转化为部分分式基

$$
\begin{align}
H(s)&=\frac{\sum_{n=0}^{N} a_ns^n}{\sum_{n=0}^{N} b_ns^n} \\
&=\frac{\xi_1\prod_{m=0}^{M}(s-z_m)}{\xi_2\prod_{n=0}^{N}(s-p_n)}
\end{align}
$$

对于分母可写作如下形式(分子也一样)
$$
\begin{align}
d(s) = \xi_2 s^{N} \prod_{n=1}^{N} \left(1 - \frac{p_n}{s}\right)
\end{align}
$$

分子分母同时出现$s^N$项，便可以将其消去，归一化$\xi$项。同时再对其进行部分分式展开，便可以得到响应的部分分式形式

$$
\begin{align}
\widetilde{H}(s)=c_0+se+\sum_{k=1}^{K} \frac{c_k}{s-p_k}
\end{align}
$$

>**物理意义**
K阶幂函数有K个解，而每个**分母**的解都对应一个极点，于是可以写成如下形式
写成部分分式求和形式之后，这里只保留了一阶系数**se**，当然，如果你觉得分子的阶数（系统的零点数量）比分母更多，欢迎添加二阶，三阶系数。但是在无源网络中这是不可能的，零点数量往往比极点少。

再带入SK迭代，便可以拿到第i次迭代的表达式

$$
\begin{align}
n^{(i)} &= se^{(i)}+ c_0^{(i)} + \sum_{n=1}^{\bar{n}} \frac{c_n^{(i)}}{s - p_n^{(0)}}\\
d^{(i)} &= 1 + \sum_{n=1}^{\bar{n}} \frac{d_n^{(i)}}{s - p_n^{(0)}}
\end{align}
$$

此时我们选取的基为$\frac{1}{s-p_n^{(0)}}$，$p_0$是在s的范围中，选取的n对共轭极点。由于sk迭代对分母的归一化作用，再无穷次迭代后，分母趋于1，则直接使用1作为分母的常数

此时的最小二乘表达式
$$
\left(e_{SK}^{(i)}\right)^2 = \frac{1}{K} \sum_{k=1}^{K} \left| H_k \frac{d^{(i)}(j\omega_k)}{d^{(i-1)}(j\omega_k)} - \frac{n^{(i)}(j\omega_k)}{d^{(i-1)}(j\omega_k)} \right|^2
$$

#### 评估拟合情况
$$
\begin{align}
w^{(i)}(s) &=\frac{d^{(i)}(s)}{d^{(i-1)}(s)}\\
&= \frac{1 + \sum_{n=1}^{\bar{n}} \frac{d_n^{(i)}}{s - p_n^{(0)}}}{1 + \sum_{n=1}^{\bar{n}} \frac{d_n^{(i-1)}}{s - p_n^{(0)}}}\\
&= \frac{\prod(s - p_n^{(i)}) / \prod(s - p_n^{(0)})}{\prod(s - p_n^{(i-1)}) / \prod(s - p_n^{(0)})}\\
&= 1 + \sum_{n=1}^{\bar{n}} \frac{w_n^{(i)}}{s - p_n^{(i-1)}}
\end{align}
$$

经过迭代后，$\omega$的二范数需要收敛到1否则就是拟合失败


### 约束（惩罚项）

#### 平凡零解

### 矩阵计算
#### 归一化
#### 状态空间

### 评估算法

### 算法优化
#### 减少计算量
#### 并行化

### 改进算法

## 总结