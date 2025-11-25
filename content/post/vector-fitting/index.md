---
title: Vector Fitting
description: Summary Of Vector Fitting
slug: vector-fitting
date: 2025-11-20 08:00:00+0000
image: cover.jpg
categories:
    - algorithm
tags:
    - algorithm
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

> 正在施工中！
> todo list:
> 1. 添加正交基证明
> 2. 撰写参数化思路

<!-- more -->

## 开始
Vector Fitting算法算是我接触的第一个算法了，也是我第一次使用`纸笔`和`numpy`推导实现的。
这个算法我学到了很多，并且在`sckit-rk`库的基础上进一步完善。

## 引入
为什么我们需要vector fitting 算法。

仿真是及其消耗时间和算力的行为，在进行设计的时候应该减少仿真的次数，用尽可能少的频点获取带宽内的响应。

数据点极少的情况下，通常会进行**内插**的方式扩充频点。这对于一个物理可实现的系统来说是毁灭性打击。

不论是使用线性内插还是,cubic内插，都会破坏数据的连续性，尤其是两个频点之间存在极点时，极点可能被跳过。此时响应会在史密斯圆图上走出一条不连续的曲线，结果也就失去了物理意义。

vector fitting的意义是，使用有限多个频点，建模无源网络的频率响应。使得响应的各个点连续，具有物理意义。

## vector fitting原理

已知一个无源网络的频率响应往往可以使用有理函数进行表示
$$
\begin{align}
H(s)=\frac{N(s)}{D(s)}=\frac{\sum_{n=0}^{N} a_ns^n}{\sum_{n=0}^{N} b_ns^n}
\end{align}
$$
N(s)和D(s)都是幂函数，而幂函数在机器学习中广泛使用。因此首先选用多项式基$s^N$作为拟合的基底

## 算法初步设计
既然已经得到了需要拟合的模型，那么直接与仿真数据做对比，一般使用l2范数，方便做最小二乘计算最优参数

$$
\begin{align}
e^2=\frac{1}{K}\sum^{K}_{k=1}|H_k-\widetilde{H}(jw_k)|^2
\end{align}
$$

最小二乘的目标是最小化$e^2$，使得拟合结果尽量贴近真实值

### 将非线性问题转化为线性问题
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

其次，也是最重要的一点，幂函数基本身就高度不正交，且随频率的增大急剧增大。如果将其写作一个矩阵，这个矩阵叫做范德蒙德病态矩阵。高阶项的权重远远大于低阶项，导致在计算最小二乘的时候，低阶项的系数被忽略，迭代后不收敛。

为了解决病态问题，需要引入SK迭代，来归一化每一项的系数

### 权重问题

$$
\begin{align}
\left(e_{SK}^{(i)}\right)^2 &= \frac{1}{K} \sum_{k=1}^{K} \left| \frac{H_k \sum_{n=0}^{N} b_n^{(i)}(j\omega_k)^n - \sum_{n=0}^{N} a_n^{(i)}(j\omega_k)^n}{\sum_{n=0}^{N} b_n^{(i-1)}(j\omega_k)^n} \right|^2
\end{align}
$$

前一项的系数$b_n^{(i-1)}$是常数，所以（5）式的分母就是常数，使用前一项的分母归一化此时的每一项，经过多次迭代，便可以将每一项的最小二乘权重归一化到`1`

### 变基
虽然我们已经使用了sk迭代，但是$s^n$受频率的影响非常大，我们又选择它作为基，所以不论怎么改善，各个系数之间的权重永远是不一样的。这导致在公式写为矩阵后，仍然包含范德蒙德块。进而导致系数不收敛。

此时便需要考虑更换基，使用与频率无关或者关系小的基。

通过 “极点预设” + “分母线性化” 得到一个新的表达式，将多项式基转化为部分分式基

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
n^{(i)} &= se^{(i)}+ w_0^{(i)} + \sum_{n=1}^{\bar{n}} \frac{w_n^{(i)}}{s - p_n^{(0)}}\\
d^{(i)} &= 1 + \sum_{n=1}^{\bar{n}} \frac{r_n^{(i)}}{s - p_n^{(0)}}
\end{align}
$$

此时我们选取的基为$\frac{1}{s-p_n^{(0)}}$，$p_0$是在s的范围中，选取的n对共轭极点。由于sk迭代对分母的归一化作用，再无穷次迭代后，分母趋于1，则直接使用1作为分母的常数

此时的最小二乘表达式
$$
\left(e_{SK}^{(i)}\right)^2 = \frac{1}{K} \sum_{k=1}^{K} \left| \frac{H_k d^{(i)}(j\omega_k) - n^{(i)}(j\omega_k)}{d^{(i-1)}(j\omega_k)} \right|^2
$$

### 写为矩阵的形式
$$
\begin{align}
\begin{bmatrix}
\Phi_0^{(i)} & -D_H \Phi_1^{(i)}
\end{bmatrix}
\begin{bmatrix}
c_H^{(i)} \\[4pt]
c_w^{(i)}
\end{bmatrix}
= V_H
\end{align}
$$

其中

$$
\begin{align}
\Phi_0^{(i)} &=
\begin{bmatrix}
1 & \frac{1}{j\omega_1 - p_1^{(i-1)}} & \cdots & \frac{1}{j\omega_1 - p_{\bar n}^{(i-1)}} \\
\vdots & \vdots & \ddots & \vdots \\
1 & \frac{1}{j\omega_{\bar k} - p_1^{(i-1)}} & \cdots & \frac{1}{j\omega_{\bar k} - p_{\bar n}^{(i-1)}}
\end{bmatrix}\\
\
\Phi_1^{(i)} &=
\begin{bmatrix}
\frac{1}{j\omega_1 - p_1^{(i-1)}} & \cdots & \frac{1}{j\omega_1 - p_{\bar n}^{(i-1)}} \\
\vdots & \ddots & \vdots \\
\frac{1}{j\omega_{\bar k} - p_1^{(i-1)}} & \cdots & \frac{1}{j\omega_{\bar k} - p_{\bar n}^{(i-1)}}
\end{bmatrix}\\
\
D_H &= \mathrm{diag}\{H_1,\ldots,H_{\bar k}\}\\
\
V_H &= [H_1 \; \cdots \; H_{\bar k}]^T\\
\
c_H^{(i)} &= [\, r_0^{(i)} \; \cdots \; r_{\bar n}^{(i)} \,]^T\\
\
c_w^{(i)} &= [\, w_1^{(i)} \; \cdots \; w_{\bar n}^{(i)} \,]^T
\end{align}
$$

### 评估拟合情况
$$
\begin{align}
w^{(i)}(s) &=\frac{d^{(i)}(s)}{d^{(i-1)}(s)}\\
&= \frac{1 + \sum_{n=1}^{\bar{n}} \frac{d_n^{(i)}}{s - p_n^{(0)}}}{1 + \sum_{n=1}^{\bar{n}} \frac{d_n^{(i-1)}}{s - p_n^{(0)}}}\\
&= \frac{\prod(s - p_n^{(i)}) / \prod(s - p_n^{(0)})}{\prod(s - p_n^{(i-1)}) / \prod(s - p_n^{(0)})}\\
&= 1 + \sum_{n=1}^{\bar{n}} \frac{w_n^{(i)}}{s - p_n^{(i-1)}}
\end{align}
$$

经过迭代后，$\omega$的一范数需要收敛到1，否则就是拟合失败

## 进一步设计
公式(11)中为什么要出现`1`这个常数项呢。有两方面的考虑
1. 分母在收敛的情况下最终一定会趋于`1`
2. 防止$d^{i}/d^{i-1}=0$导致`平凡零解`
   
但是，这也导致缺少了一个自由度，使得算法收敛变慢。因此需要将其转换为待拟合的变量，参与最小二乘计算。并使用`惩罚项`约束平凡零解。

### 平凡零解
在算法中，如果分母写成：

$$
d^{(i)}(s)=\sum_{n=1}^{\bar n}\frac{r_n^{(i)}}{s-p_n^{(0)}}
$$

则最小二乘会自然给出：

$$
r_1^{(i)}=\cdots=r_{\bar n}^{(i)}=0
$$

即：

$$
d^{(i)}(s)=0
$$

这是一个**无意义的、退化的解**，它会让 VF：

* 得不到权函数
* 得不到极点更新
* 迭代失效

所以最初设计加上 **常数项 1** 来避免这一问题。

#### 设计惩罚项
$$
\begin{align}
&\begin{bmatrix}
\Phi_0^{(i)} & -D_H \Phi_1^{(i)} \\
0 & \frac{\beta}{k} (1_k)^T \Phi_0^{(i)}
\end{bmatrix}
\begin{bmatrix}
c_H^{(i)} \\
c_w^{(i)}
\end{bmatrix}=
\begin{bmatrix}
0 \\
\beta
\end{bmatrix}\\
\
&w^{(i)}(s) = w_0^{(i)} + \sum_{n=1}^{\bar n} \frac{w_n^{(i)}}{s - p_n^{(i-1)}}\\
\
&\beta = \sqrt{ \sum_{k=1}^{\bar{k}} |H_k|^2 }
\end{align}
$$

在原来的最小二乘矩阵内加入一行惩罚项，并对权重进行归一化。
惩罚项会阻止最小二乘优化到平凡零解的情况。

## 矩阵计算极点
### 状态空间
使用状态空间方程表示系统,A矩阵为
$$
\begin{align}
A &= diag(p_1,p_2,...,p_N)\\
B &= [1,1,...,1]\\
C &= [r_1,r_2,...,r_n]\\
D &= r_0\\
D(s)&=C(sI−A)^{−1}B+D
\end{align}
$$
其中
| 矩阵    | 维度           | 含义                |
| ----- | ------------ | ----------------- |
| **A** | $(n \times n)$ | 极点位置，系统动态   |
| **B** | $(n \times m)$ | 输入如何激发每个极点        |
| **C** | $(p \times n)$ | 每个极点对输出的贡献权重        |
| **D** | $(p \times m)$ | 直接传输项 |

## 算法优化
### 并行化
在无源网络中
* 极点对应物理系统的 **固有模态**。系统特性不会随输入变化。
* 一个多端口网络的所有传输函数元素都来自同一个系统，因此物理极点应该是相同的。
* 因此 **所有 $H_{q,m}(s)$ 共享同一组极点 $p_n$**。

因此，可以直接将最小二乘改写为如下形式（为了直观，下列的公式没有添加惩罚项）
$$
\begin{bmatrix}
\Phi_0^{(i)} & 0 & \cdots & 0 & -D_{H_{11}}\,\Phi_1^{(i)} \\
0 & \Phi_0^{(i)} & \ddots & \vdots & -D_{H_{21}}\,\Phi_1^{(i)} \\
\vdots & \ddots & \ddots & 0 & \vdots \\
0 & \cdots & 0 & \Phi_0^{(i)} & -D_{H_{\bar q \bar m}}\,\Phi_1^{(i)}
\end{bmatrix}
\begin{bmatrix}
c_{H_{11}}^{(i)} \\
c_{H_{21}}^{(i)} \\
\vdots \\
c_{H_{\bar q \bar m}}^{(i)} \\
c_w^{(i)}
\end{bmatrix}
=\
\begin{bmatrix}
V_{H_{11}} \\
V_{H_{21}} \\
\vdots \\
V_{H_{\bar q \bar m}}
\end{bmatrix}
$$

使用增广矩阵，将多个端口及其响应同时添加到线性方程中，同时进行拟合。

### 实数替代复数
在原始基于部分分式的状态空间实现中，复数极点会导致 (A) 和 (C) 含有复数元素。而复数乘法的计算量约为实数的 4 倍，因此有必要通过**变基**将整个系统转化为等价的 **全实数状态空间模型**，从而显著降低计算复杂度。

#### 极点结构特性

由拉普拉斯变换（以及系统实系数假设）可知，极点只能是：

1. **实极点**：$p \in \mathbb{R}$
2. **复共轭极点对**：$p = a + jb,\quad p^* = a - jb$

因此状态空间中对应的每个复共轭对可以被替换为一个等价的 **二维实子系统**。

#### 复极点的实化变换

对一对共轭极点：

$$
p = a + jb,\qquad p^* = a - jb,
$$

将其组成向量：

$$
\begin{bmatrix}p & \ p^* \end{bmatrix}
$$

并引入线性变换矩阵：

$$
T=
\begin{bmatrix}
1 & 1 \\
i & -i
\end{bmatrix},
$$

则：

$$
T * \begin{bmatrix}p , p^* \end{bmatrix}^T=\
\begin{bmatrix}
p + p^* ,
i(p - p^*)
\end{bmatrix}^T\\
=\
\begin{bmatrix}
2a ,2b
\end{bmatrix}.
$$
**矩阵 (T) 将一对共轭极点映射为两个实数参数 (a) 与 (b)**（分别对应极点的实部与虚部）

#### 在状态空间矩阵中的应用

原始的 (A) 和 (C) 是由所有极点和残差组成的
变基后：

* **实极点对应的 1×1 实块保持不变**
* **每对复共轭极点被替换为一个 2×2 的实 Jordan 模态块**

新的实数状态矩阵形式为：

$$
\begin{align}
&A_{\text{real}} =
\begin{bmatrix}
A_{\text{real poles}} & 0 \\
0 &
\begin{matrix}
a & -b \\
b & a
\end{matrix}
\end{bmatrix}
\end{align}
$$

分块矩阵为：
$$
\begin{align}
&A_{\text{blk}}=\begin{bmatrix}a & -b\\
 b & a\end{bmatrix}\\
&B_{\text{blk}}=\begin{bmatrix}1,0\end{bmatrix}^T\\
&C_{\text{blk}}=\begin{bmatrix}2\Re(r) & 2\Im(r)\end{bmatrix}\\
\end{align}
$$

证明分块的结果仍为极点-残差形式
$$
r = \alpha + j\beta\\
C_{\text{blk}}(sI-A_{\text{blk}})^{-1}B_{\text{blk}}
=\frac{2\alpha (s-a)+2b\beta}{(s-a)^2+b^2}
=\frac{r}{s-p}+\frac{r^*}{s-p^*}.
$$

对应的 (C) 也同步应用变换，使其完全实数化。

这样就将整个系统转换为 **全实数状态空间模型**，避免了复数运算，提升数值效率与稳定性。

同时,应该对如下公式进行最小二乘计算，求得C后代入状态方程求解。(这里为了方便理解，没有将惩罚项加入)
$$
\begin{align}
A &= \
\left[
\begin{matrix}
\Re{\Phi_0^{(i)}} & 0 & \dots & -\Re {D_{H_{11}} \Phi_1^{(i)}} \\
\Im{\Phi_0^{(i)}} & 0 & \dots & -\Im {D_{H_{11}} \Phi_1^{(i)}} \\
\vdots & & & \vdots \\
0 & \dots & \Re{\Phi_0^{(i)}} & -\Re{ D_{H_{\bar{q}*m}} \Phi_1^{(i)} } \\
0 & \dots & \Im{\Phi_0^{(i)}} & -\Im{ D*{H_{\bar{q}_m}} \Phi_1^{(i)} }
\end{matrix} 
\right]
\\
x &= [c_{H_{11}}^{(i)},...,c_{H_{qm}}^{(i)},c_w^{(i)}]^T\\
b&=[\Re V_{H_{11}},\Im V_{H_{11}},...,\Re V_{H_{qm}},\Im V_{H_{qm}}]
\end{align}
$$

### QR分解
在 Vector Fitting（VF）中：迭代的目的**目标不是求分子/分母全部系数，而是迭代更新极点**
极点由分母决定，那为什么还需要让分子系数参与迭代呢？

因此，直接借用QR分解，将最小二乘矩阵分解为正交阵Q和上三角矩阵R
$$
\begin{align}
\left[ \Phi_0^{(i)} ~~-D_{H_{qm}}\Phi_1^{(i)} \right]
= \left[ Q_{qm}^1~~ Q_{qm}^2 \right]
\begin{bmatrix}
R_{qm}^{11} & R_{qm}^{12} \\
0 & R_{qm}^{22}
\end{bmatrix}
\end{align}
$$

因此整个系统就能被缩减为：

$$
\begin{align}
R_{qm}^{22} c_w^{(i)}=
(Q_{qm}^2)^T V_{H_{qm}}
\end{align}
$$

直到迭代完成，再求一次完整的最小二乘，即可获取所有端口的分子系数，最后再代入状态方程求得频率响应。

### 基正交化
普通 Vector Fitting 使用以下形式的逼近：

$$
H_k(s) = \sum_{k=1}^n \frac{r_k}{s - a_k} + d + s e.
$$

其中 ${a_k}$ 是极点，${r_k}$ 是残差。
这一部分分式基**不是正交的**

* 对于极点位置聚集、高频幅度差异很大的系统，列之间高度相关，出现**病态矩阵**的情况；
* 正交性差 → QR/SVD 求解残差和极点敏感；
* 拟合高阶系统时残差求解不稳定；

当极点近似重合时，多个极点对应的基退化为一个基，导致极点的系数自由度极高且不收敛。

因此需要在设计基函数$\phi$的时候，需要先将基转化为正交基，避免在最小二乘时出现病态。

#### 穆兹-拉格朗日正交基的构建

实极点的情况：
$$
pole = -a_p
$$

$$
\begin{align}
\phi_p(s)&=\gamma_p
\left( \prod_{j=1}^{p-1} \frac{s-a_j^*}{s+a_j} \right)
\frac{1}{s+a_p}
\end{align}
$$

复共轭的情况：
$$
\begin{align}
pole_1 &= -a_p \\
pole_2 &= -a_{p+1}= -a_p^*
\end{align}
$$

$$
\begin{align}
\phi_p(s)&=\gamma_p
\left( \prod_{j=1}^{p-1} \frac{s-a_j^*}{s+a_j} \right)
\frac{s-x}{(s+a_p)(s+a_{p+1})}\\
\phi_{p+1}(s)&=\gamma_{p+1}
\left( \prod_{j=1}^{p-1} \frac{s-a_j^*}{s+a_j} \right)
\frac{s-y}{(s+a_p)(s+a_{p+1})}
\end{align}
$$

其中
$$
\begin{align}
x &= \sqrt{a_p a_{p+1}} = |a_p|\\
y &= -\sqrt{a_p a_{p+1}} = -|a_p|\\
\gamma_p &= \sqrt{a_p + a_{p+1}} = \sqrt{2 \Re(a_p)}\\
\gamma_{p+1} &= \sqrt{a_p + a_{p+1}} = \sqrt{2 \Re(a_p)}
\end{align}
$$

修改基为正交基后，最小二乘部分不需要任何改动，但是，需要对状态方程进行修改。

#### ABCD矩阵初步构建
$$
\begin{align}
A_{P\times P} &=
\begin{bmatrix}
-a_1 & 0 & 0 & \cdots & 0 \\
2\Re(-a_1) & -a_2 & 0 & \cdots & 0 \\
2\Re(-a_1) & 2\Re(-a_2) & -a_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
2\Re(-a_1) & 2\Re(-a_2) & 2\Re(-a_3) & \cdots & -a_P
\end{bmatrix}\\
\
B_{1\times P} &=
\begin{bmatrix}
1 & 1 & \cdots & 1
\end{bmatrix}\\
\
C_{P\times 1} &=
\begin{bmatrix}
\tilde{c}_1 \sqrt{2\Re(a_1)} \\
\tilde{c}_2 \sqrt{2\Re(a_2)} \\
\vdots \\
\tilde{c}_P \sqrt{2\Re(a_P)}
\end{bmatrix}^{T}\\
\
D_{1\times 1} &= \tilde{c}_0
\end{align}
$$
#### 实极点

如果 pole 为 **real：**

$$
a_p \in \mathbb{R}
$$

则对应：

##### A Block

$$
A_p = [-a_p]
$$

##### B Block

所有 block 的 B 均为 1：

$$
B_p = 1
$$

##### C Block

$$
C_p = \tilde{c}_p \sqrt{2 \Re(a_p)}
$$

#### 复共轭极点

若极点为复共轭极点

则需要把复基变成二阶实系统，将对应的块插入原矩阵中：

##### A Block

$$
A_p=
\begin{pmatrix}
\Re(-a_p) & \Re(-a_p)-|a_p| \\
\Re(-a_p)+|a_p| & \Re(-a_p)
\end{pmatrix}
$$

##### B Block

$$
B_p=\begin{pmatrix}1 \\ 1\end{pmatrix}
$$

##### C Block

$$
C_p=\tilde{c}_p
\begin{pmatrix}
\sqrt{\Re(a_p)} \\ \sqrt{\Re(a_p)}
\end{pmatrix}
$$

## 总结

### 基变换：将问题转换到更容易解决的空间

很多问题看似困难，是因为它们被放置在了**错误的基底**中。

基变换的核心哲学：

不要硬解当前空间中的问题，把它转换到一个更适合求解、结构更简单的空间中。

在数学、物理、信号处理、机器学习中随处可见：

- 傅里叶变换：时频域转换
- 小波变换：把局部特征表达得更清晰
- 主成分分析（PCA）：找到数据能量最大的方向
- 实数化复数模型：减少计算与复杂度

### 正交化：让信息不互相干扰

许多算法性能差，不是因为模型不行，而是因为不同方向的信息相互污染、互相耦合，导致计算不稳定。

>正交化的核心哲学是：
>
>让每个分量尽可能独立，让误差不会在不同维度之间传递。

你可以在各个领域看到它：

- Gram–Schmidt与QR 分解：分解为正交矩阵和上三角矩阵；计算最小二乘

- 神经网络训练中的 batch norm

- 数值分析中正交基改善条件数

>正交化的意义不仅是数学操作，而是一个极其通用的原则：
>
>消除耦合，让系统稳定，从而更容易优化。

### 控制论视角：任何复杂系统都可以拆成状态、输入、输出

控制论提供了一个普适的框架：

$$
\dot{x}(t) = Ax+Bu(t)\\
y(t) = Cx+Du(t)
$$

这是一个思想结构，不仅仅是方程。

它告诉我们：

任何复杂系统都可以分成 内部状态（A） 与 外部作用（B）

系统必须可控、可观，否则你无法理解或操纵它

信息的流动结构比具体形式更重要

无论是：

- 信号建模

- 强化学习

- 经济系统

- 生物系统

- 软件架构

- 深度网络

都可以抽象为一个控制系统。

拉普拉斯变换可以将时域信号转换到频域，通过系统极点，展示系统的本征结构

$$
G(s) = C(SI-A)^{-1}B - D
$$

A的特征值就是极点

> 稳定性、可控性、可观性不是控制论专属，而是普适的建模原则。

### 迭代本质：在“能走的方向上”不断逼近真相

几乎所有的非线性问题，都无法一步到位。
迭代方法的哲学本质是：

把大问题拆成可解的小问题，将非线性问题转化为多个局部小规模的线性问题，通过不断更新来逼近最终解。

本质是**数据降维**

迭代思想贯穿全科学：

- Newton iteration

- Gradient descent

- Expectation–Maximization（EM）

- SK iteration

- Krylov 子空间迭代

迭代的底层思想是：

> 设计一个收敛的动力系统，把求解问题变成模拟系统动态演化。
>
>这是对“求解”思想的一次重新理解：
>不是一锤定音，而是构造一个朝向真相的路径。