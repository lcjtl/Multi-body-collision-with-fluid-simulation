# 二维多边形碰撞与流体模拟
## 作业来源
本次大作业基于*Godfried Toussaint*的论文 **"Solving Geometric Problems with theRotating Calipers" Proceedings of IEEE MELECON 1983**中的旋转卡尺进行多边形之间的精确碰撞检测，此算法在二维上实现比较简单且具有优异的复杂度。

作业中实现了刚体多边形之间的碰撞，并与多体问题和流体模拟进行了结合，实现流体与刚体多边形之间的交互。

## 运行方式
#### 运行环境：
`[Taichi] version 0.8.4, llvm 10.0.0, commit 895881b5, win, python 3.8.9`

#### 运行：
在运行 `main.py`时，可以通过命令行参数 `-sg` 来控制是否用刚体的引力控制粒子移动，否则粒子会受到一个指向中心的引力（可以通过点击屏幕改变引力中心）。

- 使用指向中心的引力 :
`python main.py`

- 使用刚体的引力:
`python main.py  -sg`


## 效果展示
1 : 使用指向中心的引力 
![center gravity](./data/center_gravity.gif)

2 : 改变中心位置

![change center](./data/change_center.gif)

3 : 使用刚体的引力

![solids gravity](./data/solids_gravity.gif)

## 整体结构
```
-LICENSE
-|data
-README.MD
-main.py
-solid.py
-particle.py
```

## 实现细节：
`main.py`是项目的入口代码，其中包含一个MainSystem类和一个`main`函数，MainSystem控制刚体系统SolidSystem和粒子系统ParticleSystem之间的交互，其中SolidSystem类的实现在`solid.py`中，ParticleSystem类的实现在`particle.py`中。

### 整体流程
1. 初始化刚体系统和粒子系统
3. 根据命令行选项控制粒子所受到的引力，默认使用中心引力，可以点击鼠标改变引力中心的位置。可以在命令行加入参数 `-sg` 来使用固体的引力来吸引粒子。
3. 判断刚体之间的碰撞，刚体与粒子之间的碰撞
4. 更新速度和位置
5. 在GUI中显示

### 刚体类
1. 初始化
   
   - 设置field，确定数据结构
   - 随机多边形的边数，大小和颜色
   - 初始化field，设置位置，速度
   - 根据顶点数等信息设置，质量和转动惯量等刚体运动信息
   - 生成边界用于碰撞
   
2. 碰撞检测

   **计算闵可夫斯基和**: minko_sum

    - 为两个多边形初始化RotatingCalipers
    - 旋转一圈，获得闵可夫斯基和
    - 根据闵可夫斯基和计算退出边和退出深度
    - 特判
    - 施加冲量

   **施加冲量**: apply_impluse

    - 改变速度
    - 改变角速度

   **计算物体在相对位置r处的K矩阵**: get_K(r)

   * 参考[`Rigid Body Dynamics`](https://graphics.pixar.com/pbm2001/pdf/notesg.pdf)

3. **计算引力**: compute_force

    - 计算刚体之间距离
    - 计算万有引力

### 粒子类

1. 初始化

   - 设置field，确定数据结构
   - 初始化field，设置位置，速度
   - 设置空间加速网格属性
   - 添加边界粒子
   
2. 初始化边界粒子init_boundary_particle_pos

   * 在左右和下方添加密度为参考密度的粒子

3. 初始化网格search_neighbors

   * 根据网格计算每个粒子的support_radius范围内的相邻粒子
   * 将密度等信息存储，用于后续力的计算

4. 核函数

   **cubic_kernel(r) **

   **cubic_kernel_derivative(r) **

   **cubic_kernel_laplace(r)**

5. 计算压力compute_pressure

   * 根据公式计算压力

6. 施加力apply_force

   * 外力，即引力
   * 压力，compute_pressure计算得出
   * 粘性力，根据周围粒子速度计算得出

### 主类

1. 初始化

   - 用创建好的刚体系统和粒子系统初始化
   - 负责系统之间的交互
2. 改变粒子受到的引力: apply_gravity_to_particle()
   * 用中心引力，则施加向鼠标点击的位置的力
   * 否则，施加刚体之间的万有引力
3. 粒子与刚体之间的碰撞交互: collision()
   * 判断碰撞
   * 依据碰撞深度对粒子施加退出的速度
   * 依据碰撞点的相对速度，施加刚体的运动阻力

