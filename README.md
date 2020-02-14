# EfficientNet-1D

<span style='color:red'>**(Update in progress)**</span>

1D implementation of EfficientNet.

Quite different with 2D implementation.

Running is performed based on experiment configuration file under exp_config directory.

The experiment configuration is csv file, contains hyper parameter settings. (See sample.csv)

Do not modify columns in csv file. If modification is needed, modify all related parts in code consistently.



**Note:**

* $\gamma$ (input resolution) part is not implemented. (Due to reason below) **Set $\gamma=1$.**
* Tensorboard part is not implemented. **Set use_tensorboard=False**.
* **Unfortunately, dataset used here is confidential. Implementation of other opened dataset will be updated later.**





## EfficientNet: 1D vs 2D

The key idea of 2D EfficientNet is compound scaling, considering **depth**(# of layers), **width**(# of channels) and **resolution**(input image size). This is done by under equation,
$$
d=\alpha^\phi\\
w=\beta^\phi\\
r=\gamma^\phi\\
s.t.\ \ \alpha\cdot\beta^2\cdot\gamma^2=2
$$
so that can scale the total computation by controlling compound scale parameter $\phi$
$$
d\cdot w^2\cdot r^2=2^\phi
$$
To know the difference between 2D and 1D case, First thing is to do is think why computation is proportional to $d\cdot w^2\cdot r^2$. Below figure shows why this happens.

