# Riesz Map Study
  In order to explore the effect of the Riesz Map durign a topology optimization, this repository brings two different files to be executed.

## Requiremnts
Fenics > 2019.1.0

Pyadjoint > 2019.1.0

ROL > 0.0.14

For l2, I have to fix the file: /usr/lib/python3.8/site-packages/fenics_adjoint/types/compat.py in the vector.vector()

## How to use
### Gradient representation
In order to view the gradient representation in l2 and L2 for different size meshes:
```
python riesz_map.py --k 2
```
It is possible to change k value such as 3, 4, 5 to see other results.

### Topology Optimization
To see if the topology optimization result changes with different inner products, execute:
```
python optimization_rm.py
```
## Results
The expectation is that even the difference in the inner products, the topology optimization results go to the same result. The difference lies on the performance basically.
![l2 vs L2](https://github.com/bruno-caldas/riesz_representation/raw/master/figures/result_l2_vs_L2.png)
