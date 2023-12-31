# visual-tactile-simulate

A minimal python package to simulate visual deformation and tactile response of deformable objects.

Currently simulate *quasi-static* linear FEM models. 

Key features:
+ Stable simulation from stable implicit equilibrium solving. 
+ Flexible to different material models, 
    - stiffness matrices are derived from elastic energy using PyTorch auto-diff.
+ Efficeincy from sparse linear solver. 
    - solve one time step for ~4000 tetrahedra mesh takes 0.03s

Example of cube simulation, red arrow indicates pushed deformation, green arrow indicates contact force:

![sim_cube_fem](https://github.com/ShaoxiongYao/visual-tactile-simulate/assets/49648374/a442b3dc-0625-4030-a52c-b5cb497e6abe)

Example of tree deformation: 

![sim_tree](https://github.com/ShaoxiongYao/visual-tactile-simulate/assets/49648374/0a2e33f1-69d6-4223-9223-a4374959a4c8)


Testing scripts:

```
python tests/test_sim_tree.py
```

The simulation of the tree deformation will be displayed.
