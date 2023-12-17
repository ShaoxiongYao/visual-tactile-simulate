# visual-tactile-simulate

A minimal package to simulate visual deformation and tactile response of deformable objects.

We currently support simulating *quasi-static* linear FEM models. The simulation is implicit and stable for high stiffness materials.

Major components:
    + material model definition
        + compute forces
        + compute stiffness matrix (using autodiff)
    + simulator
        + forward simulation
    + estimator

Example tree simulation: