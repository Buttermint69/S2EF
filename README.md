# S2EF
# (Structure to Energy and forces)
  Quantum simulations in the material science discipline are majorly driven by gold
standard models built on Density Functional Theory (DFT). Broadly the solid-state material
simulations or calculations can be divided into three blocks, first structural relaxations of a
system by iteratively optimizing the positions in the defined space until convergence; second,
calculations of forces and energies exerted by each element of the system and the system; third,
calculation of physical properties of the system like optical, electronic properties, etc.\
  Though the traditional approach yields higher accuracy with real system/material, the
trade-off in terms of computational cost and efficiency. Traditionally, relaxed energies of the
structure are found by first performing structural relaxations through an iterative local
optimization process that estimates the gradients (atomic forces) using DFT, which is in turn
used to update atom positions until convergence. After the convergence, the energy and forces
on every single atom of the relaxed structures are calculated by using predefined methods and
functions. This computationally expensive process typically requires hundreds of DFT
calculations to converge (hours or days of computing per relaxation) and forms the basis of
most computational chemistry efforts.\
  One approach to solving the energy and forces task is using ML to approximate energies
and forces from DFT relaxations. In the approach, we train a machine learning model fuel by
DFT calculation to train an independent function for approximation of energy, after the training
the ML model is expected to calculate the energies and forces of given atomic systems and
their atomic positions. This approach is relatively faster and more efficient, for example, a
standard relaxation using DFT takes 8-10 hours, while ML approaches are desired that can
bring this down to < 10 seconds per relaxation calculation, at least a 1000x improvement.
In this work, we’ve tried to solve the task of “Given any Structure, predict the structure
Energy and per-atom Forces” (S2EF). Given any structure of catalyst and adsorbate i.e., the
position of atoms in the Euclidean space, we’ve tried to predict the energy of the system by
using the subset of machine learning, graph neural networks.
