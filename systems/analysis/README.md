# CLF and CBF
This folder contains the code for the paper

[Convex synthesis and verification of control-Lyapunov and barrier functions with input constraints](https://arxiv.org/pdf/2210.00629.pdf)

*Hongkai Dai and Frank Permenter, American Control Conference, 2023*

The code is written in C++ with python bindings.

To check the implementation of verifying/synthesizing CLF/CBF, refer the code control_lyapunov.h/cc and control_barrier.h/cc

For using this code, you could check the examples in "test/pendulum_trig_clf_demo.py" or "test/quadrotor3d_trig_clf_demo.py". Similarly you will find the examples on CBF in the same "test" folder.

# Optimization solver
Please make sure that you have Mosek solver on your machine. You can check the Drake installation [manual](https://drake.mit.edu/bazel.html#developing-drake-using-bazel) on using Mosek inside Drake. Specifically you will need to set the environment variable
```
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
```

