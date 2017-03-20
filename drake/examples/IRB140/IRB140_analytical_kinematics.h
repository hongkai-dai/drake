#include "drake/multibody/rigid_body_tree.h"

#include <Eigen/Core>

namespace drake {
namespace examples {
namespace IRB140 {
/**
 * This class uses the formulation in [1] to compute the forward
 * and inverse kinematics of ABB 1RB 140 robot in the analytical form.
 * [1] Introduction to Robotics by J. J. Craig, 2003
 */
class IRB140AnalyticalKinematics {
 protected:
  const double l0;  // offset of joint 1 in base link in the z direction.
  const double l1_x;  // offset of joint 2 in link 1 in the x direction.
  const double l1_y;  // offset of joint 2 in link 1 in the -y direction.
  const double l2;  // offset of joint 3 in link 2 in the -y direction.
  const double 
};
}
}
}