#include <iostream>

#include "drake/geometry/optimization/hyperellipsoid.h"
#include "drake/solvers/csdp_solver.h"

namespace drake {
namespace solvers {
using geometry::optimization::Hyperellipsoid;

int DoMain() {
  Hyperellipsoid E = Hyperellipsoid::MakeUnitBall(2);
  Hyperellipsoid E2 =
      Hyperellipsoid::MakeHypersphere(1.0, Eigen::Vector2d{4.0, 0.0});

  auto [sigma, x] = E.MinimumUniformScalingToTouch(E2);
  std::cout << "sigma: " << sigma << "\n";
  std::cout << "x: " << x(0) << ", " << x(1) << "\n";
  return 0;
}
}  // namespace solvers
}  // namespace drake

int main() {
  return drake::solvers::DoMain();
}
