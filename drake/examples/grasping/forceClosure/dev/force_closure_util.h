#include <Eigen/Core>

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
/**
 * Returns a polytopic inner approximation of the unit sphere in 6 dimensional
 * space. This polytope has 7 evenly spaced vertices.
 * The computation is adapted from section II.E of
 * Fast Computation of Optimal Contact Forces by Stephen Boyd and Ben Wegbreit
 */
Eigen::Matrix<double, 6, 7> GenerateWrenchPolytopeInnerSphere7Vertices();

/**
 * Returns a polytopic inner approximation of the unit sphere in 6 dimensional
 * space. This polytope has 12 evenly spaced vertices.
 * @return W    W(j, 2*i) = 0 for j ≠ i
 *              W(i, 2*i) = 1
 *              W(j, 2*i+1) = 0 for j ≠ i
 *              W(i, 2*i+1) = -1
 */
Eigen::Matrix<double, 6, 12> GenerateWrenchPolytopeInnerSphere12Vertices();

/**
 * Computes the exact Q1 metric for contact points with linearized friction cones.
 * Namely the largest radius of the ellipsoid in the contact wrench set.
 * The ellipsoid is defined as w' * Q * w <= r²
 * For more information, refer to
 *   Grasp Metrics: Optimality and Complexity. By B Mishra
 * and
 *   Grasping and Fixturing as Submodular Coverage Problems. By John Schulman et.al.
 * @param contact_pts The contact locations.
 * @param friction_edges friction_edges[i] is the edges of the linearized friction cone at i'th contact location.
 * @param Q The weighted norm in the wrench space.
 * @return The largest radius r.
 */
double ForceClosureQ1metricLinearizedFrictionCone(const Eigen::Matrix3Xd& contact_pts, const std::vector<Eigen::Matrix3Xd>& friction_edges, const Eigen::Matrix<double, 6, 6>& Q);
}  // namespace forceClosure
}  // namespace grasping
}  // namespace examples
}  // namespace drake
