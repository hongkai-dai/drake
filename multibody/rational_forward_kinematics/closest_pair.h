#include "drake/multibody/tree/multibody_tree.h"

namespace drake {
namespace multibody {
struct ClosestPair {
  Eigen::Vector3d p_WCa;
  Eigen::Vector3d p_WCb;
  double distance;
};

}  // namespace multibody
}  // namespace drake
