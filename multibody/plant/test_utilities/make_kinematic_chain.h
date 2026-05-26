#pragma once

#include <memory>

#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace multibody {
namespace test {

// Adds `num_links` rigid bodies to `plant`, welds the first body to the world,
// and connects each subsequent body to its parent with a revolute joint. The
// helper is intentionally non-generic; numerical values (mass, CoM offset,
// rotational inertia, joint-axis directions) are deterministic but otherwise
// arbitrary. Two invariants are guaranteed because they materially affect the
// algorithms that benchmark against this plant:
//
//   1. Each link's center of mass is not located at its body origin.
//   2. Consecutive revolute-joint axes are not parallel.
//
// Bodies are named "link0" through "link{num_links - 1}". Joint i connects
// link{i} (parent) to link{i + 1} (child) and is named "joint{i}", for
// i in [0, num_links - 1).
//
// `plant` must be non-null and not yet finalized. `num_links` must be >= 1.
// This function does not call Finalize(); the caller may add further model
// elements first.
void AddKinematicChain(MultibodyPlant<double>* plant, int num_links);

// Convenience factory. Creates a new continuous-time MultibodyPlant<double>,
// calls AddKinematicChain(plant, num_links), then Finalize()s the plant.
std::unique_ptr<MultibodyPlant<double>> MakeKinematicChain(int num_links);

}  // namespace test
}  // namespace multibody
}  // namespace drake
