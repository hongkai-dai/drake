#include "drake/multibody/plant/test_utilities/make_kinematic_chain.h"

#include <array>
#include <string>

#include "drake/common/drake_throw.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/rotational_inertia.h"
#include "drake/multibody/tree/spatial_inertia.h"

namespace drake {
namespace multibody {
namespace test {
namespace {

using Eigen::Vector3d;

SpatialInertia<double> MakeLinkInertia(int i) {
  const double mass = 0.5 + 0.1 * i;
  const Vector3d p_BoBcm_B =
      Vector3d(0.1, 0.05, 0.02) + 0.01 * Vector3d(i, i, i);
  const double Ixx = 0.02 + 0.001 * i;
  const double Iyy = 0.03 + 0.001 * i;
  const double Izz = 0.04 + 0.001 * i;
  const RotationalInertia<double> I_Bcm(Ixx, Iyy, Izz);
  return SpatialInertia<double>::MakeFromCentralInertia(mass, p_BoBcm_B, I_Bcm);
}

// Returns a sequence of three mutually non-parallel unit axes. Cycling through
// them guarantees that any two consecutive joints have non-parallel axes
// (pairwise dot products are ~0.19).
Vector3d MakeJointAxis(int i) {
  static const std::array<Vector3d, 3> kAxes = {
      Vector3d(1.0, 0.2, 0.0).normalized(),
      Vector3d(0.0, 1.0, 0.2).normalized(),
      Vector3d(0.2, 0.0, 1.0).normalized(),
  };
  return kAxes[i % kAxes.size()];
}

}  // namespace

void AddKinematicChain(MultibodyPlant<double>* plant, int num_links) {
  DRAKE_THROW_UNLESS(plant != nullptr);
  DRAKE_THROW_UNLESS(!plant->is_finalized());
  DRAKE_THROW_UNLESS(num_links >= 1);

  const RigidBody<double>& link0 =
      plant->AddRigidBody("link0", MakeLinkInertia(0));
  plant->WeldFrames(plant->world_frame(), link0.body_frame());

  // Fixed translational offset of the joint frame F in the parent body P, so
  // that consecutive link body origins do not all coincide. The offset is the
  // same for every joint; the rotation of F relative to P is identity, so the
  // joint axis is still expressed in the parent body frame.
  const math::RigidTransformd X_PF(Eigen::Vector3d(0.25, 0.0, 0.0));

  for (int i = 1; i < num_links; ++i) {
    const RigidBody<double>& parent =
        plant->GetBodyByName("link" + std::to_string(i - 1));
    const RigidBody<double>& child =
        plant->AddRigidBody("link" + std::to_string(i), MakeLinkInertia(i));
    plant->AddJoint<RevoluteJoint>("joint" + std::to_string(i - 1), parent,
                                   X_PF, child, {}, MakeJointAxis(i - 1));
  }
}

std::unique_ptr<MultibodyPlant<double>> MakeKinematicChain(int num_links) {
  auto plant = std::make_unique<MultibodyPlant<double>>(0.0);
  AddKinematicChain(plant.get(), num_links);
  plant->Finalize();
  return plant;
}

}  // namespace test
}  // namespace multibody
}  // namespace drake
