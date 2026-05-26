#include "drake/multibody/plant/test_utilities/make_kinematic_chain.h"

#include <cmath>
#include <memory>

#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/revolute_joint.h"

namespace drake {
namespace multibody {
namespace test {
namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;

void CheckChain(int num_links) {
  auto plant = MakeKinematicChain(num_links);
  ASSERT_NE(plant, nullptr);
  ASSERT_TRUE(plant->is_finalized());

  // world + N links.
  EXPECT_EQ(plant->num_bodies(), num_links + 1);
  // link0 is welded, so there are (N - 1) revolute DoFs.
  EXPECT_EQ(plant->num_positions(), num_links - 1);
  EXPECT_EQ(plant->num_velocities(), num_links - 1);
  // 1 weld + (N - 1) revolute.
  EXPECT_EQ(plant->num_joints(), num_links);

  // Each link must have a non-origin CoM.
  for (int i = 0; i < num_links; ++i) {
    const RigidBody<double>& body =
        plant->GetBodyByName("link" + std::to_string(i));
    EXPECT_GT(body.default_com().norm(), 0.0)
        << "link" << i << " has CoM at origin";
  }

  // Consecutive revolute joints must have non-parallel axes.
  for (int i = 0; i + 1 < num_links - 1; ++i) {
    const auto& j0 =
        plant->GetJointByName<RevoluteJoint>("joint" + std::to_string(i));
    const auto& j1 =
        plant->GetJointByName<RevoluteJoint>("joint" + std::to_string(i + 1));
    const double cos_angle = j0.revolute_axis().dot(j1.revolute_axis());
    EXPECT_LT(std::abs(cos_angle), 1.0 - 1e-6)
        << "joint" << i << " and joint" << (i + 1) << " are parallel";
  }

  // Sanity: the plant must be usable for dynamics computations. The mass
  // matrix at q = 0 should be symmetric positive-definite of size nv x nv.
  auto context = plant->CreateDefaultContext();
  plant->SetPositions(context.get(), VectorXd::Zero(plant->num_positions()));
  MatrixXd M(plant->num_velocities(), plant->num_velocities());
  plant->CalcMassMatrix(*context, &M);
  ASSERT_EQ(M.rows(), num_links - 1);
  ASSERT_EQ(M.cols(), num_links - 1);
  EXPECT_TRUE(M.isApprox(M.transpose(), 1e-12));
  if (num_links > 1) {
    Eigen::SelfAdjointEigenSolver<MatrixXd> eig(M);
    ASSERT_EQ(eig.info(), Eigen::Success);
    EXPECT_GT(eig.eigenvalues().minCoeff(), 0.0);
  }
}

GTEST_TEST(MakeKinematicChainTest, SmallChain) {
  CheckChain(3);
}

GTEST_TEST(MakeKinematicChainTest, MediumChain) {
  CheckChain(10);
}

GTEST_TEST(MakeKinematicChainTest, SingleLinkIsWeldedWithNoDofs) {
  auto plant = MakeKinematicChain(1);
  EXPECT_EQ(plant->num_bodies(), 2);  // world + link0
  EXPECT_EQ(plant->num_positions(), 0);
  EXPECT_EQ(plant->num_velocities(), 0);
  EXPECT_EQ(plant->num_joints(), 1);  // the weld
  const RigidBody<double>& body = plant->GetBodyByName("link0");
  EXPECT_GT(body.default_com().norm(), 0.0);
}

GTEST_TEST(MakeKinematicChainTest, InvalidArguments) {
  DRAKE_EXPECT_THROWS_MESSAGE(MakeKinematicChain(0), ".*num_links >= 1.*");
  MultibodyPlant<double>* null_plant = nullptr;
  DRAKE_EXPECT_THROWS_MESSAGE(AddKinematicChain(null_plant, 3),
                              ".*plant != nullptr.*");
}

GTEST_TEST(MakeKinematicChainTest, AddToExistingPlant) {
  MultibodyPlant<double> plant(0.0);
  AddKinematicChain(&plant, 5);
  // Caller can still add to the plant before Finalize().
  EXPECT_FALSE(plant.is_finalized());
  plant.Finalize();
  EXPECT_EQ(plant.num_positions(), 4);
}

}  // namespace
}  // namespace test
}  // namespace multibody
}  // namespace drake
