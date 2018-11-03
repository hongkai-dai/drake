#pragma once

#include <memory>
#include <string>

#include "drake/multibody/multibody_tree/multibody_plant/multibody_plant.h"
#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

namespace drake {
namespace multibody {

std::unique_ptr<multibody_plant::MultibodyPlant<double>> ConstructIiwaPlant(
    const std::string& iiwa_sdf_name);

Eigen::Matrix<double, 3, 8> GenerateBoxVertices(const Eigen::Vector3d& size,
                                                const Eigen::Isometry3d& pose);

std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>
GenerateIiwaLinkPolytopes(const multibody_plant::MultibodyPlant<double>& iiwa);

/*
std::unique_ptr<RigidBodyTreed> ConstructKukaRBT();

std::unique_ptr<RigidBodyTreed> ConstructSchunkGripperRBT();


void AddBoxToTree(RigidBodyTreed* tree,
                  const Eigen::Ref<const Eigen::Vector3d>& box_size,
                  const Eigen::Isometry3d& box_pose, const std::string& name,
                  const Eigen::Vector4d& color = Eigen::Vector4d(0.3, 0.4, 0.5,
                                                                 0.5));

void AddSphereToBody(RigidBodyTreed* tree, int link_idx,
                     const Eigen::Vector3d& pt, const std::string& name,
                     double radius = 0.01);
struct Box {
  Box(const Eigen::Ref<const Eigen::Vector3d>& m_size,
      const Eigen::Isometry3d& m_pose, const std::string& m_name,
      const Eigen::Vector4d& m_color)
      : size{m_size}, pose{m_pose}, name{m_name}, color{m_color} {}
  Eigen::Vector3d size;
  Eigen::Isometry3d pose;
  std::string name;
  Eigen::Vector4d color;
};

struct BodyContactSphere {
  BodyContactSphere(int m_link_idx,
                    const Eigen::Ref<const Eigen::Vector3d>& m_p_BQ,
                    const std::string m_name, double m_radius)
      : link_idx{m_link_idx}, p_BQ{m_p_BQ}, name{m_name}, radius{m_radius} {}
  int link_idx;
  Eigen::Vector3d p_BQ;
  std::string name;
  double radius;
};*/
}  // namespace multibody
}  // namespace drake
