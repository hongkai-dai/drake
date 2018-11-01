#pragma once

#include <memory>
#include <string>

#include "drake/multibody/multibody_tree/multibody_plant/multibody_plant.h"

namespace drake {
namespace multibody {

std::unique_ptr<multibody_plant::MultibodyPlant<double>> ConstructIiwaPlant(
    const std::string& iiwa_sdf_name);

Eigen::Matrix<double, 3, 8> GenerateBoxVertices(const Eigen::Vector3d& size,
                                                const Eigen::Isometry3d& pose);
}  // namespace multibody
}  // namespace drake
