#pragma once

#include <string>

#include "drake/manipulation/dev/remote_tree_viewer_wrapper.h"
#include "drake/manipulation/planner/object_contact_planning.h"

namespace drake {
namespace manipulation {
namespace planner {
class Block {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Block)

  Block();

  const Eigen::Matrix<double, 3, 8>& p_BV() const { return p_BV_; }

  double width() const { return 0.1; }

  double height() const { return 0.15; }

  double mass() const { return 1; }

  Eigen::Vector3d dimension() const {
    return Eigen::Vector3d(width(), width(), height());
  }

  Eigen::Vector3d center_of_mass() const { return Eigen::Vector3d::Zero(); }

  std::vector<int> bottom_vertex_indices() const { return {1, 3, 5, 7}; }

  std::vector<int> top_vertex_indices() const { return {0, 2, 4, 6}; }

  std::vector<int> positive_x_vertex_indices() const { return {0, 1, 2, 3}; }

  std::vector<int> negative_x_vertex_indices() const { return {4, 5, 6, 7}; }

  std::vector<int> positive_y_vertex_indices() const { return {0, 1, 4, 5}; }

  std::vector<int> negative_y_vertex_indices() const { return {2, 3, 6, 7}; }

  std::vector<int> bottom_and_positive_x_vertex_indices() const {
    return {0, 1, 2, 3, 5, 7};
  }

  const std::vector<BodyContactPoint>& Q() const { return Q_; }

  std::vector<int> facet_Q_indices() const { return {0, 1, 2, 3, 4, 5}; }

  std::vector<int> edge_Q_indices() const {
    return {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  }

  double mu() const { return 0.5; }

  const Eigen::Matrix3d& I_B() const { return I_B_; }

  Eigen::Matrix3Xd vertex_position(
      const std::vector<int>& vertex_indices) const {
    Eigen::Matrix3Xd pos(3, vertex_indices.size());
    for (int i = 0; i < static_cast<int>(vertex_indices.size()); ++i) {
      pos.col(i) = p_BV_.col(vertex_indices[i]);
    }
    return pos;
  }

  Eigen::Matrix3Xd contact_Q_position(const std::vector<int>& Q_indices) const {
    Eigen::Matrix3Xd pos(3, Q_indices.size());
    for (int i = 0; i < static_cast<int>(Q_indices.size()); ++i) {
      pos.col(i) = Q_[Q_indices[i]].p_BQ();
    }
    return pos;
  }

 private:
  Eigen::Matrix<double, 3, 8> p_BV_;
  Eigen::Matrix3d I_B_;
  std::vector<BodyContactPoint> Q_;
};

void VisualizeBlock(dev::RemoteTreeViewerWrapper* viewer,
                    const Eigen::Ref<const Eigen::Matrix3d>& R_WB,
                    const Eigen::Ref<const Eigen::Vector3d>& p_WB,
                    const Block& block);

void VisualizeForce(dev::RemoteTreeViewerWrapper* viewer,
                    const Eigen::Ref<const Eigen::Vector3d>& p_WP,
                    const Eigen::Ref<const Eigen::Vector3d>& f_WP,
                    double normalizer, const std::string& path,
                    const Eigen::Ref<const Eigen::Vector4d>& color);

void VisualizeTable(dev::RemoteTreeViewerWrapper* viewer);

void AllVerticesAboveTable(const Block& block, ObjectContactPlanning* problem);

solvers::MatrixDecisionVariable<3, Eigen::Dynamic> SetTableContactVertices(
    const Block& block, const std::vector<int>& vertex_indices, double mu_table,
    int knot, double distance_big_M, ObjectContactPlanning* problem);

}  // namespace planner
}  // namespace manipulation
}  // namespace drake
