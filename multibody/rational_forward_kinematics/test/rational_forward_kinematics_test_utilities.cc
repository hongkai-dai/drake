#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"

#include "drake/common/find_resource.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/multibody/benchmarks/kuka_iiwa_robot/make_kuka_iiwa_model.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/vector_system.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"

namespace drake {
namespace multibody {
std::unique_ptr<MultibodyPlant<double>> ConstructIiwaPlant(
    const std::string& iiwa_sdf_name) {
  const std::string file_path =
      "drake/manipulation/models/iiwa_description/sdf/" + iiwa_sdf_name;
  auto plant = std::make_unique<MultibodyPlant<double>>(0);
  Parser parser(plant.get());
  parser.AddModelFromFile(FindResourceOrThrow(file_path));
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("iiwa_link_0"));
  plant->Finalize();
  return plant;
}

Eigen::Matrix<double, 3, 8> GenerateBoxVertices(const Eigen::Vector3d& size,
                                                const Eigen::Isometry3d& pose) {
  Eigen::Matrix<double, 3, 8> vertices;
  // clang-format off
  vertices << 1, 1, 1, 1, -1, -1, -1, -1,
              1, 1, -1, -1, 1, 1, -1, -1,
              1, -1, 1, -1, 1, -1, 1, -1;
  // clang-format on
  for (int i = 0; i < 3; ++i) {
    DRAKE_ASSERT(size(i) > 0);
    vertices.row(i) *= size(i) / 2;
  }
  vertices = pose.linear() * vertices +
             pose.translation() * Eigen::Matrix<double, 1, 8>::Ones();

  return vertices;
}

std::vector<std::shared_ptr<const ConvexPolytope>> GenerateIiwaLinkPolytopes(
    const MultibodyPlant<double>& iiwa) {
  std::vector<std::shared_ptr<const ConvexPolytope>> link_polytopes;
  const BodyIndex link7_idx = iiwa.GetBodyByName("iiwa_link_7").index();
  Eigen::Isometry3d link7_box_pose = Eigen::Isometry3d::Identity();
  link7_box_pose.translation() << 0, 0, 0.05;
  Eigen::Matrix<double, 3, 8> link7_pts =
      GenerateBoxVertices(Eigen::Vector3d(0.04, 0.14, 0.1), link7_box_pose);
  link_polytopes.push_back(
      std::make_shared<const ConvexPolytope>(link7_idx, link7_pts));
  return link_polytopes;
}

std::unique_ptr<MultibodyPlant<double>> ConstructDualArmIiwaPlant(
    const std::string& iiwa_sdf_name, const Eigen::Isometry3d& X_WL,
    const Eigen::Isometry3d& X_WR, ModelInstanceIndex* left_iiwa_instance,
    ModelInstanceIndex* right_iiwa_instance) {
  const std::string file_path =
      "drake/manipulation/models/iiwa_description/sdf/" + iiwa_sdf_name;
  auto plant = std::make_unique<MultibodyPlant<double>>(0);
  *left_iiwa_instance =
      Parser(plant.get())
          .AddModelFromFile(FindResourceOrThrow(file_path), "left_iiwa");
  *right_iiwa_instance =
      Parser(plant.get())
          .AddModelFromFile(FindResourceOrThrow(file_path), "right_iiwa");
  plant->WeldFrames(plant->world_frame(),
                    plant->GetFrameByName("iiwa_link_0", *left_iiwa_instance),
                    X_WL);
  plant->WeldFrames(plant->world_frame(),
                    plant->GetFrameByName("iiwa_link_0", *right_iiwa_instance),
                    X_WR);

  plant->Finalize();
  return plant;
}

IiwaTest::IiwaTest()
    : iiwa_(ConstructIiwaPlant("iiwa14_no_collision.sdf")),
      iiwa_tree_(internal::GetInternalTree(*iiwa_)),
      world_{iiwa_->world_body().index()} {
  for (int i = 0; i < 8; ++i) {
    iiwa_link_[i] =
        iiwa_->GetBodyByName("iiwa_link_" + std::to_string(i)).index();
    iiwa_joint_[i] =
        iiwa_tree_.get_topology().get_body(iiwa_link_[i]).inboard_mobilizer;
  }
}

namespace internal {
class MultibodyPlantPostureSource : public systems::VectorSystem<double> {
 public:
  MultibodyPlantPostureSource(const MultibodyPlant<double>& plant)
      : systems::VectorSystem<double>(0, plant.num_positions()),
        q_(plant.num_positions()) {}

  void SetPosture(const Eigen::Ref<const Eigen::VectorXd>& q) {
    DRAKE_DEMAND(q.size() == q_.size());
    q_ = q;
  }

 private:
  virtual void DoCalcVectorOutput(
      const systems::Context<double>&,
      const Eigen::VectorBlock<const VectorX<double>>&,
      const Eigen::VectorBlock<const VectorX<double>>&,
      Eigen::VectorBlock<VectorX<double>>* output) const override {
    *output = q_;
  }
  Eigen::VectorXd q_;
};
}  // namespace internal

MultibodyPlantVisualizer::MultibodyPlantVisualizer(
    const MultibodyPlant<double>& plant,
    std::unique_ptr<geometry::SceneGraph<double>> scene_graph) {
  systems::DiagramBuilder<double> builder;
  auto scene_graph_ptr = builder.AddSystem(std::move(scene_graph));
  posture_source_ =
      builder.AddSystem<internal::MultibodyPlantPostureSource>(plant);
  auto to_pose = builder.AddSystem<
      systems::rendering::MultibodyPositionToGeometryPose<double>>(plant);
  builder.Connect(posture_source_->get_output_port(),
                  to_pose->get_input_port());
  builder.Connect(
      to_pose->get_output_port(),
      scene_graph_ptr->get_source_pose_port(plant.get_source_id().value()));

  geometry::ConnectDrakeVisualizer(&builder, *scene_graph_ptr);
  diagram_ = builder.Build();
}

void MultibodyPlantVisualizer::VisualizePosture(
    const Eigen::Ref<const Eigen::VectorXd>& q) {
  posture_source_->SetPosture(q);
  systems::Simulator<double> simulator(*diagram_);
  simulator.set_publish_every_time_step(false);
  simulator.StepTo(0.1);
}

}  // namespace multibody
}  // namespace drake
