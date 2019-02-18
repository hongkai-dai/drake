#include <memory>

#include "drake/common/find_resource.h"
#include "drake/manipulation/dev/remote_tree_viewer_wrapper.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"

namespace drake {
namespace multibody {

void VisualizeBodyPoint(manipulation::dev::RemoteTreeViewerWrapper* viewer,
                        const MultibodyPlant<double>& plant,
                        const systems::Context<double>& context,
                        BodyIndex body_index,
                        const Eigen::Ref<const Eigen::Vector3d>& p_BQ,
                        double radius, const Eigen::Vector4d& color,
                        const std::string& name) {
  Eigen::Vector3d p_WQ;
  plant.CalcPointsPositions(context, plant.get_body(body_index).body_frame(),
                            p_BQ, plant.world_frame(), &p_WQ);
  Eigen::Isometry3d X_WQ = Eigen::Isometry3d::Identity();
  X_WQ.translation() = p_WQ;
  viewer->PublishGeometry(DrakeShapes::Sphere(radius), X_WQ, color, {name});
}

int DoMain() {
  const std::string file_path =
      "drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf";
  auto plant = std::make_unique<MultibodyPlant<double>>();
  auto scene_graph = std::make_unique<geometry::SceneGraph<double>>();
  plant->RegisterAsSourceForSceneGraph(scene_graph.get());
  Parser(plant.get()).AddModelFromFile(FindResourceOrThrow(file_path));
  Parser(plant.get())
      .AddModelFromFile(
          FindResourceOrThrow("drake/manipulation/models/wsg_50_description/"
                              "sdf/schunk_wsg_50_fixed_joint.sdf"));
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("iiwa_link_0"));
  // weld the schunk gripper to iiwa link 7.
  Eigen::Isometry3d X_7S =
      Eigen::Translation3d(0, 0, 0.1) *
      Eigen::AngleAxisd(-21.0 / 180 * M_PI, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitX()) *
      Eigen::Isometry3d::Identity();
  plant->WeldFrames(plant->GetFrameByName("iiwa_link_7"),
                    plant->GetFrameByName("body"), X_7S);

  Isometry3<double> X_WTableTop = Isometry3<double>::Identity();
  X_WTableTop.translation() << 0, 0.5, 0.5;
  geometry::Box table_top(0.4, 0.4, 0.2);
  plant->RegisterCollisionGeometry(plant->world_body(), X_WTableTop, table_top,
                                   "table_top", CoulombFriction<double>(),
                                   scene_graph.get());

  plant->Finalize(scene_graph.get());

  MultibodyPlantVisualizer visualizer(*plant, std::move(scene_graph));
  Eigen::Matrix<double, 7, 1> q;
  q << 0.4, 0.4, 0.4, -0.4, 0.4, 0.4, 0.4;
  // This is only for visualizing a sampled configuration in the verified
  // collision free box in the configuration space.
  // double delta = 0.271484;
  // q(0) -= delta;
  // q(1) += delta;
  // q(2) -= delta;
  // q(3) -= delta;
  // q(4) -= delta;
  // q(5) += delta;

  visualizer.VisualizePosture(q);

  // Now add the link points to represent collision.
  auto context = plant->CreateDefaultContext();
  plant->SetPositions(context.get(), q);
  manipulation::dev::RemoteTreeViewerWrapper viewer;

  std::array<BodyIndex, 8> iiwa_link;
  for (int i = 0; i < 8; ++i) {
    iiwa_link[i] =
        plant->GetBodyByName("iiwa_link_" + std::to_string(i)).index();
  }
  // Schunk gripper points.
  Eigen::Matrix<double, 3, 8> p_SV;
  p_SV.col(0) << -0.065, -0.035, 0.02;
  p_SV.col(1) << 0.065, -0.035, 0.02;
  p_SV.col(2) << -0.065, -0.035, -0.02;
  p_SV.col(3) << 0.065, -0.035, -0.02;
  p_SV.col(4) << -0.065, 0.105, 0.02;
  p_SV.col(5) << 0.065, 0.105, 0.02;
  p_SV.col(6) << -0.065, 0.105, -0.02;
  p_SV.col(7) << 0.065, 0.105, -0.02;
  Eigen::Matrix<double, 3, 8> p_7V = X_7S * p_SV;
  for (int i = 0; i < p_7V.cols(); ++i) {
    VisualizeBodyPoint(&viewer, *plant, *context, iiwa_link[7], p_7V.col(i),
                       0.01, {1, 0, 0, 1}, "gripper_point" + std::to_string(i));
  }
  std::vector<std::shared_ptr<const ConvexPolytope>> link_polytopes;
  link_polytopes.push_back(
      std::make_shared<ConvexPolytope>(iiwa_link[7], p_7V));

  // iiwa_link6 points.
  Eigen::Matrix<double, 3, 14> p_6V;
  p_6V.col(0) << 0.03, -0.05, 0.06;
  p_6V.col(1) << -0.03, -0.05, 0.06;
  p_6V.col(2) << 0.04, -0.09, 0.02;
  p_6V.col(3) << -0.04, -0.09, 0.02;
  p_6V.col(4) << -0.03, -0.05, -0.09;
  p_6V.col(5) << 0.03, -0.05, -0.09;
  p_6V.col(6) << 0.03, 0.05, -0.09;
  p_6V.col(7) << -0.03, 0.05, -0.09;
  p_6V.col(8) << -0.07, 0, 0;
  p_6V.col(9) << 0.07, 0, 0;
  p_6V.col(10) << 0.03, 0.12, 0.03;
  p_6V.col(11) << 0.03, 0.12, -0.03;
  p_6V.col(12) << -0.03, 0.12, 0.03;
  p_6V.col(13) << -0.03, 0.12, -0.03;
  for (int i = 0; i < p_6V.cols(); ++i) {
    VisualizeBodyPoint(&viewer, *plant, *context, iiwa_link[6], p_6V.col(i),
                       0.01, {1, 0, 0, 1}, "link6_pt" + std::to_string(i));
  }
  link_polytopes.push_back(
      std::make_shared<ConvexPolytope>(iiwa_link[6], p_6V));

  Eigen::Matrix<double, 3, 11> p_5V1;
  p_5V1.col(0) << 0.05, 0.07, -0.05;
  p_5V1.col(1) << 0.05, -0.05, -0.05;
  p_5V1.col(2) << -0.05, 0.07, -0.05;
  p_5V1.col(3) << -0.05, -0.05, -0.05;
  p_5V1.col(4) << 0.05, 0.05, 0.08;
  p_5V1.col(5) << 0.05, -0.05, 0.07;
  p_5V1.col(6) << -0.05, 0.05, 0.08;
  p_5V1.col(7) << -0.05, -0.05, 0.07;
  p_5V1.col(8) << 0.04, 0.08, 0.15;
  p_5V1.col(9) << -0.04, 0.08, 0.15;
  for (int i = 0; i < p_5V1.cols(); ++i) {
    VisualizeBodyPoint(&viewer, *plant, *context, iiwa_link[5], p_5V1.col(i),
                       0.01, {1, 0, 0, 1}, "link5_V1_pt" + std::to_string(i));
  }
  link_polytopes.push_back(
      std::make_shared<ConvexPolytope>(iiwa_link[5], p_5V1));

  Eigen::Matrix<double, 3, 8> p_4V;
  p_4V.col(0) << 0.04, -0.02, 0.11;
  p_4V.col(1) << -0.04, -0.02, 0.11;
  p_4V.col(2) << 0.06, -0.05, 0;
  p_4V.col(3) << -0.06, -0.05, 0;
  p_4V.col(4) << 0.06, 0.12, 0;
  p_4V.col(5) << -0.06, 0.12, 0;
  p_4V.col(6) << 0.05, 0.12, 0.09;
  p_4V.col(7) << -0.05, 0.12, 0.09;
  for (int i = 0; i < p_4V.cols(); ++i) {
    VisualizeBodyPoint(&viewer, *plant, *context, iiwa_link[4], p_4V.col(i),
                       0.01, {1, 0, 0, 1}, "link4_pt" + std::to_string(i));
  }
  link_polytopes.push_back(
      std::make_shared<ConvexPolytope>(iiwa_link[4], p_4V));

  // Add a table.
  Eigen::Isometry3d X_WT = Eigen::Isometry3d::Identity();
  X_WT.translation() << 0.5, 0, 0.25;
  Eigen::Vector3d table_size(0.3, 0.6, 0.5);
  viewer.PublishGeometry(DrakeShapes::Box(table_size), X_WT, {0, 0, 1, 1},
                         {"table"});

  std::vector<std::shared_ptr<const ConvexPolytope>> obstacles;
  obstacles.push_back(std::make_shared<ConvexPolytope>(
      plant->world_body().index(), GenerateBoxVertices(table_size, X_WT)));

  // Add a box on the table.
  Eigen::Isometry3d X_WBox1 = Eigen::Isometry3d::Identity();
  X_WBox1.translation() << 0.5, 0, 0.57;
  Eigen::Vector3d box1_size(0.15, 0.2, 0.14);
  viewer.PublishGeometry(DrakeShapes::Box(box1_size), X_WBox1, {0, 0, 1, 1},
                         {"box1"});
  obstacles.push_back(std::make_shared<ConvexPolytope>(
      plant->world_body().index(), GenerateBoxVertices(box1_size, X_WBox1)));

  ConfigurationSpaceCollisionFreeRegion dut(*plant, link_polytopes, obstacles,
                                            SeparatingPlaneOrder::kAffine);

  double rho = dut.FindLargestBoxThroughBinarySearch(
      q, {}, Eigen::VectorXd::Constant(7, -1), Eigen::VectorXd::Constant(7, 1),
      0, 1, 0.01);
  std::cout << "rho = " << rho
            << ", corresponding to angle (deg): " << rho / M_PI * 180.0 << "\n";

  return 0;
}

}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
