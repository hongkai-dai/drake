#include "drake/common/find_resource.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/geometry/optimization/dev/cspace_free_polytope.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace geometry {
namespace optimization {
class IiwaDiagram {
 public:
  explicit IiwaDiagram(int num_iiwas = 1)
      : meshcat_{std::make_shared<geometry::Meshcat>()} {
    systems::DiagramBuilder<double> builder;
    auto [plant, sg] = multibody::AddMultibodyPlantSceneGraph(&builder, 0.);
    plant_ = &plant;
    scene_graph_ = &sg;

    multibody::Parser parser(plant_);
    const std::string iiwa_file_path = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/sdf/"
        "iiwa14_coarse_collision.sdf");
    for (int iiwa_count = 0; iiwa_count < num_iiwas; ++iiwa_count) {
      const auto iiwa_instance = parser.AddModelFromFile(
          iiwa_file_path, fmt::format("iiwa{}", iiwa_count));
      plant_->WeldFrames(plant_->world_frame(),
                         plant_->GetFrameByName("iiwa_link_0", iiwa_instance));
      iiwa_instances_.push_back(iiwa_instance);

      const std::string schunk_file_path = FindResourceOrThrow(
          "drake/manipulation/models/wsg_50_description/sdf/"
          "schunk_wsg_50_welded_fingers.sdf");

      const multibody::Frame<double>& link7 =
          plant_->GetFrameByName("iiwa_link_7", iiwa_instance);
      const math::RigidTransformd X_7G(
          math::RollPitchYaw<double>(M_PI_2, 0, M_PI_2),
          Eigen::Vector3d(0, 0, 0.114));
      const auto wsg_instance = parser.AddModelFromFile(
          schunk_file_path, fmt::format("gripper{}", iiwa_count));
      wsg_instances_.push_back(wsg_instance);
      const auto& schunk_frame = plant_->GetFrameByName("body", wsg_instance);
      plant_->WeldFrames(link7, schunk_frame, X_7G);
      // SceneGraph should ignore the collision between any geometries on the
      // gripper, and between the gripper and link 6
      geometry::GeometrySet gripper_link6_geometries;
      auto add_gripper_geometries =
          [this, wsg_instance,
           &gripper_link6_geometries](const std::string& body_name) {
            const geometry::FrameId frame_id = plant_->GetBodyFrameIdOrThrow(
                plant_->GetBodyByName(body_name, wsg_instance).index());
            gripper_link6_geometries.Add(frame_id);
          };
      add_gripper_geometries("body");
      add_gripper_geometries("left_finger");
      add_gripper_geometries("right_finger");

      const geometry::FrameId link_6_frame_id = plant_->GetBodyFrameIdOrThrow(
          plant_->GetBodyByName("iiwa_link_6", iiwa_instance).index());
      const auto& inspector = scene_graph_->model_inspector();
      const std::vector<geometry::GeometryId> link_6_geometries =
          inspector.GetGeometries(link_6_frame_id, geometry::Role::kProximity);
      for (const auto geometry : link_6_geometries) {
        gripper_link6_geometries.Add(geometry);
      }

      scene_graph_->collision_filter_manager().Apply(
          geometry::CollisionFilterDeclaration().ExcludeWithin(
              gripper_link6_geometries));

      // Ignore collision between IIWA links.
      std::vector<geometry::GeometryId> iiwa_geometry_ids;
      for (const auto& body_index : plant_->GetBodyIndices(iiwa_instance)) {
        const std::vector<geometry::GeometryId> body_geometry_ids =
            plant_->GetCollisionGeometriesForBody(plant_->get_body(body_index));
        iiwa_geometry_ids.insert(iiwa_geometry_ids.end(),
                                 body_geometry_ids.begin(),
                                 body_geometry_ids.end());
      }
      scene_graph_->collision_filter_manager().Apply(
          geometry::CollisionFilterDeclaration().ExcludeWithin(
              geometry::GeometrySet(iiwa_geometry_ids)));
    }

    const std::string shelf_file_path = FindResourceOrThrow(
        "drake/geometry/optimization/dev/models/shelves.sdf");
    const auto shelf_instance =
        parser.AddModelFromFile(shelf_file_path, "shelves");
    const auto& shelf_frame =
        plant_->GetFrameByName("shelves_body", shelf_instance);
    const math::RigidTransformd X_WShelf(Eigen::Vector3d(0.8, 0, 0.4));
    plant_->WeldFrames(plant_->world_frame(), shelf_frame, X_WShelf);

    plant_->Finalize();
    //// If we have multiple IIWAs, then color them differently
    // for (int iiwa_count = 1;
    //     iiwa_count < static_cast<int>(iiwa_instances_.size()); ++iiwa_count)
    //     {
    //  double rgba_r = 0.5;
    //  double rgba_g = 0.4;
    //  double rgba_b = 0.2;
    //  double rgba_a = 0.7;
    //  if (iiwa_count > 1) {
    //    rgba_r = 0.3;
    //    rgba_g = 0.4;
    //    rgba_b = 0.6;
    //  }
    //  for (const auto& body_index :
    //       plant_->GetBodyIndices(iiwa_instances_[iiwa_count])) {
    //    SetDiffuse(*plant_, scene_graph_, body_index, std::nullopt, rgba_r,
    //               rgba_g, rgba_b, rgba_a);
    //  }
    //  for (const auto& body_index :
    //       plant_->GetBodyIndices(wsg_instances_[iiwa_count])) {
    //    SetDiffuse(*plant_, scene_graph_, body_index, std::nullopt, rgba_r,
    //               rgba_g, rgba_b, rgba_a);
    //  }
    //}

    // for (const auto& body_index : plant_->GetBodyIndices(shelf_instance)) {
    //  SetDiffuse(*plant_, scene_graph_, body_index, std::nullopt, 0.2, 0.2,
    //  0.2,
    //             1);
    //             }

    geometry::MeshcatVisualizerParams meshcat_params{};
    meshcat_params.role = geometry::Role::kIllustration;
    visualizer_ = &geometry::MeshcatVisualizer<double>::AddToBuilder(
        &builder, *scene_graph_, meshcat_, meshcat_params);
    diagram_ = builder.Build();
  }

  const systems::Diagram<double>& diagram() const { return *diagram_; }

  const multibody::MultibodyPlant<double>& plant() const { return *plant_; }

  const geometry::SceneGraph<double>& scene_graph() const {
    return *scene_graph_;
  }

  const std::vector<multibody::ModelInstanceIndex>& iiwa_instances() const {
    return iiwa_instances_;
  }

 private:
  std::unique_ptr<systems::Diagram<double>> diagram_;
  multibody::MultibodyPlant<double>* plant_;
  geometry::SceneGraph<double>* scene_graph_;
  std::shared_ptr<geometry::Meshcat> meshcat_;
  geometry::MeshcatVisualizer<double>* visualizer_;
  std::vector<multibody::ModelInstanceIndex> iiwa_instances_;
  std::vector<multibody::ModelInstanceIndex> wsg_instances_;
};

Eigen::VectorXd FindInitialPosture(
    const multibody::MultibodyPlant<double>& plant,
    systems::Context<double>* plant_context) {
  multibody::InverseKinematics ik(plant, plant_context);
  const auto& link7 = plant.GetFrameByName("iiwa_link_7");
  const auto& shelf = plant.GetFrameByName("shelves_body");
  ik.AddPositionConstraint(link7, Eigen::Vector3d::Zero(), shelf,
                           Eigen::Vector3d(-0.25, -0.2, -0.2),
                           Eigen::Vector3d(-.0, 0.2, 0.2));
  ik.AddMinimumDistanceConstraint(0.02);

  Eigen::Matrix<double, 7, 1> q_init;
  q_init << 0.1, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2;
  ik.get_mutable_prog()->SetInitialGuess(ik.q(), q_init);
  const auto result = solvers::Solve(ik.prog());
  if (!result.is_success()) {
    drake::log()->warn("Cannot find the posture\n");
  }
  return result.GetSolution(ik.q());
}

void BuildCandidateCspacePolytope(const Eigen::VectorXd q_free,
                                  Eigen::MatrixXd* C, Eigen::VectorXd* d) {
  const int C_rows = 26;
  C->resize(C_rows, 7);
  // Create arbitrary polytope normals.
  // clang-format off
  (*C) << 0.5, 0.3, 0.2, -0.1, -1, 0, 0.5,
          -0.1, 0.4, 0.2, 0.1, 0.5, -0.2, 0.3,
          0.4, 1.2, -0.3, 0.2, 0.1, 0.4, 0.5,
          -0.5, -2, -1.5, 0.3, 0.6, 0.1, -0.2,
          0.2, 0.1, -0.5, 0.3, 0.4, 1.4, 0.5,
          0.1, -0.5, 0.4, 1.5, -0.3, 0.2, 0.1,
          0.2, 0.3, 1.3, 0.2, -0.3, -0.5, -0.2,
          1.4, 0.1, -0.1, 0.2, -0.3, 0.1, 0.5,
          -1.1, 0.2, 0.3, -0.1, 0.5, 0.2, -0.1,
          0.2, -0.3, -1.2, 0.5, -0.3, 0.1, 0.3,
          0.2, -1.5, 0.1, 0.4, -0.3, -0.2, 0.6,
          0.1, 0.4, -0.2, 0.3, 0.9, -0.5, 0.8,
          -0.2, 0.3, -0.1, 0.8, -0.4, 0.2, 1.4,
          0.1, -0.2, 0.2, -0.3, 1.2, -0.3, 0.1,
          0.3, -0.1, 0.2, 0.5, -0.3, -2.1, 1.2,
          0.4, -0.3, 1.5, -0.3, 1.8, -0.1, 0.4,
          1.2, -0.3, 0.4, 0.8, 1.2, -0.4, -0.8,
          0.4, -0.2, 0.5, 1.4, 0.7, -0.2, -0.9,
          -0.1, 0.4, -0.2, 0.3, 1.5, 0.1, -0.6,
          -0.1, -0.3, 0.2, 1.1, -1.2, 1.3, 2.1,
          0.1, -0.4, 0.2, 1.3, 1.2, 0.3, -1.1,
          0.1, -1.4, 0.2, 0.3, 0.2, 0.3, -0.7,
          -0.3, -0.5, 0.4, -1.5, -0.2, 1.3, -2.1,
          0.5, 1.2, -0.3, 1.2, 0.5, -2.1, 0.4,
          -0.2, 2.1, 3.2, 1.5, 0.3, -1.8, 0.3,
          -2.1, 0.3, 2.0, 0.2, 1.3, -0.5, -0.6;
  // clang-format on
  for (int i = 0; i < C_rows; ++i) {
    C->row(i).normalize();
  }
  *d = (*C) * (q_free / 2).array().tan().matrix();
  if (!geometry::optimization::HPolyhedron(*C, *d).IsBounded()) {
    throw std::runtime_error("C*t <= d is not bounded");
  }
}

void SearchCspacePolytope() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  const IiwaDiagram iiwa_diagram{};
  auto diagram_context = iiwa_diagram.diagram().CreateDefaultContext();
  auto& plant_context =
      iiwa_diagram.plant().GetMyMutableContextFromRoot(diagram_context.get());
  const auto q0 = FindInitialPosture(iiwa_diagram.plant(), &plant_context);
  iiwa_diagram.plant().SetPositions(&plant_context, q0);

  Eigen::MatrixXd C_init;
  Eigen::VectorXd d_init;

  BuildCandidateCspacePolytope(q0, &C_init, &d_init);

  const Eigen::VectorXd q_star = Eigen::VectorXd::Zero(7);

  const CspaceFreePolytope dut(&(iiwa_diagram.plant()),
                               &(iiwa_diagram.scene_graph()),
                               SeparatingPlaneOrder::kAffine, q_star);

  CspaceFreePolytope::IgnoredCollisionPairs ignored_collision_pairs{};

  CspaceFreePolytope::BinarySearchOptions binary_search_options;
  binary_search_options.epsilon_max = 0.001;
  binary_search_options.epsilon_min = 0.;
  binary_search_options.max_iter = 2;

  const auto binary_search_result = dut.BinarySearch(
      ignored_collision_pairs, C_init, d_init, binary_search_options);
  DRAKE_DEMAND(binary_search_result.has_value());

  CspaceFreePolytope::BilinearAlternationOptions bilinear_alternation_options;
  bilinear_alternation_options.max_iter = 3;
  bilinear_alternation_options.convergence_tol = 1E-16;

  const auto bilinear_alternation_result = dut.SearchWithBilinearAlternation(
      ignored_collision_pairs, binary_search_result->C, binary_search_result->d,
      false /* search_margin */, bilinear_alternation_options);
  DRAKE_DEMAND(bilinear_alternation_result.has_value());
}

int DoMain() {
  SearchCspacePolytope();
  return 0;
}
}  // namespace optimization
}  // namespace geometry
}  // namespace drake

int main() { return drake::geometry::optimization::DoMain(); }
