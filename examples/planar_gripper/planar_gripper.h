#pragma once

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
namespace planar_gripper {

 class PlanarGripper : public systems::Diagram<double> {
  public:
   explicit PlanarGripper(double time_step = 1e-3,
                          bool use_position_control = true);

   /// Sets up the diagram using the planar brick.
   // TODO(rcory) Rename this to something like
   //   SetupPlanarBrickPlantAndFinalize()
   void SetupPlanarBrick(std::string orientation);

   // TODO(rcory) Rename this to something like FinalizeAndBuild()
   void Finalize();

   /// Returns a reference to the main plant responsible for the dynamics of
   /// the robot and the environment.  This can be used to, e.g., add
   /// additional elements into the world before calling Finalize().
   const multibody::MultibodyPlant<double>& get_multibody_plant() const {
     return *plant_;
   }

   /// Returns a mutable reference to the main plant responsible for the
   /// dynamics of the robot and the environment.  This can be used to, e.g.,
   /// add additional elements into the world before calling Finalize().
   multibody::MultibodyPlant<double>& get_mutable_multibody_plant() {
     return *plant_;
   }

   /// Returns a reference to the SceneGraph responsible for all of the geometry
   /// for the robot and the environment.  This can be used to, e.g., add
   /// additional elements into the world before calling Finalize().
   const geometry::SceneGraph<double>& get_scene_graph() const {
     return *scene_graph_;
   }

   /// Returns a mutable reference to the SceneGraph responsible for all of the
   /// geometry for the robot and the environment.  This can be used to, e.g.,
   /// add additional elements into the world before calling Finalize().
   geometry::SceneGraph<double>& get_mutable_scene_graph() { return *scene_graph_; }

   /// Return a reference to the plant used by the inverse dynamics controller
   /// (which contains only a model of the gripper).
   const multibody::MultibodyPlant<double>& get_control_plant() const {
     return *owned_control_plant_;
   }

   /// Get the number of joints in the gripper (only -- does not include the
   /// brick).
   int num_gripper_joints() const { return kNumJoints; }

   /// Convenience method for getting all of the joint angles of the gripper.
   /// This does not include the brick.
   VectorX<double> GetGripperPosition(
       const systems::Context<double>& diagram_context) const;

   /// Convenience method for setting all of the joint angles of the gripper.
   /// @p q must have size num_gripper_joints().
   /// @pre `state` must be the systems::State<double> object contained in
   /// `diagram_context`.
   void SetGripperPosition(const systems::Context<double>& diagram_context,
                           systems::State<double>* diagram_state,
                           const Eigen::Ref<const VectorX<double>>& q) const;

   /// Convenience method for setting all of the joint angles of gripper.
   /// @p q must have size num_gripper_joints().
   void SetGripperPosition(systems::Context<double>* diagram_context,
                        const Eigen::Ref<const VectorX<double>>& q) const {
     SetGripperPosition(
         *diagram_context, &diagram_context->get_mutable_state(), q);
   }

   /// Convenience method for getting all of the joint velocities of the Kuka
   // IIWA.  This does not include the gripper.
   VectorX<double> GetGripperVelocity(
       const systems::Context<double>& diagram_context) const;

   /// Convenience method for setting all of the joint velocities of the Gripper.
   /// @v must have size num_iiwa_joints().
   /// @pre `state` must be the systems::State<double> object contained in
   /// `diagram_context`.
   void SetGripperVelocity(const systems::Context<double>& diagram_context,
                        systems::State<double>* diagram_state,
                        const Eigen::Ref<const VectorX<double>>& v) const;

   /// Convenience method for setting all of the joint velocities of the gripper.
   /// @v must have size num_iiwa_joints().
   void SetGripperVelocity(systems::Context<double>* diagram_context,
                        const Eigen::Ref<const VectorX<double>>& v) const {
     SetGripperVelocity(*diagram_context, &diagram_context->get_mutable_state(), v);
   }

   /// Convenience method for setting all of the joint angles of the brick.
   /// @p q must have size 3 (y, z, theta).
   // TODO(rcory) Implement the const Context version that sets State instead.
   void SetBrickPosition(systems::Context<double>& diagram_context,
                           const Eigen::Ref<const VectorX<double>>& q);

   void AddFloor(MultibodyPlant<double>* plant,
                 const geometry::SceneGraph<double>& scene_graph);

   void set_brick_floor_penetration(double value) {
     if (is_plant_finalized_) {
       throw std::logic_error(
           "set_brick_floor_penetration must be called before "
           "SetupPlanarBrick().");
     }
     brick_floor_penetration_ = value;
   }

   void set_floor_coef_static_friction(double value) {
     if (is_plant_finalized_) {
       throw std::logic_error(
           "set_floor_ceof_static_friction must be called before "
           "SetupPlanarBrick().");
     }
     floor_coef_static_friction_ = value;
   }

   void set_floor_coef_kinetic_friction(double value) {
     if (is_plant_finalized_) {
       throw std::logic_error(
           "set_floor_coef_kinetic_friction must be called before "
           "SetupPlanarBrick().");
     }
     floor_coef_kinetic_friction_ = value;
   }

   void set_penetration_allowance(double value) {
     if (!is_plant_finalized_) {
       throw std::logic_error(
           "set_penetration_allowance must be called after "
           "SetupPlanarBrick().");
     }
     plant_->set_penetration_allowance(value);
   }

  private:
   // These are only valid until Finalize() is called.
   std::unique_ptr<multibody::MultibodyPlant<double>> owned_plant_;
   std::unique_ptr<geometry::SceneGraph<double>> owned_scene_graph_;
   
   // These are valid for the lifetime of this system.
   std::unique_ptr<multibody::MultibodyPlant<double>> owned_control_plant_;
   multibody::MultibodyPlant<double>* control_plant_;
   multibody::MultibodyPlant<double>* plant_;
   geometry::SceneGraph<double>* scene_graph_;
   bool is_plant_finalized_{false};
   bool is_diagram_finalized_{false};

   multibody::ModelInstanceIndex gripper_index_;
   multibody::ModelInstanceIndex brick_index_;

   bool use_position_control_{true};
   double brick_floor_penetration_{0};  // For the vertical case.
   double floor_coef_static_friction_{0};
   double floor_coef_kinetic_friction_{0};
 };



}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake