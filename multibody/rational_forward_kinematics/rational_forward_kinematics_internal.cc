#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"

#include <queue>

namespace drake {
namespace multibody {
namespace internal {

void AddChildrenToReshuffledBody(const MultibodyPlant<double>& plant,
                                 ReshuffledBody* body,
                                 std::unordered_set<BodyIndex>* visited) {
  const internal::MultibodyTree<double>& tree =
      internal::GetInternalTree(plant);
  const internal::BodyTopology& body_topology =
      tree.get_topology().get_body(body->body_index);
  auto it = visited->find(body_topology.parent_body);
  if (body_topology.parent_body.is_valid() && it == visited->end()) {
    auto child = std::make_unique<ReshuffledBody>(
        body_topology.parent_body, body,
        &(tree.get_mobilizer(body_topology.inboard_mobilizer)));
    body->children.push_back(std::move(child));
    visited->emplace_hint(it, body_topology.parent_body);
  }
  for (BodyIndex body_index(0); body_index < plant.num_bodies(); ++body_index) {
    const internal::BodyTopology& body_topology_i =
        tree.get_topology().get_body(body_index);
    if (body_topology_i.parent_body.is_valid() &&
        body_topology_i.parent_body == body->body_index) {
      it = visited->find(body_index);
      if (it == visited->end()) {
        body->children.emplace_back(new ReshuffledBody(
            body_index, body,
            &(tree.get_mobilizer(body_topology_i.inboard_mobilizer))));
        visited->emplace_hint(it, body_index);
      }
    }
  }
}

void ReshuffleKinematicsTree(const MultibodyPlant<double>& plant,
                             ReshuffledBody* root) {
  DRAKE_DEMAND(root->parent == nullptr);
  DRAKE_DEMAND(root->mobilizer == nullptr);
  root->children.clear();
  std::unordered_set<BodyIndex> visited;
  visited.emplace(root->body_index);
  std::queue<ReshuffledBody*> queue_bodies;
  queue_bodies.push(root);
  while (!queue_bodies.empty()) {
    ReshuffledBody* reshuffled_body = queue_bodies.front();
    queue_bodies.pop();
    AddChildrenToReshuffledBody(plant, reshuffled_body, &visited);
    if (!reshuffled_body->children.empty()) {
      for (int i = 0; i < static_cast<int>(reshuffled_body->children.size());
           ++i) {
        queue_bodies.push(reshuffled_body->children[i].get());
      }
    }
  }
}
}
}  // namespace multibody
}  // namespace drake
