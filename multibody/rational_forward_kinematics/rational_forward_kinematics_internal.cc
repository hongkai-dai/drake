#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"

#include <algorithm>
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

  if (body_topology.parent_body.is_valid()) {
    auto it = visited->find(body_topology.parent_body);
    if (it == visited->end()) {
      auto child = std::make_unique<ReshuffledBody>(
          body_topology.parent_body, body,
          &(tree.get_mobilizer(body_topology.inboard_mobilizer)));
      body->children.push_back(std::move(child));
      visited->emplace_hint(it, body_topology.parent_body);
    }
  }
  for (BodyIndex body_index(0); body_index < plant.num_bodies(); ++body_index) {
    const internal::BodyTopology& body_topology_i =
        tree.get_topology().get_body(body_index);
    if (body_topology_i.parent_body.is_valid() &&
        body_topology_i.parent_body == body->body_index) {
      auto it = visited->find(body_index);
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

struct BodyOnPath {
  BodyOnPath(BodyIndex m_index, int m_distance_to_start, BodyOnPath* m_parent)
      : index(m_index),
        distance_to_start(m_distance_to_start),
        parent(m_parent) {}
  BodyIndex index;
  int distance_to_start;
  BodyOnPath* parent;
};

struct BodyOnPathHash {
 public:
  size_t operator()(const BodyOnPath& body) const {
    return std::hash<BodyIndex>()(body.index);
  }
};

struct BodyOnPathEqual {
 public:
  bool operator()(const BodyOnPath& body1, const BodyOnPath& body2) const {
    return body1.index == body2.index;
  };
};

std::vector<BodyIndex> FindShortestPath(const MultibodyPlant<double>& plant,
                                        BodyIndex start, BodyIndex end) {
  DRAKE_ASSERT(start.is_valid() && end.is_valid());
  const MultibodyTree<double>& tree = internal::GetInternalTree(plant);
  // Do a breadth first search in the tree.
  std::unordered_map<BodyIndex, std::unique_ptr<BodyOnPath>> visited_bodies;
  BodyIndex start_copy = start;
  visited_bodies.emplace(std::make_pair<BodyIndex, std::unique_ptr<BodyOnPath>>(
      std::move(start_copy), std::make_unique<BodyOnPath>(start, 0, nullptr)));
  std::queue<BodyOnPath*> queue_bodies;
  queue_bodies.push(visited_bodies[start].get());
  BodyOnPath* queue_front = queue_bodies.front();
  while (queue_front->index != end) {
    queue_bodies.pop();
    const BodyTopology& body_node =
        tree.get_topology().get_body(queue_front->index);
    if (body_node.parent_body.is_valid()) {
      BodyIndex parent = body_node.parent_body;
      visited_bodies.emplace(
          std::make_pair<BodyIndex, std::unique_ptr<BodyOnPath>>(
              std::move(parent),
              std::make_unique<BodyOnPath>(body_node.parent_body,
                                           queue_front->distance_to_start + 1,
                                           queue_front)));
      queue_bodies.emplace(visited_bodies[body_node.parent_body].get());
    }
    for (BodyIndex child : body_node.child_bodies) {
      if (child.is_valid()) {
        BodyIndex child_copy = child;
        visited_bodies.emplace(
            std::make_pair<BodyIndex, std::unique_ptr<BodyOnPath>>(
                std::move(child_copy),
                std::make_unique<BodyOnPath>(
                    child, queue_front->distance_to_start + 1, queue_front)));
        queue_bodies.emplace(visited_bodies[child].get());
      }
    }
    queue_front = queue_bodies.front();
  }
  // Backup the path
  std::vector<BodyIndex> path;
  BodyOnPath* path_body = queue_front;
  while (path_body->index != start) {
    path.push_back(path_body->index);
    path_body = path_body->parent;
  }
  path.push_back(start);
  std::reverse(path.begin(), path.end());
  return path;
}
}  // namespace internal
}  // namespace multibody
}  // namespace drake
