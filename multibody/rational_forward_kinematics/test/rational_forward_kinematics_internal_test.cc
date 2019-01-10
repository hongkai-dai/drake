#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"

#include <gtest/gtest.h>

#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"

namespace drake {
namespace multibody {
namespace internal {
const Mobilizer<double>* GetInboardMobilizer(const MultibodyTree<double>& tree,
                                             BodyIndex body_index) {
  return &(tree.get_mobilizer(
      tree.get_topology().get_body(body_index).inboard_mobilizer));
}

// @param was_parent if this child was the parent before reshuffling. This
// determines if the mobilizer between the child and the parent was the inboard
// mobilizer of the child or the parent.
void CheckChildInReshuffledBody(const MultibodyTree<double>& tree,
                                const ReshuffledBody& reshuffled_body,
                                int child_index,
                                BodyIndex child_body_index_expected) {
  EXPECT_EQ(reshuffled_body.children[child_index]->body_index,
            child_body_index_expected);
  EXPECT_EQ(reshuffled_body.children[child_index]->parent->body_index,
            reshuffled_body.body_index);
  BodyIndex inboard_mobilizer_body;
  if (tree.get_topology().get_body(reshuffled_body.body_index).parent_body ==
      child_body_index_expected) {
    inboard_mobilizer_body = reshuffled_body.body_index;
  } else if (tree.get_topology()
                 .get_body(child_body_index_expected)
                 .parent_body == reshuffled_body.body_index) {
    inboard_mobilizer_body = child_body_index_expected;
  }

  EXPECT_EQ(reshuffled_body.children[child_index]->mobilizer,
            GetInboardMobilizer(tree, inboard_mobilizer_body));
}

GTEST_TEST(TestAddChildrenToReshuffledBody, IiwaReshuffleAtWorld) {
  auto iiwa = ConstructIiwaPlant("iiwa14_no_collision.sdf");
  const auto& iiwa_tree = GetInternalTree(*iiwa);
  const BodyIndex world_index = iiwa->world_body().index();
  ReshuffledBody reshuffled_world(world_index, nullptr, nullptr);
  std::unordered_set<BodyIndex> visited;
  visited.emplace(world_index);
  // Add children of the world.
  AddChildrenToReshuffledBody(*iiwa, &reshuffled_world, &visited);
  const BodyIndex iiwa_link_0 = iiwa->GetBodyByName("iiwa_link_0").index();
  EXPECT_EQ(reshuffled_world.children.size(), 1);
  CheckChildInReshuffledBody(iiwa_tree, reshuffled_world, 0, iiwa_link_0);
  EXPECT_EQ(visited.size(), 2);
  EXPECT_EQ(visited, std::unordered_set<BodyIndex>({world_index, iiwa_link_0}));

  ReshuffledBody* reshuffled_link_0 = reshuffled_world.children[0].get();

  // Continue to add children of link 0
  AddChildrenToReshuffledBody(*iiwa, reshuffled_link_0, &visited);
  EXPECT_EQ(reshuffled_world.children.size(), 1);
  EXPECT_EQ(reshuffled_link_0->children.size(), 1);
  const BodyIndex iiwa_link_1 = iiwa->GetBodyByName("iiwa_link_1").index();
  CheckChildInReshuffledBody(iiwa_tree, *reshuffled_link_0, 0, iiwa_link_1);
  EXPECT_EQ(visited.size(), 3);
  EXPECT_EQ(visited, std::unordered_set<BodyIndex>(
                         {world_index, iiwa_link_0, iiwa_link_1}));

  // Continue to add children of link 1
  ReshuffledBody* reshuffled_link_1 = reshuffled_link_0->children[0].get();
  AddChildrenToReshuffledBody(*iiwa, reshuffled_link_1, &visited);
  EXPECT_EQ(reshuffled_link_1->children.size(), 1);
  const BodyIndex iiwa_link_2 = iiwa->GetBodyByName("iiwa_link_2").index();
  CheckChildInReshuffledBody(iiwa_tree, *reshuffled_link_1, 0, iiwa_link_2);
  EXPECT_EQ(reshuffled_link_1->children[0]->body_index, iiwa_link_2);
  EXPECT_EQ(reshuffled_link_1->children[0]->parent, reshuffled_link_1);
  EXPECT_EQ(reshuffled_link_1->children[0]->mobilizer,
            GetInboardMobilizer(iiwa_tree, iiwa_link_2));
  EXPECT_EQ(visited.size(), 4);
  EXPECT_EQ(visited, std::unordered_set<BodyIndex>(
                         {world_index, iiwa_link_0, iiwa_link_1, iiwa_link_2}));
}

GTEST_TEST(TestAddChildrenToReshuffledBody, IiwaReshufflAtLink3) {
  auto iiwa = ConstructIiwaPlant("iiwa14_no_collision.sdf");
  const auto& iiwa_tree = GetInternalTree(*iiwa);
  const BodyIndex iiwa_link_3 = iiwa->GetBodyByName("iiwa_link_3").index();
  ReshuffledBody reshuffled_link_3(iiwa_link_3, nullptr, nullptr);
  std::unordered_set<BodyIndex> visited;
  visited.insert(iiwa_link_3);
  AddChildrenToReshuffledBody(*iiwa, &reshuffled_link_3, &visited);
  EXPECT_EQ(reshuffled_link_3.children.size(), 2);
  const BodyIndex iiwa_link_2 = iiwa->GetBodyByName("iiwa_link_2").index();
  CheckChildInReshuffledBody(iiwa_tree, reshuffled_link_3, 0, iiwa_link_2);
  const BodyIndex iiwa_link_4 = iiwa->GetBodyByName("iiwa_link_4").index();
  CheckChildInReshuffledBody(iiwa_tree, reshuffled_link_3, 1, iiwa_link_4);
  EXPECT_EQ(visited.size(), 3);
  EXPECT_EQ(visited, std::unordered_set<BodyIndex>(
                         {iiwa_link_2, iiwa_link_3, iiwa_link_4}));

  // Continue to add children of iiwa_link_2
  ReshuffledBody* reshuffled_link_2 = reshuffled_link_3.children[0].get();
  AddChildrenToReshuffledBody(*iiwa, reshuffled_link_2, &visited);
  EXPECT_EQ(reshuffled_link_2->children.size(), 1);
  EXPECT_EQ(reshuffled_link_2->parent->body_index, iiwa_link_3);
  const BodyIndex iiwa_link_1 = iiwa->GetBodyByName("iiwa_link_1").index();
  CheckChildInReshuffledBody(iiwa_tree, *reshuffled_link_2, 0, iiwa_link_1);
  EXPECT_EQ(visited.size(), 4);
  EXPECT_EQ(visited, std::unordered_set<BodyIndex>(
                         {iiwa_link_1, iiwa_link_2, iiwa_link_3, iiwa_link_4}));

  // Continue to add children of iiwa_link_1
  ReshuffledBody* reshuffled_link_1 = reshuffled_link_2->children[0].get();
  AddChildrenToReshuffledBody(*iiwa, reshuffled_link_1, &visited);
  EXPECT_EQ(reshuffled_link_1->children.size(), 1);
  const BodyIndex iiwa_link_0 = iiwa->GetBodyByName("iiwa_link_0").index();
  CheckChildInReshuffledBody(iiwa_tree, *reshuffled_link_1, 0, iiwa_link_0);
  EXPECT_EQ(visited.size(), 5);
  EXPECT_EQ(visited, std::unordered_set<BodyIndex>({iiwa_link_0, iiwa_link_1,
                                                    iiwa_link_2, iiwa_link_3,
                                                    iiwa_link_4}));

  // Continue to add children of iiwa_link_0
  ReshuffledBody* reshuffled_link_0 = reshuffled_link_1->children[0].get();
  AddChildrenToReshuffledBody(*iiwa, reshuffled_link_0, &visited);
  EXPECT_EQ(reshuffled_link_0->children.size(), 1);
  const BodyIndex world = iiwa->world_body().index();
  CheckChildInReshuffledBody(iiwa_tree, *reshuffled_link_0, 0, world);
  EXPECT_EQ(visited.size(), 6);
  EXPECT_EQ(visited, std::unordered_set<BodyIndex>({world, iiwa_link_0,
                                                    iiwa_link_1, iiwa_link_2,
                                                    iiwa_link_3, iiwa_link_4}));

  // Continue to add children of world.
  ReshuffledBody* reshuffled_world = reshuffled_link_0->children[0].get();
  AddChildrenToReshuffledBody(*iiwa, reshuffled_world, &visited);
  EXPECT_TRUE(reshuffled_world->children.empty());
  EXPECT_EQ(visited.size(), 6);
  EXPECT_EQ(visited, std::unordered_set<BodyIndex>({world, iiwa_link_0,
                                                    iiwa_link_1, iiwa_link_2,
                                                    iiwa_link_3, iiwa_link_4}));
}
}  // namespace internal
}  // namespace multibody
}  // namespace drake
