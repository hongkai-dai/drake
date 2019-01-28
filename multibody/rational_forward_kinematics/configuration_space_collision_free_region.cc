#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace multibody {
using symbolic::RationalFunction;

ConfigurationSpaceCollisionFreeRegion::ConfigurationSpaceCollisionFreeRegion(
    const MultibodyPlant<double>& plant,
    const std::vector<std::shared_ptr<const ConvexPolytope>>& link_polytopes,
    const std::vector<std::shared_ptr<const ConvexPolytope>>& obstacles)
    : rational_forward_kinematics_(plant), obstacles_{obstacles} {
  // First group the link polytopes by the attached link.
  for (const auto& link_polytope : link_polytopes) {
    DRAKE_DEMAND(link_polytope->body_index() != plant.world_body().index());
    const auto it = link_polytopes_.find(link_polytope->body_index());
    if (it == link_polytopes_.end()) {
      link_polytopes_.emplace_hint(
          it, std::make_pair(link_polytope->body_index(),
                             std::vector<std::shared_ptr<const ConvexPolytope>>(
                                 {link_polytope})));
    } else {
      it->second.push_back(link_polytope);
    }
  }
  // Now create the separation planes.
  // By default, we only consider the pairs between a link polytope and a world
  // obstacle.
  separation_planes_.reserve(link_polytopes.size() * obstacles.size());
  for (const auto& obstacle : obstacles_) {
    DRAKE_DEMAND(obstacle->body_index() == plant.world_body().index());
    for (const auto& link_polytope_pairs : link_polytopes_) {
      for (const auto& link_polytope : link_polytope_pairs.second) {
        Vector3<symbolic::Variable> a;
        for (int i = 0; i < 3; ++i) {
          a(i) = symbolic::Variable(
              "a" + std::to_string(separation_planes_.size() * 3 + i));
        }
        // Expressed body is the middle link in the chain from the world to
        // the link_polytope.
        separation_planes_.emplace_back(
            a, link_polytope, obstacle,
            internal::FindBodyInTheMiddleOfChain(plant, obstacle->body_index(),
                                                 link_polytope->body_index()));
        map_polytopes_to_separation_planes_.emplace(
            std::make_pair(link_polytope->get_id(), obstacle->get_id()),
            &(separation_planes_[separation_planes_.size() - 1]));
      }
    }
  }
}

bool ConfigurationSpaceCollisionFreeRegion::IsLinkPairCollisionIgnored(
    ConvexGeometry::Id id1, ConvexGeometry::Id id2,
    const FilteredCollisionPairs& filtered_collision_pairs) const {
  return filtered_collision_pairs.count(std::make_pair(id1, id2)) > 0 ||
         filtered_collision_pairs.count(std::make_pair(id2, id1)) > 0;
}

std::vector<LinkVertexOnPlaneSideRational>
ConfigurationSpaceCollisionFreeRegion::GenerateLinkOnOneSideOfPlaneRationals(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs) const {
  auto context = rational_forward_kinematics_.plant().CreateDefaultContext();
  rational_forward_kinematics_.plant().SetPositions(context.get(), q_star);

  const BodyIndex world_index =
      rational_forward_kinematics_.plant().world_body().index();
  std::vector<LinkVertexOnPlaneSideRational> rationals;
  for (const auto& body_to_polytopes : link_polytopes_) {
    const BodyIndex expressed_body_index = internal::FindBodyInTheMiddleOfChain(
        rational_forward_kinematics_.plant(), world_index,
        body_to_polytopes.first);
    const symbolic::Variables middle_to_link_variables(
        rational_forward_kinematics_.FindTOnPath(expressed_body_index,
                                                 body_to_polytopes.first));
    const symbolic::Variables world_to_middle_variables(
        rational_forward_kinematics_.FindTOnPath(world_index,
                                                 expressed_body_index));
    // Compute the pose of the link (B) and the world (W) in the expressed link
    // (A).
    // TODO(hongkai.dai): save the poses to an unordered set.
    const RationalForwardKinematics::Pose<symbolic::Polynomial> X_AB =
        rational_forward_kinematics_.CalcLinkPoseAsMultilinearPolynomial(
            q_star, body_to_polytopes.first, expressed_body_index);
    const RationalForwardKinematics::Pose<symbolic::Polynomial> X_AW =
        rational_forward_kinematics_.CalcLinkPoseAsMultilinearPolynomial(
            q_star, world_index, expressed_body_index);
    for (const auto& link_polytope : body_to_polytopes.second) {
      for (const auto& obstacle : obstacles_) {
        if (!IsLinkPairCollisionIgnored(link_polytope->get_id(),
                                        obstacle->get_id(),
                                        filtered_collision_pairs)) {
          const auto& a_A = map_polytopes_to_separation_planes_
                                .find(std::make_pair(link_polytope->get_id(),
                                                     obstacle->get_id()))
                                ->second->a;
          Eigen::Vector3d p_AC;
          rational_forward_kinematics_.plant().CalcPointsPositions(
              *context, rational_forward_kinematics_.plant()
                            .get_body(obstacle->body_index())
                            .body_frame(),
              obstacle->p_BC(), rational_forward_kinematics_.plant()
                                    .get_body(expressed_body_index)
                                    .body_frame(),
              &p_AC);
          const std::vector<LinkVertexOnPlaneSideRational>
              positive_side_rationals =
                  GenerateLinkOnOneSideOfPlaneRationalFunction(
                      rational_forward_kinematics_, link_polytope, X_AB, a_A,
                      p_AC, PlaneSide::kPositive);
          const std::vector<LinkVertexOnPlaneSideRational>
              negative_side_rationals =
                  GenerateLinkOnOneSideOfPlaneRationalFunction(
                      rational_forward_kinematics_, obstacle, X_AW, a_A, p_AC,
                      PlaneSide::kNegative);
          // I cannot use "insert" function to append vectors, since
          // LinkVertexOnPlaneSideRational contains const members, hence it does
          // not have an assignment operator.
          std::copy(positive_side_rationals.begin(),
                    positive_side_rationals.end(),
                    std::back_inserter(rationals));
          std::copy(negative_side_rationals.begin(),
                    negative_side_rationals.end(),
                    std::back_inserter(rationals));
        }
      }
    }
  }
  return rationals;
}

namespace {
struct KinematicsChain {
  KinematicsChain(BodyIndex m_end1, BodyIndex m_end2) {
    if (m_end1 < m_end2) {
      end1 = m_end1;
      end2 = m_end2;
    } else {
      end1 = m_end2;
      end2 = m_end1;
    }
  }

  bool operator==(const KinematicsChain& other) const {
    return end1 == other.end1 && end2 == other.end2;
  }
  BodyIndex end1;
  BodyIndex end2;
};

struct KinematicsChainHash {
  size_t operator()(const KinematicsChain& c) const {
    return c.end1 * 100 + c.end2;
  }
};
}  // namespace

std::unique_ptr<solvers::MathematicalProgram>
ConfigurationSpaceCollisionFreeRegion::ConstructProgramToVerifyCollisionFreeBox(
    const std::vector<LinkVertexOnPlaneSideRational>& rationals,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const VerificationOption& verification_option) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  // Add t as indeterminates
  const auto& t = rational_forward_kinematics_.t();
  prog->AddIndeterminates(t);
  // Add separation planes as decision variables
  for (const auto& separation_plane : separation_planes_) {
    if (!IsLinkPairCollisionIgnored(
            separation_plane.positive_side_polytope->get_id(),
            separation_plane.negative_side_polytope->get_id(),
            filtered_collision_pairs)) {
      prog->AddDecisionVariables(separation_plane.a);
    }
  }

  // Now build the polynomials t - t_lower and t_upper - t
  DRAKE_DEMAND(t_lower.size() == t_upper.size());
  // maps t(i) to (t(i) - t_lower(i), t_upper(i) - t(i))
  std::unordered_map<symbolic::Variable::Id,
                     std::pair<symbolic::Polynomial, symbolic::Polynomial>>
      map_t_to_box_bounds;
  map_t_to_box_bounds.reserve(t.size());
  const symbolic::Monomial monomial_one{};
  for (int i = 0; i < t.size(); ++i) {
    map_t_to_box_bounds.emplace(
        rational_forward_kinematics_.t()(i).get_id(),
        std::make_pair(symbolic::Polynomial({{symbolic::Monomial(t(i)), 1},
                                             {monomial_one, -t_lower(i)}}),
                       symbolic::Polynomial({{symbolic::Monomial(t(i)), -1},
                                             {monomial_one, t_upper(i)}})));
  }

  // map the kinematics chain to (t_chain, monomial_basis), where t_chain are
  // t on the kinematics chain.
  std::unordered_map<KinematicsChain, std::pair<VectorX<symbolic::Variable>,
                                                VectorX<symbolic::Monomial>>,
                     KinematicsChainHash>
      map_kinematics_chain_to_monomial_basis;

  for (const auto& rational : rationals) {
    // First check if the monomial basis for this kinematics chain has been
    // computed.
    const KinematicsChain rational_kinematics_chain(
        rational.link_polytope->body_index(), rational.expressed_body_index);
    const auto it =
        map_kinematics_chain_to_monomial_basis.find(rational_kinematics_chain);
    VectorX<symbolic::Variable> t_chain;
    VectorX<symbolic::Monomial> monomial_basis_chain;
    if (it == map_kinematics_chain_to_monomial_basis.end()) {
      t_chain = rational_forward_kinematics_.FindTOnPath(
          rational.link_polytope->body_index(), rational.expressed_body_index);
      monomial_basis_chain =
          GenerateMonomialBasisWithOrderUpToOne(symbolic::Variables(t_chain));
      map_kinematics_chain_to_monomial_basis.emplace_hint(
          it, std::make_pair(rational_kinematics_chain,
                             std::make_pair(t_chain, monomial_basis_chain)));
    } else {
      t_chain = it->second.first;
      monomial_basis_chain = it->second.second;
    }
    VectorX<symbolic::Polynomial> t_minus_t_lower(t_chain.size());
    VectorX<symbolic::Polynomial> t_upper_minus_t(t_chain.size());
    for (int i = 0; i < t_chain.size(); ++i) {
      auto it_t = map_t_to_box_bounds.find(t_chain(i).get_id());
      t_minus_t_lower(i) = it_t->second.first;
      t_upper_minus_t(i) = it_t->second.second;
    }
    // Now add the constraint that t_lower <= t <= t_upper implies the rational
    // being nonnegative.
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        prog.get(), rational.rational, t_minus_t_lower, t_upper_minus_t,
        monomial_basis_chain, verification_option);
  }

  return prog;
}

double ConfigurationSpaceCollisionFreeRegion::FindLargestBoxThroughBinarySearch(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const Eigen::Ref<const Eigen::VectorXd>& negative_delta_t,
    const Eigen::Ref<const Eigen::VectorXd>& positive_delta_t,
    double rho_lower_initial, double rho_upper_initial, double rho_tolerance,
    const VerificationOption& verification_option) const {
  DRAKE_DEMAND(negative_delta_t.size() == positive_delta_t.size());
  DRAKE_DEMAND((negative_delta_t.array() <= 0).all());
  DRAKE_DEMAND((positive_delta_t.array() >= 0).all());
  DRAKE_DEMAND(rho_lower_initial >= 0);
  DRAKE_DEMAND(rho_lower_initial <= rho_upper_initial);
  DRAKE_DEMAND(rho_tolerance > 0);
  Eigen::VectorXd q_upper(rational_forward_kinematics_.plant().num_positions());
  Eigen::VectorXd q_lower(rational_forward_kinematics_.plant().num_positions());
  for (JointIndex i{0}; i < rational_forward_kinematics_.plant().num_joints();
       ++i) {
    const auto& joint = rational_forward_kinematics_.plant().get_joint(i);
    q_upper.segment(joint.position_start(), joint.num_positions()) =
        joint.position_upper_limits();
    q_lower.segment(joint.position_start(), joint.num_positions()) =
        joint.position_lower_limits();
  }
  const Eigen::VectorXd t_upper_limit =
      rational_forward_kinematics_.ComputeTValue(q_upper, q_star);
  const Eigen::VectorXd t_lower_limit =
      rational_forward_kinematics_.ComputeTValue(q_lower, q_star);
  double rho_upper = rho_upper_initial;
  double rho_lower = rho_lower_initial;

  const std::vector<LinkVertexOnPlaneSideRational> rationals =
      GenerateLinkOnOneSideOfPlaneRationals(q_star, filtered_collision_pairs);
  solvers::MosekSolver solver;
  solver.set_stream_logging(true, "");
  solvers::MathematicalProgramResult result;
  while (rho_upper - rho_lower > rho_tolerance) {
    const double rho = (rho_upper + rho_lower) / 2;
    Eigen::VectorXd t_lower(rational_forward_kinematics_.t().size());
    Eigen::VectorXd t_upper(rational_forward_kinematics_.t().size());
    for (int i = 0; i < rational_forward_kinematics_.t().size(); ++i) {
      t_lower(i) = std::max(rho * negative_delta_t(i), t_lower_limit(i));
      t_upper(i) = std::min(rho * positive_delta_t(i), t_upper_limit(i));
    }
    auto prog = ConstructProgramToVerifyCollisionFreeBox(
        rationals, t_lower, t_upper, filtered_collision_pairs,
        verification_option);
    solver.Solve(*prog, {}, {}, &result);
    if (result.get_solution_result() ==
        solvers::SolutionResult::kSolutionFound) {
      // rho is feasible.
      rho_lower = rho;
    } else {
      // rho is infeasible.
      rho_upper = rho;
    }
  }
  return rho_lower;
}

std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    std::shared_ptr<const ConvexPolytope> link_polytope,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    BodyIndex expressed_body_index,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side) {
  // Compute the link pose
  const auto X_AB =
      rational_forward_kinematics.CalcLinkPoseAsMultilinearPolynomial(
          q_star, link_polytope->body_index(), expressed_body_index);

  return GenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link_polytope, X_AB, a_A, p_AC, plane_side);
}

std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    std::shared_ptr<const ConvexPolytope> link_polytope,
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side) {
  std::vector<LinkVertexOnPlaneSideRational> rational_fun;
  rational_fun.reserve(link_polytope->p_BV().cols());
  const symbolic::Monomial monomial_one{};
  Vector3<symbolic::Polynomial> a_A_poly;
  for (int i = 0; i < 3; ++i) {
    a_A_poly(i) = symbolic::Polynomial({{monomial_one, a_A(i)}});
  }
  for (int i = 0; i < link_polytope->p_BV().cols(); ++i) {
    // Step 1: Compute vertex position.
    const Vector3<symbolic::Polynomial> p_AVi =
        X_AB_multilinear.p_AB +
        X_AB_multilinear.R_AB * link_polytope->p_BV().col(i);

    // Step 2: Compute a_A.dot(p_AVi - p_AC)
    const symbolic::Polynomial point_on_hyperplane_side =
        a_A_poly.dot(p_AVi - p_AC);

    // Step 3: Convert the multilinear polynomial to rational function.
    rational_fun.emplace_back(
        rational_forward_kinematics
            .ConvertMultilinearPolynomialToRationalFunction(
                plane_side == PlaneSide::kPositive
                    ? point_on_hyperplane_side - 1
                    : 1 - point_on_hyperplane_side),
        link_polytope, X_AB_multilinear.frame_A_index,
        link_polytope->p_BV().col(i), a_A, plane_side);
  }
  return rational_fun;
}

void AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
    solvers::MathematicalProgram* prog,
    const symbolic::RationalFunction& polytope_on_one_side_rational,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& t_minus_t_lower,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& t_upper_minus_t,
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    const VerificationOption& verification_option) {
  DRAKE_DEMAND(t_minus_t_lower.size() == t_upper_minus_t.size());
  symbolic::Polynomial verified_polynomial =
      polytope_on_one_side_rational.numerator();
  for (int i = 0; i < t_minus_t_lower.size(); ++i) {
    const auto l_lower =
        prog->NewNonnegativePolynomial(monomial_basis,
                                       verification_option.lagrangian_type)
            .first;
    const auto l_upper =
        prog->NewNonnegativePolynomial(monomial_basis,
                                       verification_option.lagrangian_type)
            .first;
    verified_polynomial -= l_lower * t_minus_t_lower(i);
    verified_polynomial -= l_upper * t_upper_minus_t(i);
  }
  // Replace the following lines with prog->AddSosConstraint when we resolve
  // the
  // speed issue.
  const symbolic::Polynomial verified_polynomial_expected =
      prog->NewNonnegativePolynomial(monomial_basis,
                                     verification_option.link_polynomial_type)
          .first;
  const symbolic::Polynomial poly_diff{verified_polynomial -
                                       verified_polynomial_expected};
  for (const auto& item : poly_diff.monomial_to_coefficient_map()) {
    prog->AddLinearEqualityConstraint(item.second, 0);
  }
}
}  // namespace multibody
}  // namespace drake
