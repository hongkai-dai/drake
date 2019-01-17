#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"

namespace drake {
namespace multibody {
using symbolic::RationalFunction;

ConfigurationSpaceCollisionFreeRegion::ConfigurationSpaceCollisionFreeRegion(
    const MultibodyPlant<double>& plant,
    const std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>&
        link_polytopes,
    const std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>&
        obstacles)
    : rational_forward_kinematics_{plant},
      link_polytopes_{static_cast<size_t>(plant.num_bodies())},
      obstacles_{obstacles},
      obstacle_center_{obstacles_.size()},
      a_hyperplane_(link_polytopes_.size()) {
  const int num_links = plant.num_bodies();
  const int num_obstacles = static_cast<int>(obstacles_.size());
  DRAKE_DEMAND(num_obstacles > 0);
  DRAKE_DEMAND(static_cast<int>(link_polytopes_.size()) == num_links);
  for (const auto& obstacle : obstacles_) {
    DRAKE_ASSERT(obstacle.body_index == 0);
  }
  for (const auto& link_polytope : link_polytopes) {
    DRAKE_ASSERT(link_polytope.body_index != 0);
    link_polytopes_[link_polytope.body_index].push_back(link_polytope);
  }
  for (int i = 1; i < num_links; ++i) {
    const int num_link_polytopes = static_cast<int>(link_polytopes_[i].size());
    a_hyperplane_[i].resize(num_link_polytopes);
    for (int j = 0; j < num_link_polytopes; ++j) {
      a_hyperplane_[i][j].resize(num_obstacles);
      for (int k = 0; k < num_obstacles; ++k) {
        const std::string a_name = "a[" + std::to_string(i) + "][" +
                                   std::to_string(j) + "][" +
                                   std::to_string(k) + "]";
        for (int l = 0; l < 3; ++l) {
          a_hyperplane_[i][j][k](l) =
              symbolic::Variable(a_name + "(" + std::to_string(l) + ")");
        }
      }
    }
  }
  for (int i = 0; i < num_obstacles; ++i) {
    obstacle_center_[i] =
        obstacles_[i].vertices.rowwise().sum() / obstacles_[i].vertices.cols();
  }
}

std::vector<symbolic::RationalFunction> ConfigurationSpaceCollisionFreeRegion::
    GenerateLinkOutsideHalfspaceRationalFunction(
        const Eigen::VectorXd& q_star) const {
  const std::vector<RationalForwardKinematics::Pose<symbolic::Polynomial>>
      link_poses_poly =
          rational_forward_kinematics_.CalcLinkPosesAsMultilinearPolynomials(
              q_star, BodyIndex{0});
  std::vector<symbolic::RationalFunction> collision_free_rationals;
  const symbolic::Monomial monomial_one{};
  for (int i = 1; i < rational_forward_kinematics_.plant().num_bodies(); ++i) {
    for (int j = 0; j < static_cast<int>(link_polytopes_[i].size()); ++j) {
      const int num_polytope_vertices = link_polytopes_[i][j].vertices.cols();
      Matrix3X<symbolic::Polynomial> p_WV(3, num_polytope_vertices);
      for (int l = 0; l < num_polytope_vertices; ++l) {
        p_WV.col(l) =
            link_poses_poly[i].p_AB +
            link_poses_poly[i].R_AB * link_polytopes_[i][j].vertices.col(l);
      }
      for (int k = 0; k < static_cast<int>(obstacles_.size()); ++k) {
        // For each pair of link polytope and obstacle polytope, we need to
        // impose the constraint that all vertices of the link polytope are on
        // the "outer" side of the hyperplane. So each vertex of the link
        // polytope will introduce one polynomial. Likewise, we will impose the
        // constraint that each vertex of the obstacle polytope is in the
        // "inner" side of the hyperplane. This will be some linear constraints
        // on the hyperplane parameter a.
        // We want to impose the constraint a_hyperplane[i][j]k]ᵀ (p_WV -
        // p_WB_center) >= 1
        Vector3<symbolic::Polynomial> a_poly;
        for (int idx = 0; idx < 3; ++idx) {
          a_poly(idx) = symbolic::Polynomial(
              {{monomial_one, a_hyperplane_[i][j][k](idx)}});
        }
        for (int l = 0; l < link_polytopes_[i][j].vertices.cols(); ++l) {
          const symbolic::Polynomial outside_hyperplane_poly =
              a_poly.dot(p_WV.col(l) - obstacle_center_[k]) - 1;
          const symbolic::Polynomial outside_hyperplane_poly_trimmed =
              outside_hyperplane_poly.RemoveTermsWithSmallCoefficients(1e-12);
          const symbolic::RationalFunction outside_hyperplane_rational =
              rational_forward_kinematics_
                  .ConvertMultilinearPolynomialToRationalFunction(
                      outside_hyperplane_poly_trimmed);
          collision_free_rationals.push_back(outside_hyperplane_rational);
        }
      }
    }
  }
  return collision_free_rationals;
}

std::vector<symbolic::Polynomial>
ConfigurationSpaceCollisionFreeRegion::GenerateLinkOutsideHalfspacePolynomials(
    const Eigen::VectorXd& q_star) const {
  const std::vector<symbolic::RationalFunction> collision_free_rationals =
      GenerateLinkOutsideHalfspaceRationalFunction(q_star);
  std::vector<symbolic::Polynomial> collision_free_polynomials;
  collision_free_polynomials.reserve(collision_free_rationals.size());
  for (const auto& rational : collision_free_rationals) {
    collision_free_polynomials.push_back(rational.numerator());
  }
  return collision_free_polynomials;
}

std::vector<symbolic::Expression> ConfigurationSpaceCollisionFreeRegion::
    GenerateObstacleInsideHalfspaceExpression() const {
  std::vector<symbolic::Expression> exprs;
  for (int i = 1; i < rational_forward_kinematics_.plant().num_bodies(); ++i) {
    for (int j = 0; j < static_cast<int>(link_polytopes_[i].size()); ++j) {
      for (int k = 0; k < static_cast<int>(obstacles_.size()); ++k) {
        for (int l = 0; l < obstacles_[k].vertices.cols(); ++l) {
          exprs.push_back(
              a_hyperplane_[i][j][k].dot(obstacles_[k].vertices.col(l) -
                                         obstacle_center_[k]) -
              1);
        }
      }
    }
  }
  return exprs;
}

void ConfigurationSpaceCollisionFreeRegion::
    AddIndeterminatesAndObstacleInsideHalfspaceToProgram(
        const Eigen::Ref<const Eigen::VectorXd>& q_star,
        solvers::MathematicalProgram* prog) const {
  // Check the size of q_star.
  DRAKE_ASSERT(q_star.rows() ==
               rational_forward_kinematics_.plant().num_positions());
  DRAKE_ASSERT(prog);
  // t are the indeterminates.
  prog->AddIndeterminates(rational_forward_kinematics_.t());
  // The separating hyperplanes are the decision variables.
  for (int i = 1; i < static_cast<int>(a_hyperplane_.size()); ++i) {
    for (int j = 0; j < static_cast<int>(a_hyperplane_[i].size()); ++j) {
      for (int k = 0; k < static_cast<int>(a_hyperplane_[i][j].size()); ++k) {
        prog->AddDecisionVariables(a_hyperplane_[i][j][k]);
      }
    }
  }

  // Obstacles inside halfspace a'(v - p) <= 1
  const auto& obstacles_inside_halfspace =
      GenerateObstacleInsideHalfspaceExpression();
  for (const auto& expr : obstacles_inside_halfspace) {
    prog->AddLinearConstraint(expr <= 0);
  }
}

void ConfigurationSpaceCollisionFreeRegion::
    ConstructProgramToVerifyEllipsoidalFreeRegionAroundPosture(
        const Eigen::Ref<const Eigen::VectorXd>& q_star,
        const Eigen::Ref<const Eigen::VectorXd>& weights, double rho,
        const ConfigurationSpaceCollisionFreeRegion::VerificationOptions&
            options,
        solvers::MathematicalProgram* prog) const {
  DRAKE_ASSERT(weights.rows() == rational_forward_kinematics_.t().rows());
  DRAKE_ASSERT((weights.array() >= 0).all());
  DRAKE_ASSERT(rho >= 0);

  AddIndeterminatesAndObstacleInsideHalfspaceToProgram(q_star, prog);

  const auto& links_outside_halfspace =
      GenerateLinkOutsideHalfspacePolynomials(q_star);
  const symbolic::Monomial monomial_one{};
  using MonomialBasis = VectorX<symbolic::Monomial>;
  // For each variables t, we need two monomial basis. The first one is for the
  // Lagrangian multiplier, which contains all monomials of form ∏tᵢⁿⁱ, where
  // nᵢ <= 1. The second one is for the verified polynomial with the lagrangian
  // multiplier, containing all monomials of order all up to 1, except one may
  // up to 2. The value contains (ρ - ∑ᵢ wᵢtᵢ², lagrangian_monomial_basis,
  // link_outside_monomial_basis).
  std::unordered_map<
      symbolic::Variables,
      std::tuple<symbolic::Polynomial, MonomialBasis, MonomialBasis>>
      map_variables_to_indeterminate_bound_and_monomial_basis;
  for (const auto& link_outside_halfspace : links_outside_halfspace) {
    const symbolic::Variables& t_indeterminates =
        link_outside_halfspace.indeterminates();
    // Find if the polynomial ρ - ∑ᵢ wᵢtᵢ² and the monomial basis for
    // t_indeterminates has been computed already. If not, then generate the
    // monomial basis.
    symbolic::Polynomial neighbourhood_poly;
    MonomialBasis lagrangian_monomial_basis, link_outside_monomial_basis;
    auto it = map_variables_to_indeterminate_bound_and_monomial_basis.find(
        t_indeterminates);
    if (it == map_variables_to_indeterminate_bound_and_monomial_basis.end()) {
      // Compute the neighbourhood polynomial ρ - ∑ᵢ wᵢtᵢ².
      symbolic::Polynomial::MapType neighbourhood_poly_map;
      neighbourhood_poly_map.emplace(monomial_one, rho);
      for (int i = 0; i < rational_forward_kinematics_.t().rows(); ++i) {
        if (t_indeterminates.find(rational_forward_kinematics_.t()(i)) !=
            t_indeterminates.end()) {
          neighbourhood_poly_map.emplace(
              symbolic::Monomial({rational_forward_kinematics_.t()(i), 2}),
              -weights(i));
        }
      }
      neighbourhood_poly = symbolic::Polynomial(neighbourhood_poly_map);

      // Compute the monomial basis.
      lagrangian_monomial_basis =
          GenerateMonomialBasisWithOrderUpToOne(t_indeterminates);
      link_outside_monomial_basis =
          GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
              t_indeterminates);
      std::cout << "lagrangian_monomial_basis size: "
                << lagrangian_monomial_basis.rows()
                << "\nlink_outside_monomial_basis: "
                << link_outside_monomial_basis.rows() << "\n";
      map_variables_to_indeterminate_bound_and_monomial_basis.emplace_hint(
          it, t_indeterminates,
          std::make_tuple(neighbourhood_poly, lagrangian_monomial_basis,
                          link_outside_monomial_basis));
    } else {
      neighbourhood_poly = std::get<0>(it->second);
      lagrangian_monomial_basis = std::get<1>(it->second);
      link_outside_monomial_basis = std::get<2>(it->second);
    }

    // Create the Lagrangian multiplier
    const symbolic::Polynomial lagrangian =
        prog->NewNonnegativePolynomial(lagrangian_monomial_basis,
                                       options.lagrangian_type)
            .first;

    const symbolic::Polynomial link_outside_verification_poly =
        link_outside_halfspace - lagrangian * neighbourhood_poly;

    const symbolic::Polynomial link_outside_verification_poly_expected =
        prog->NewNonnegativePolynomial(link_outside_monomial_basis,
                                       options.link_polynomial_type)
            .first;
    const symbolic::Polynomial diff_poly{
        link_outside_verification_poly -
        link_outside_verification_poly_expected};
    for (const auto& diff_poly_item : diff_poly.monomial_to_coefficient_map()) {
      prog->AddLinearEqualityConstraint(diff_poly_item.second == 0);
    }
    std::cout << "Add sos constraint.\n";
  }
}

std::vector<std::vector<std::pair<symbolic::Polynomial, symbolic::Polynomial>>>
ConfigurationSpaceCollisionFreeRegion::
    ConstructProgramToVerifyBoxFreeRegionAroundPosture(
        const Eigen::Ref<const Eigen::VectorXd>& q_star,
        const Eigen::Ref<const Eigen::VectorXd>& t_lower,
        const Eigen::Ref<const Eigen::VectorXd>& t_upper,
        const ConfigurationSpaceCollisionFreeRegion::VerificationOptions&
            options,
        solvers::MathematicalProgram* prog) const {
  DRAKE_ASSERT((t_lower.array() < t_upper.array()).all());
  DRAKE_ASSERT(t_lower.size() == rational_forward_kinematics_.t().size());
  AddIndeterminatesAndObstacleInsideHalfspaceToProgram(q_star, prog);

  const auto& links_outside_halfspace =
      GenerateLinkOutsideHalfspacePolynomials(q_star);
  const symbolic::Monomial monomial_one{};
  using MonomialBasis = VectorX<symbolic::Monomial>;
  std::unordered_map<symbolic::Variables, MonomialBasis>
      map_variables_to_monomial_basis;
  // Map each variable t(i) to t_upper(i) - t(i) and t(i) - t_lower(i)
  std::unordered_map<symbolic::Variable::Id,
                     std::pair<symbolic::Polynomial, symbolic::Polynomial>>
      map_variable_to_box;
  const auto& t = rational_forward_kinematics_.t();
  for (int i = 0; i < t_lower.size(); ++i) {
    const symbolic::Polynomial p1(
        {{monomial_one, t_upper(i)}, {symbolic::Monomial(t(i)), -1}});
    const symbolic::Polynomial p2(
        {{monomial_one, -t_lower(i)}, {symbolic::Monomial(t(i)), 1}});
    map_variable_to_box.emplace(t(i).get_id(), std::make_pair(p1, p2));
  }
  std::vector<
      std::vector<std::pair<symbolic::Polynomial, symbolic::Polynomial>>>
      lagrangians_pairs(links_outside_halfspace.size());
  int link_outside_halfspace_count = 0;
  for (const auto& link_outside_halfspace : links_outside_halfspace) {
    const symbolic::Variables& t_indeterminates =
        link_outside_halfspace.indeterminates();
    lagrangians_pairs[link_outside_halfspace_count].reserve(
        2 * t_indeterminates.size());
    auto it = map_variables_to_monomial_basis.find(t_indeterminates);
    MonomialBasis lagrangian_monomial_basis;
    if (it == map_variables_to_monomial_basis.end()) {
      lagrangian_monomial_basis =
          GenerateMonomialBasisWithOrderUpToOne(t_indeterminates);
      map_variables_to_monomial_basis.emplace_hint(it, t_indeterminates,
                                                   lagrangian_monomial_basis);
    } else {
      lagrangian_monomial_basis = it->second;
    }

    // Computes the sum between lagrangian multipliers and the box condition as
    // sum (t_upper(j) - t(j)) * l1j(t) + (t(j) - t_lower(j)) * l2j(t)
    symbolic::Polynomial sum_lagrangian_times_condition{};
    for (const auto& ti : t_indeterminates) {
      const symbolic::Polynomial l1j =
          prog->NewNonnegativePolynomial(lagrangian_monomial_basis,
                                         options.lagrangian_type)
              .first;
      const symbolic::Polynomial l2j =
          prog->NewNonnegativePolynomial(lagrangian_monomial_basis,
                                         options.lagrangian_type)
              .first;
      const auto& box_conditions = map_variable_to_box.at(ti.get_id());
      sum_lagrangian_times_condition +=
          box_conditions.first * l1j + box_conditions.second * l2j;
      lagrangians_pairs[link_outside_halfspace_count].emplace_back(
          box_conditions.first, l1j);
      lagrangians_pairs[link_outside_halfspace_count].emplace_back(
          box_conditions.second, l2j);
    }

    const symbolic::Polynomial aggregated_polynomial =
        link_outside_halfspace - sum_lagrangian_times_condition;

    const symbolic::Polynomial aggregated_polynomial_expected =
        prog->NewNonnegativePolynomial(lagrangian_monomial_basis,
                                       options.link_polynomial_type)
            .first;

    const symbolic::Polynomial diff_poly{aggregated_polynomial -
                                         aggregated_polynomial_expected};
    for (const auto& diff_poly_item : diff_poly.monomial_to_coefficient_map()) {
      prog->AddLinearEqualityConstraint(diff_poly_item.second == 0);
    }

    std::cout << "Add condition for " << link_outside_halfspace_count
              << "'th (vertex-obstacle) pair.\n";
    link_outside_halfspace_count++;
  }
  DRAKE_DEMAND(link_outside_halfspace_count ==
               static_cast<int>(links_outside_halfspace.size()));
  return lagrangians_pairs;
}

std::vector<symbolic::RationalFunction>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const ConfigurationSpaceCollisionFreeRegion::Polytope& link_polytope,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    BodyIndex expressed_body_index,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side) {
  // Steps:
  // 1. Compute the pose of the link as a multilinear function.
  // 2. Compute the vertices position p_AVi as a multilinear function.
  // 3. Compute a_A.dot(p_AVi - p_AC)
  // 4. Convert a_A.dot(p_AVi - p_AC) - 1 or 1 - a_A.dot(p_AVi - p_AC) as a
  // rational function of t

  // Step 1: Compute the link pose
  const auto X_AB =
      rational_forward_kinematics.CalcLinkPoseAsMultilinearPolynomial(
          q_star, link_polytope.body_index, expressed_body_index);

  std::vector<symbolic::RationalFunction> rational_fun;
  rational_fun.reserve(link_polytope.vertices.cols());
  const symbolic::Monomial monomial_one{};
  Vector3<symbolic::Polynomial> a_A_poly;
  for (int i = 0; i < 3; ++i) {
    a_A_poly(i) = symbolic::Polynomial({{monomial_one, a_A(i)}});
  }
  for (int i = 0; i < link_polytope.vertices.cols(); ++i) {
    // Step 2: Compute vertex position.
    const Vector3<symbolic::Polynomial> p_AVi =
        X_AB.p_AB + X_AB.R_AB * link_polytope.vertices.col(i);

    // Step 3: Compute a_A.dot(p_AVi - p_AC)
    const symbolic::Polynomial point_on_hyperplane_side =
        a_A_poly.dot(p_AVi - p_AC);

    // Step 4: Convert the multilinear polynomial to rational function.
    if (plane_side == PlaneSide::kPositive) {
      rational_fun.push_back(
          rational_forward_kinematics
              .ConvertMultilinearPolynomialToRationalFunction(
                  point_on_hyperplane_side - 1));
    } else {
      rational_fun.push_back(
          rational_forward_kinematics
              .ConvertMultilinearPolynomialToRationalFunction(
                  1 - point_on_hyperplane_side));
    }
  }
  return rational_fun;
}
}  // namespace multibody
}  // namespace drake
