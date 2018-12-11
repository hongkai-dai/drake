#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"

namespace drake {
namespace multibody {
using symbolic::RationalFunction;

ConfigurationSpaceCollisionFreeRegion::ConfigurationSpaceCollisionFreeRegion(
    const MultibodyTree<double>& tree,
    const std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>&
        link_polytopes,
    const std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>&
        obstacles)
    : rational_forward_kinematics_{tree},
      link_polytopes_{static_cast<size_t>(tree.num_bodies())},
      obstacles_{obstacles},
      obstacle_center_{obstacles_.size()},
      a_hyperplane_(link_polytopes_.size()) {
  const int num_links = tree.num_bodies();
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
              q_star, 0);
  std::vector<symbolic::RationalFunction> collision_free_rationals;
  const symbolic::Monomial monomial_one{};
  for (int i = 1; i < rational_forward_kinematics_.tree().num_bodies(); ++i) {
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
  for (int i = 1; i < rational_forward_kinematics_.tree().num_bodies(); ++i) {
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
    ConstructProgramToVerifyFreeRegionAroundPosture(
        const Eigen::Ref<const Eigen::VectorXd>& q_star,
        const Eigen::Ref<const Eigen::VectorXd>& weights, double rho,
        const ConfigurationSpaceCollisionFreeRegion::VerificationOptions&
            options,
        solvers::MathematicalProgram* prog) const {
  // Check the size of q_star.
  DRAKE_ASSERT(q_star.rows() ==
               rational_forward_kinematics_.tree().num_positions());
  DRAKE_ASSERT(weights.rows() == rational_forward_kinematics_.t().rows());
  DRAKE_ASSERT((weights.array() >= 0).all());
  DRAKE_ASSERT(rho >= 0);
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
    symbolic::Polynomial lagrangian;
    switch (options.lagrangian_type_) {
      case ConfigurationSpaceCollisionFreeRegion::PositivePolynomial::kSOS: {
        lagrangian = prog->NewSosPolynomial(lagrangian_monomial_basis).first;
        break;
      }
      case ConfigurationSpaceCollisionFreeRegion::PositivePolynomial::kSDSOS: {
        throw std::runtime_error("Not implemented yet.");
        break;
      }
      case ConfigurationSpaceCollisionFreeRegion::PositivePolynomial::kDSOS: {
        throw std::runtime_error("Not implemented yet.");
        break;
      }
    }

    const symbolic::Polynomial link_outside_verification_poly =
        link_outside_halfspace - lagrangian * neighbourhood_poly;
    switch (options.link_polynomial_type_) {
      case ConfigurationSpaceCollisionFreeRegion::PositivePolynomial::kSOS: {
        std::cout << "Call add sos constraint.\n";
        prog->AddSosConstraint(link_outside_verification_poly,
                               link_outside_monomial_basis);
        std::cout << "Finish calling add sos constraint.\n";
        break;
      }
      case ConfigurationSpaceCollisionFreeRegion::PositivePolynomial::kSDSOS: {
        throw std::runtime_error("Not implemented yet.");
      }
      case ConfigurationSpaceCollisionFreeRegion::PositivePolynomial::kDSOS: {
        throw std::runtime_error("Not implemented yet.");
      }
    }
  }
}

}  // namespace multibody
}  // namespace drake
