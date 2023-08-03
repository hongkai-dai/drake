#include "drake/geometry/optimization/cspace_free_box.h"

#include <array>
#include <limits>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>

#include "drake/common/fmt_eigen.h"
#include "drake/geometry/optimization/cspace_free_internal.h"

namespace drake {
namespace geometry {
namespace optimization {
namespace {
const double kInf = std::numeric_limits<double>::infinity();

// Compute the volume of the elongated box in the TC space {s | s_box_lower <= s
// <= s_box_upper + box_volume_delta}.
[[nodiscard]] double ComputeTCBoxVolume(
    const Eigen::Ref<const Eigen::VectorXd>& s_box_lower,
    const Eigen::Ref<const Eigen::VectorXd>& s_box_upper,
    const std::optional<Eigen::VectorXd>& box_volume_delta) {
  DRAKE_DEMAND((s_box_upper.array() >= s_box_lower.array()).all());
  return (s_box_upper - s_box_lower +
          box_volume_delta.value_or(Eigen::VectorXd::Zero(s_box_lower.rows())))
      .prod();
}

// Compute the volume of the C-space box {q | q_box_lower <= q <= q_box_upper}.
[[nodiscard]] double ComputeCBoxVolume(
    const Eigen::Ref<const Eigen::VectorXd>& q_box_lower,
    const Eigen::Ref<const Eigen::VectorXd>& q_box_upper) {
  DRAKE_DEMAND((q_box_upper.array() >= q_box_lower.array()).all());
  return (q_box_upper - q_box_lower).prod();
}
}  // namespace

CspaceFreeBox::SeparationCertificateResult
CspaceFreeBox::SeparationCertificate::GetSolution(
    int plane_index, const Vector3<symbolic::Polynomial>& a,
    const symbolic::Polynomial& b,
    const VectorX<symbolic::Variable>& plane_decision_vars,
    const solvers::MathematicalProgramResult& result) const {
  CspaceFreeBox::SeparationCertificateResult ret{};
  ret.plane_index = plane_index;

  auto set_lagrangians =
      [&result](const std::vector<CspaceFreeBox::SeparatingPlaneLagrangians>&
                    lagrangians_vec,
                std::vector<CspaceFreeBox::SeparatingPlaneLagrangians>*
                    lagrangians_result) {
        lagrangians_result->reserve(lagrangians_vec.size());
        for (const auto& lagrangians : lagrangians_vec) {
          lagrangians_result->push_back(lagrangians.GetSolution(result));
        }
      };
  set_lagrangians(this->positive_side_rational_lagrangians,
                  &ret.positive_side_rational_lagrangians);
  set_lagrangians(this->negative_side_rational_lagrangians,
                  &ret.negative_side_rational_lagrangians);
  for (int i = 0; i < 3; ++i) {
    ret.a(i) = result.GetSolution(a(i));
  }
  ret.b = result.GetSolution(b);

  ret.plane_decision_var_vals = result.GetSolution(plane_decision_vars);
  return ret;
}

void CspaceFreeBox::SearchResult::SetBox(
    const Eigen::Ref<const Eigen::VectorXd>& q_box_lower,
    const Eigen::Ref<const Eigen::VectorXd>& q_box_upper) {
  DRAKE_DEMAND(q_box_lower.rows() == q_box_upper.rows());
  DRAKE_DEMAND((q_box_lower.array() <= q_box_upper.array()).all());
  q_box_lower_ = q_box_lower;
  q_box_upper_ = q_box_upper;
}

void CspaceFreeBox::SearchResult::UpdateQStar(
    int i, const Eigen::Ref<const Eigen::VectorXd>& q_star) {
  q_star_.insert_or_assign(i, q_star);
}

CspaceFreeBox::CspaceFreeBox(const multibody::MultibodyPlant<double>* plant,
                             const geometry::SceneGraph<double>* scene_graph,
                             SeparatingPlaneOrder plane_order,
                             const Options& options)
    : CspaceFreePolytopeBase(plant, scene_graph, plane_order,
                             CspaceFreePolytopeBase::SForPlane::kOnChain,
                             options) {}

CspaceFreeBox::~CspaceFreeBox() {}

bool CspaceFreeBox::FindSeparationCertificateGivenBox(
    const Eigen::Ref<const Eigen::VectorXd>& q_box_lower,
    const Eigen::Ref<const Eigen::VectorXd>& q_box_upper,
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const FindSeparationCertificateOptions& options, Eigen::VectorXd* q_star,
    std::unordered_map<SortedPair<geometry::GeometryId>,
                       SeparationCertificateResult>* certificates) const {
  DRAKE_DEMAND(q_star != nullptr);
  DRAKE_DEMAND(certificates != nullptr);
  DRAKE_DEMAND(
      (q_box_lower.array() >=
       this->rational_forward_kin().plant().GetPositionLowerLimits().array())
          .all());
  DRAKE_DEMAND(
      (q_box_upper.array() <=
       this->rational_forward_kin().plant().GetPositionUpperLimits().array())
          .all());
  std::vector<std::optional<SeparationCertificateResult>> certificates_vec;
  Eigen::VectorXd s_box_lower;
  Eigen::VectorXd s_box_upper;
  DRAKE_DEMAND(q_star != nullptr);
  this->ComputeSBox(q_box_lower, q_box_upper, &s_box_lower, &s_box_upper,
                    q_star);
  PolynomialsToCertify polynomials_to_certify;
  this->GeneratePolynomialsToCertify(s_box_lower, s_box_upper, *q_star,
                                     ignored_collision_pairs,
                                     &polynomials_to_certify);
  this->FindSeparationCertificateGivenBoxImpl(polynomials_to_certify, options,
                                              &certificates_vec);

  certificates->clear();
  bool is_success = true;
  for (const auto& certificate : certificates_vec) {
    if (certificate.has_value()) {
      const auto& plane = separating_planes()[certificate->plane_index];
      certificates->emplace(plane.geometry_pair(), certificate.value());
    } else {
      is_success = false;
    }
  }
  return is_success;
}

std::unique_ptr<solvers::MathematicalProgram>
CspaceFreeBox::InitializeBoxSearchProgram(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const std::vector<PlaneSeparatesGeometries>& plane_geometries_vec,
    const std::vector<std::optional<SeparationCertificateResult>>&
        certificates_vec,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& s_box_lower,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& s_box_upper,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& s_minus_s_box_lower,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& s_box_upper_minus_s,
    int gram_total_size) const {
  const int s_size = rational_forward_kin().s().rows();
  DRAKE_DEMAND(s_box_lower.rows() == s_size);
  DRAKE_DEMAND(s_box_upper.rows() == s_size);
  DRAKE_DEMAND(plane_geometries_vec.size() == certificates_vec.size());
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(rational_forward_kin().s());
  // Add the indeterminates y if we need to certify non-polytopic collision
  // geometry
  for (const auto& plane_geometries : plane_geometries_vec) {
    const auto& plane = this->separating_planes()[plane_geometries.plane_index];
    if (plane.positive_side_geometry->type() != CIrisGeometryType::kPolytope ||
        plane.negative_side_geometry->type() != CIrisGeometryType::kPolytope) {
      prog->AddIndeterminates(y_slack());
      break;
    }
  }

  prog->AddDecisionVariables(s_box_lower);
  prog->AddDecisionVariables(s_box_upper);

  this->AddBoxInJointLimitConstraint(prog.get(), q_star, s_box_lower,
                                     s_box_upper);
  const auto gram_vars = prog->NewContinuousVariables(gram_total_size, "Gram");

  int gram_var_count = 0;
  for (int plane_count = 0; plane_count < ssize(plane_geometries_vec);
       ++plane_count) {
    const auto& plane_geometries = plane_geometries_vec[plane_count];
    const int plane_index = plane_geometries.plane_index;
    const auto& plane = separating_planes()[plane_index];
    prog->AddDecisionVariables(plane.decision_variables);
    const auto& certificate = certificates_vec[plane_count];
    DRAKE_THROW_UNLESS(certificate.has_value());
    DRAKE_THROW_UNLESS(certificate->plane_index == plane_index);

    // Add the constraint that positive_side_rationals and
    // negative_side_rationals are nonnegative in the C-space box.
    for (PlaneSide plane_side : {PlaneSide::kPositive, PlaneSide::kNegative}) {
      const auto& monomial_basis_array_given_side =
          this->map_body_to_monomial_basis_array().at(
              SortedPair<multibody::BodyIndex>(
                  plane.expressed_body,
                  plane.geometry(plane_side)->body_index()));
      const auto& rationals = plane_geometries.rationals(plane_side);
      const auto& lagrangians_vec = certificate->lagrangians(plane_side);
      DRAKE_THROW_UNLESS(rationals.size() == lagrangians_vec.size());
      for (int i = 0; i < ssize(rationals); ++i) {
        const int num_y =
            internal::GetNumYInRational(rationals[i], this->y_slack());
        const int num_gram_vars_per_sos = internal::GetGramVarSize(
            monomial_basis_array_given_side, this->with_cross_y(), num_y);
        internal::GramAndMonomialBasis gram_and_monomial_basis(
            monomial_basis_array_given_side, this->with_cross_y(), num_y);
        const symbolic::Polynomial poly =
            rationals[i].numerator() -
            certificate->lagrangians(plane_side)[i].s_box_lower().dot(
                s_minus_s_box_lower) -
            certificate->lagrangians(plane_side)[i].s_box_upper().dot(
                s_box_upper_minus_s);
        symbolic::Polynomial poly_sos;
        gram_and_monomial_basis.AddSos(
            prog.get(),
            gram_vars.segment(gram_var_count, num_gram_vars_per_sos),
            &poly_sos);
        gram_var_count += num_gram_vars_per_sos;
        prog->AddEqualityConstraintBetweenPolynomials(poly, poly_sos);
      }
    }
  }

  DRAKE_DEMAND(gram_var_count == gram_total_size);
  return prog;
}

std::unique_ptr<solvers::MathematicalProgram>
CspaceFreeBox::InitializeBoxSearchProgram(
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const std::unordered_map<SortedPair<geometry::GeometryId>,
                             SeparationCertificateResult>& certificates,
    VectorX<symbolic::Variable>* s_box_lower,
    VectorX<symbolic::Variable>* s_box_upper) const {
  DRAKE_DEMAND(s_box_lower != nullptr);
  DRAKE_DEMAND(s_box_upper != nullptr);
  const int s_size = rational_forward_kin().s().rows();
  *s_box_lower = symbolic::MakeVectorContinuousVariable(s_size, "s_box_lower");
  *s_box_upper = symbolic::MakeVectorContinuousVariable(s_size, "s_box_upper");
  std::vector<PlaneSeparatesGeometries> plane_geometries_vec;

  this->GeneratePlaneGeometriesVec(q_star, ignored_collision_pairs,
                                   &plane_geometries_vec);
  VectorX<symbolic::Polynomial> s_minus_s_box_lower;
  VectorX<symbolic::Polynomial> s_box_upper_minus_s;
  this->CalcSBoundsPolynomial<symbolic::Variable>(
      *s_box_lower, *s_box_upper, &s_minus_s_box_lower, &s_box_upper_minus_s);
  const int gram_total_size =
      this->GetGramVarSizeForBoxSearchProgram(plane_geometries_vec);
  // certificates_vec[i].plane_index = plane_geometries_vec[i].plane_index
  std::vector<std::optional<SeparationCertificateResult>> certificates_vec;
  for (int i = 0; i < ssize(plane_geometries_vec); ++i) {
    const auto& plane =
        this->separating_planes()[plane_geometries_vec[i].plane_index];
    certificates_vec.emplace_back(certificates.at(plane.geometry_pair()));
  }
  return this->InitializeBoxSearchProgram(
      q_star, plane_geometries_vec, certificates_vec, *s_box_lower,
      *s_box_upper, s_minus_s_box_lower, s_box_upper_minus_s, gram_total_size);
}

CspaceFreeBox::SeparatingPlaneLagrangians
CspaceFreeBox::SeparatingPlaneLagrangians::GetSolution(
    const solvers::MathematicalProgramResult& result) const {
  CspaceFreeBox::SeparatingPlaneLagrangians ret(this->s_box_lower_.rows());
  for (int i = 0; i < this->s_box_lower_.rows(); ++i) {
    ret.s_box_lower_(i) = result.GetSolution(this->s_box_lower_(i));
    ret.s_box_upper_(i) = result.GetSolution(this->s_box_upper_(i));
  }
  return ret;
}

void CspaceFreeBox::ComputeSBox(
    const Eigen::Ref<const Eigen::VectorXd>& q_box_lower,
    const Eigen::Ref<const Eigen::VectorXd>& q_box_upper,
    Eigen::VectorXd* s_box_lower, Eigen::VectorXd* s_box_upper,
    Eigen::VectorXd* q_star) const {
  if ((q_box_lower.array() > q_box_upper.array()).any()) {
    throw std::runtime_error(
        fmt::format("CspaceFreeBox: q_box_lower={} has some entries larger "
                    "than q_box_upper={}.",
                    fmt_eigen(q_box_lower.transpose()),
                    fmt_eigen(q_box_upper.transpose())));
  }
  const auto& plant = this->rational_forward_kin().plant();
  const Eigen::VectorXd q_position_lower = plant.GetPositionLowerLimits();
  const Eigen::VectorXd q_position_upper = plant.GetPositionUpperLimits();
  if ((q_box_lower.array() > q_position_upper.array()).any()) {
    throw std::runtime_error(fmt::format(
        "CspaceFreeBox: q_box_lower={} has some entries larger the the robot "
        "position upper limit={}.",
        fmt_eigen(q_box_lower.transpose()),
        fmt_eigen(q_position_upper.transpose())));
  }
  if ((q_box_upper.array() < q_position_lower.array()).any()) {
    throw std::runtime_error(fmt::format(
        "CspaceFreeBox: q_box_upper={} has some entries smaller than the robot "
        "position lower limit={}.",
        fmt_eigen(q_box_upper.transpose()),
        fmt_eigen(q_position_lower.transpose())));
  }
  const Eigen::VectorXd q_lower =
      q_box_lower.array().max(q_position_lower.array()).matrix();
  const Eigen::VectorXd q_upper =
      q_box_upper.array().min(q_position_upper.array()).matrix();
  *q_star = 0.5 * (q_lower + q_upper);
  *s_box_lower = this->rational_forward_kin().ComputeSValue(
      q_lower, *q_star, true /* angles_wrap_to_inf */);
  *s_box_upper = this->rational_forward_kin().ComputeSValue(
      q_upper, *q_star, true /* angles_wrap_to_inf */);
}

void CspaceFreeBox::GeneratePolynomialsToCertify(
    const Eigen::Ref<const Eigen::VectorXd>& s_box_lower,
    const Eigen::Ref<const Eigen::VectorXd>& s_box_upper,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const IgnoredCollisionPairs& ignored_collision_pairs,
    PolynomialsToCertify* certify_polynomials) const {
  this->CalcSBoundsPolynomial<double>(
      s_box_lower, s_box_upper, &(certify_polynomials->s_minus_s_box_lower),
      &(certify_polynomials->s_box_upper_minus_s));
  GeneratePlaneGeometriesVec(q_star, ignored_collision_pairs,
                             &(certify_polynomials->plane_geometries));
}

void CspaceFreeBox::GeneratePlaneGeometriesVec(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const IgnoredCollisionPairs& ignored_collision_pairs,
    std::vector<PlaneSeparatesGeometries>* plane_geometries_vec) const {
  DRAKE_DEMAND(plane_geometries_vec != nullptr);
  plane_geometries_vec->clear();
  std::map<int, const CSpaceSeparatingPlane<symbolic::Variable>*>
      separating_planes_map;
  for (int i = 0; i < static_cast<int>(separating_planes().size()); ++i) {
    const auto& plane = separating_planes()[i];
    if (ignored_collision_pairs.count(SortedPair<geometry::GeometryId>(
            plane.positive_side_geometry->id(),
            plane.negative_side_geometry->id())) == 0) {
      separating_planes_map.emplace(i, &plane);
    }
  }

  internal::GenerateRationals(separating_planes_map, y_slack(), q_star,
                              rational_forward_kin(), plane_geometries_vec);
}

CspaceFreeBox::SeparationCertificateProgram
CspaceFreeBox::ConstructPlaneSearchProgram(
    const PlaneSeparatesGeometries& plane_geometries,
    const VectorX<symbolic::Polynomial>& s_minus_s_lower,
    const VectorX<symbolic::Polynomial>& s_upper_minus_s) const {
  SeparationCertificateProgram ret;
  ret.plane_index = plane_geometries.plane_index;
  const auto& plane = separating_planes()[plane_geometries.plane_index];
  const SortedPair<multibody::BodyIndex> body_pair{
      plane.positive_side_geometry->body_index(),
      plane.negative_side_geometry->body_index()};
  const std::vector<int> s_for_plane_indices =
      this->map_body_pair_to_s_on_chain().at(body_pair);
  const std::unordered_set<int> s_for_plane_indices_set{
      s_for_plane_indices.begin(), s_for_plane_indices.end()};
  VectorX<symbolic::Variable> s_for_plane(s_for_plane_indices.size());
  for (int i = 0; i < ssize(s_for_plane_indices); ++i) {
    s_for_plane(i) = this->rational_forward_kin().s()(s_for_plane_indices[i]);
  }
  ret.prog->AddIndeterminates(s_for_plane);
  ret.prog->AddDecisionVariables(plane.decision_variables);

  // First count the total size of the gram matrix variables.
  int gram_var_count = 0;
  auto count_gram = [this, &s_for_plane_indices](
                        const symbolic ::RationalFunction& rational,
                        const std::array<VectorX<symbolic::Monomial>, 4>&
                            monomial_basis_array) {
    // Each rational >= 0 requires the Lagrangian multiplier for s-s_lower and
    // s_upper - s.
    const int s_for_plane_size = static_cast<int>(s_for_plane_indices.size());
    const int num_sos = 1 + 2 * s_for_plane_size;
    const int y_size = internal::GetNumYInRational(rational, this->y_slack());
    const int num_gram_vars_per_sos = internal::GetGramVarSize(
        monomial_basis_array, this->with_cross_y(), y_size);
    return num_gram_vars_per_sos * num_sos;
  };

  for (PlaneSide plane_side : {PlaneSide::kPositive, PlaneSide::kNegative}) {
    const SortedPair<multibody::BodyIndex> one_side_body_pair(
        plane.expressed_body, plane.geometry(plane_side)->body_index());
    const std::array<VectorX<symbolic::Monomial>, 4>& monomial_basis_array =
        this->map_body_to_monomial_basis_array().at(one_side_body_pair);
    for (const auto& rational : plane_geometries.rationals(plane_side)) {
      gram_var_count += count_gram(rational, monomial_basis_array);
    }
  }

  const auto gram_vars =
      ret.prog->NewContinuousVariables(gram_var_count, "Gram");

  gram_var_count = 0;
  auto add_rational_nonnegative =
      [this, &s_for_plane_indices_set, &s_minus_s_lower, &s_upper_minus_s,
       &gram_vars, &gram_var_count](
          solvers::MathematicalProgram* prog,
          const symbolic::RationalFunction& rational,
          const std::array<VectorX<symbolic::Monomial>, 4>&
              monomial_basis_array) -> SeparatingPlaneLagrangians {
    const int y_size = internal::GetNumYInRational(rational, this->y_slack());
    internal::GramAndMonomialBasis gram_and_monomial_basis(
        monomial_basis_array, this->with_cross_y(), y_size);
    const int num_gram_vars_per_sos = gram_and_monomial_basis.gram_var_size;
    const int s_size = this->rational_forward_kin().s().rows();
    SeparatingPlaneLagrangians lagrangians{s_size};

    symbolic::Polynomial poly = rational.numerator();

    for (int i = 0; i < s_size; ++i) {
      if (s_for_plane_indices_set.count(i) > 0) {
        gram_and_monomial_basis.AddSos(
            prog, gram_vars.segment(gram_var_count, num_gram_vars_per_sos),
            &lagrangians.mutable_s_box_lower()(i));
        gram_var_count += num_gram_vars_per_sos;
        gram_and_monomial_basis.AddSos(
            prog, gram_vars.segment(gram_var_count, num_gram_vars_per_sos),
            &lagrangians.mutable_s_box_upper()(i));
        gram_var_count += num_gram_vars_per_sos;
        poly -= lagrangians.s_box_lower()(i) * s_minus_s_lower(i) +
                lagrangians.s_box_upper()(i) * s_upper_minus_s(i);
      } else {
        lagrangians.mutable_s_box_lower()(i) = symbolic::Polynomial();
        lagrangians.mutable_s_box_upper()(i) = symbolic::Polynomial();
      }
    }
    symbolic::Polynomial poly_sos;
    gram_and_monomial_basis.AddSos(
        prog, gram_vars.segment(gram_var_count, num_gram_vars_per_sos),
        &poly_sos);
    gram_var_count += num_gram_vars_per_sos;
    prog->AddEqualityConstraintBetweenPolynomials(poly, poly_sos);
    return lagrangians;
  };

  if (plane.positive_side_geometry->type() != CIrisGeometryType::kPolytope ||
      plane.negative_side_geometry->type() != CIrisGeometryType::kPolytope) {
    ret.prog->AddIndeterminates(y_slack());
  }

  for (PlaneSide plane_side : {PlaneSide::kPositive, PlaneSide::kNegative}) {
    ret.certificate.mutable_lagrangians(plane_side)
        .reserve(plane_geometries.rationals(plane_side).size());
    const SortedPair<multibody::BodyIndex> one_side_body_pair(
        plane.expressed_body, plane.geometry(plane_side)->body_index());
    const std::array<VectorX<symbolic::Monomial>, 4>& monomial_basis_array =
        this->map_body_to_monomial_basis_array().at(one_side_body_pair);
    for (const auto& rational : plane_geometries.rationals(plane_side)) {
      ret.certificate.mutable_lagrangians(plane_side)
          .push_back(add_rational_nonnegative(ret.prog.get_mutable(), rational,
                                              monomial_basis_array));
    }
  }

  DRAKE_DEMAND(gram_var_count == gram_vars.rows());
  return ret;
}

void CspaceFreeBox::FindSeparationCertificateGivenBoxImpl(
    const CspaceFreeBox::PolynomialsToCertify& polynomials_to_certify,
    const FindSeparationCertificateOptions& options,
    std::vector<std::optional<CspaceFreeBox::SeparationCertificateResult>>*
        certificates_vec) const {
  std::vector<int> active_plane_indices;
  active_plane_indices.reserve(polynomials_to_certify.plane_geometries.size());
  for (const auto& plane_geometries : polynomials_to_certify.plane_geometries) {
    active_plane_indices.push_back(plane_geometries.plane_index);
  }
  *certificates_vec = std::vector<std::optional<SeparationCertificateResult>>(
      active_plane_indices.size(), std::nullopt);
  // This lambda function formulates and solves a small SOS program for each
  // pair of geometries.
  auto solve_small_sos =
      [this, &polynomials_to_certify, &active_plane_indices, &options,
       certificates_vec](int plane_count) -> std::pair<bool, int> {
    const int plane_index = active_plane_indices[plane_count];
    auto certificate_program = this->ConstructPlaneSearchProgram(
        polynomials_to_certify.plane_geometries[plane_count],
        polynomials_to_certify.s_minus_s_box_lower,
        polynomials_to_certify.s_box_upper_minus_s);

    solvers::MathematicalProgramResult result;
    solvers::MakeSolver(options.solver_id)
        ->Solve(*certificate_program.prog, std::nullopt, options.solver_options,
                &result);
    if (result.is_success()) {
      (*certificates_vec)[plane_count].emplace(
          certificate_program.certificate.GetSolution(
              plane_index, separating_planes()[plane_index].a,
              separating_planes()[plane_index].b,
              separating_planes()[plane_index].decision_variables, result));
      return std::make_pair(true, plane_count);
    } else {
      (*certificates_vec)[plane_count].reset();
      return std::make_pair(false, plane_count);
    }
  };
  this->SolveCertificationForEachPlaneInParallel(
      active_plane_indices, solve_small_sos, options.num_threads,
      options.verbose, options.terminate_at_failure);
}

void CspaceFreeBox::AddCspaceBoxContainment(
    solvers::MathematicalProgram* prog,
    const VectorX<symbolic::Variable>& s_box_lower,
    const VectorX<symbolic::Variable>& s_box_upper,
    const Eigen::MatrixXd& s_inner_pts) const {
  const int s_size = this->rational_forward_kin().s().rows();
  DRAKE_THROW_UNLESS(s_inner_pts.rows() == s_size);
  DRAKE_THROW_UNLESS(s_box_lower.rows() == s_size);
  DRAKE_THROW_UNLESS(s_box_upper.rows() == s_size);
  // We have the constraint s_box_lower <= s <= s_box_upper.
  prog->AddBoundingBoxConstraint(Eigen::VectorXd::Constant(s_size, -kInf),
                                 s_inner_pts.rowwise().minCoeff(), s_box_lower);
  prog->AddBoundingBoxConstraint(s_inner_pts.rowwise().maxCoeff(),
                                 Eigen::VectorXd::Constant(s_size, kInf),
                                 s_box_upper);
}

int CspaceFreeBox::GetGramVarSizeForBoxSearchProgram(
    const std::vector<PlaneSeparatesGeometries>& plane_geometries_vec) const {
  auto count_gram_per_rational =
      [this](const symbolic::RationalFunction& rational,
             const std::array<VectorX<symbolic::Monomial>, 4>&
                 monomial_basis_array) -> int {
    // Each rational will add one sos constraint for
    // rational.numerator() - λ_lower(s)ᵀ * (s - s_box_lower) - λ_upper(s)ᵀ *
    // (s_box_upper - s) is sos. Note that the Lagrangian multiplier λ_lower and
    // λ_upper are fixed and we don't need to declare Gram matrix decision
    // variables for them.
    const int num_y = internal::GetNumYInRational(rational, this->y_slack());
    return internal::GetGramVarSize(monomial_basis_array, this->with_cross_y(),
                                    num_y);
  };
  return CspaceFreePolytopeBase::GetGramVarSizeForPolytopeSearchProgram(
      plane_geometries_vec, {} /* ignored_collision_pairs */,
      count_gram_per_rational);
}

void CspaceFreeBox::AddBoxInJointLimitConstraint(
    solvers::MathematicalProgram* prog,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const VectorX<symbolic::Variable>& s_box_lower,
    const VectorX<symbolic::Variable>& s_box_upper) const {
  const Eigen::VectorXd q_joint_limit_lower =
      this->rational_forward_kin().plant().GetPositionLowerLimits();
  const Eigen::VectorXd q_joint_limit_upper =
      this->rational_forward_kin().plant().GetPositionUpperLimits();
  DRAKE_DEMAND((q_star.array() <= q_joint_limit_upper.array()).all());
  DRAKE_DEMAND((q_star.array() >= q_joint_limit_lower.array()).all());
  const Eigen::VectorXd s_joint_limit_lower =
      this->rational_forward_kin().ComputeSValue(q_joint_limit_lower, q_star,
                                                 true /*angles_wrap_to_inf */);
  const Eigen::VectorXd s_joint_limit_upper =
      this->rational_forward_kin().ComputeSValue(q_joint_limit_upper, q_star,
                                                 true /*angles_wrap_to_inf */);
  prog->AddBoundingBoxConstraint(s_joint_limit_lower, s_joint_limit_upper,
                                 s_box_lower);
  prog->AddBoundingBoxConstraint(s_joint_limit_lower, s_joint_limit_upper,
                                 s_box_upper);
  // Add the constraint s_box_upper >= s_box_lower
  const int s_size = rational_forward_kin().s().rows();
  Eigen::MatrixXd A(s_size, 2 * s_size);
  A.leftCols(s_size).setIdentity();
  A.rightCols(s_size) = -Eigen::MatrixXd::Identity(s_size, s_size);
  prog->AddLinearConstraint(A, Eigen::VectorXd::Zero(s_size),
                            Eigen::VectorXd::Constant(s_size, kInf),
                            {s_box_upper, s_box_lower});
}

std::optional<CspaceFreeBox::FindBoxGivenLagrangianResult>
CspaceFreeBox::FindBoxGivenLagrangian(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const std::vector<PlaneSeparatesGeometries>& plane_geometries_vec,
    const std::vector<std::optional<SeparationCertificateResult>>&
        certificates_vec,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& s_box_lower,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& s_box_upper,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& s_minus_s_box_lower,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& s_box_upper_minus_s,
    int gram_total_size, const FindBoxGivenLagrangianOptions& options) const {
  auto prog = this->InitializeBoxSearchProgram(
      q_star, plane_geometries_vec, certificates_vec, s_box_lower, s_box_upper,
      s_minus_s_box_lower, s_box_upper_minus_s, gram_total_size);

  const int s_size = this->rational_forward_kin().s().rows();
  const Eigen::VectorXd box_volume_delta =
      options.box_volume_delta.value_or(Eigen::VectorXd::Zero(s_size));
  AddMaximizeBoxVolumeCost(prog.get(), s_box_lower, s_box_upper,
                           box_volume_delta);
  if (options.q_inner_pts.has_value()) {
    DRAKE_DEMAND(options.q_inner_pts->rows() ==
                 this->rational_forward_kin().plant().num_positions());
    Eigen::MatrixXd s_inner_pts(this->rational_forward_kin().s().rows(),
                                options.q_inner_pts->cols());
    for (int i = 0; i < options.q_inner_pts->cols(); ++i) {
      s_inner_pts.col(i) = this->rational_forward_kin().ComputeSValue(
          options.q_inner_pts->col(i), q_star);
    }
    this->AddCspaceBoxContainment(prog.get(), s_box_lower, s_box_upper,
                                  s_inner_pts);
  }
  const solvers::MathematicalProgramResult result =
      internal::SolveWithBackoff(prog.get(), options.backoff_scale,
                                 options.solver_options, options.solver_id);
  std::optional<CspaceFreeBox::FindBoxGivenLagrangianResult> ret;
  if (result.is_success()) {
    ret.emplace(CspaceFreeBox::FindBoxGivenLagrangianResult());
    ret->s_box_lower = result.GetSolution(s_box_lower);
    ret->s_box_upper = result.GetSolution(s_box_upper);
    for (int i = 0; i < ssize(plane_geometries_vec); ++i) {
      const int plane_index = plane_geometries_vec[i].plane_index;
      const auto& plane = this->separating_planes()[plane_index];
      Vector3<symbolic::Polynomial> a;
      for (int j = 0; j < 3; ++j) {
        a(j) = result.GetSolution(plane.a(j));
      }
      ret->a.emplace(plane_index, a);
      ret->b.emplace(plane_index, result.GetSolution(plane.b));
    }
  }
  return ret;
}

std::vector<CspaceFreeBox::SearchResult>
CspaceFreeBox::SearchWithBilinearAlternation(
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const Eigen::Ref<const Eigen::VectorXd>& q_box_lower_init,
    const Eigen::Ref<const Eigen::VectorXd>& q_box_upper_init,
    const BilinearAlternationOptions& options) const {
  const int nq = this->rational_forward_kin().plant().num_positions();
  DRAKE_THROW_UNLESS(q_box_lower_init.rows() == nq);
  DRAKE_THROW_UNLESS(q_box_upper_init.rows() == nq);
  DRAKE_THROW_UNLESS(
      (q_box_lower_init.array() <= q_box_upper_init.array()).all());
  DRAKE_THROW_UNLESS(options.max_iter >= 0);
  DRAKE_THROW_UNLESS(options.convergence_tol >= 0);
  std::vector<CspaceFreeBox::SearchResult> ret{};
  int iter = 0;
  // Create symbolic variables for s_box_lower and s_box_upper.
  const int s_size = rational_forward_kin().s().rows();
  const VectorX<symbolic::Variable> s_box_lower_var =
      symbolic::MakeVectorContinuousVariable(s_size, "s_box_lower");
  const VectorX<symbolic::Variable> s_box_upper_var =
      symbolic::MakeVectorContinuousVariable(s_size, "s_box_upper");
  VectorX<symbolic::Polynomial> s_minus_s_box_lower;
  VectorX<symbolic::Polynomial> s_box_upper_minus_s;
  this->CalcSBoundsPolynomial<symbolic::Variable>(
      s_box_lower_var, s_box_upper_var, &s_minus_s_box_lower,
      &s_box_upper_minus_s);
  Eigen::VectorXd q_box_lower = q_box_lower_init;
  Eigen::VectorXd q_box_upper = q_box_upper_init;
  Eigen::VectorXd s_box_lower_val;
  Eigen::VectorXd s_box_upper_val;
  Eigen::VectorXd q_star;
  while (iter < options.max_iter) {
    this->ComputeSBox(q_box_lower, q_box_upper, &s_box_lower_val,
                      &s_box_upper_val, &q_star);
    double tc_box_volume_prev =
        ComputeTCBoxVolume(s_box_lower_val, s_box_upper_val,
                           options.find_box_options.box_volume_delta);
    PolynomialsToCertify polynomials_to_certify;
    this->GeneratePolynomialsToCertify(s_box_lower_val, s_box_upper_val, q_star,
                                       ignored_collision_pairs,
                                       &polynomials_to_certify);
    std::vector<std::optional<SeparationCertificateResult>> certificates_vec;
    this->FindSeparationCertificateGivenBoxImpl(polynomials_to_certify,
                                                options.find_lagrangian_options,
                                                &certificates_vec);

    if (std::any_of(
            certificates_vec.begin(), certificates_vec.end(),
            [](const std::optional<SeparationCertificateResult>& certificate) {
              return !certificate.has_value();
            })) {
      drake::log()->debug(
          "Cannot find the separation certificate at iteration {} given the "
          "box",
          iter);
      break;
    }
    ret.emplace_back();
    ret.back().SetBox(q_box_lower, q_box_upper);
    ret.back().num_iter_ = iter;
    ret.back().separating_planes_.Set(certificates_vec);
    for (int i = 0; i < ssize(certificates_vec); ++i) {
      if (certificates_vec[i].has_value()) {
        ret.back().UpdateQStar(i, q_star);
      }
    }

    // Now fix the Lagrangian and search for C-space box and separating planes.
    const int gram_total_size_in_box_program =
        this->GetGramVarSizeForBoxSearchProgram(
            polynomials_to_certify.plane_geometries);
    const std::optional<FindBoxGivenLagrangianResult> box_result =
        this->FindBoxGivenLagrangian(
            q_star, polynomials_to_certify.plane_geometries, certificates_vec,
            s_box_lower_var, s_box_upper_var, s_minus_s_box_lower,
            s_box_upper_minus_s, gram_total_size_in_box_program,
            options.find_box_options);
    if (box_result.has_value()) {
      const Eigen::VectorXd q_box_lower_sol =
          this->rational_forward_kin().ComputeQValue(box_result->s_box_lower,
                                                     q_star);
      const Eigen::VectorXd q_box_upper_sol =
          this->rational_forward_kin().ComputeQValue(box_result->s_box_upper,
                                                     q_star);
      ret.back().SetBox(q_box_lower_sol, q_box_upper_sol);
      ret.back().separating_planes_.Set(box_result->a, box_result->b);
      for (const auto& [plane_index, a_result]: box_result->a) {
        ret.back().UpdateQStar(plane_index, q_star);
      }
      ret.back().num_iter_ = iter;

      // Compute the cost function value.
      const double tc_box_volume =
          ComputeTCBoxVolume(box_result->s_box_lower, box_result->s_box_upper,
                             options.find_box_options.box_volume_delta);
      const double c_box_volume =
          ComputeCBoxVolume(q_box_lower_sol, q_box_upper_sol);
      drake::log()->debug(
          "Iteration {}: TC-space inflated box volume={}, C-space box "
          "volume={}",
          iter, tc_box_volume, c_box_volume);
      if ((tc_box_volume - tc_box_volume_prev) / tc_box_volume_prev <
          options.convergence_tol) {
        // Converged.
        break;
      }
    } else {
      drake::log()->debug(
          "SearchWithBilinearAlternation: cannot find the separation "
          "certificate at iteration {} given the Lagrangians.",
          iter);
      break;
    }
    ++iter;
  }
  return ret;
}

std::optional<CspaceFreeBox::SearchResult> CspaceFreeBox::BinarySearch(
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const Eigen::Ref<const Eigen::VectorXd>& q_box_lower_init,
    const Eigen::Ref<const Eigen::VectorXd>& q_box_upper_init,
    const Eigen::Ref<const Eigen::VectorXd>& q_center,
    const BinarySearchOptions& options) const {
  const int nq = this->rational_forward_kin().plant().num_positions();
  DRAKE_THROW_UNLESS(q_box_lower_init.rows() == nq);
  DRAKE_THROW_UNLESS((q_box_lower_init.array() <= q_center.array()).all());
  DRAKE_THROW_UNLESS((q_center.array() <= q_box_upper_init.array()).all());
  const Eigen::VectorXd q_joint_limit_lower =
      this->rational_forward_kin().plant().GetPositionLowerLimits();
  DRAKE_THROW_UNLESS((q_joint_limit_lower.array() <= q_center.array()).all());
  const Eigen::VectorXd q_joint_limit_upper =
      this->rational_forward_kin().plant().GetPositionUpperLimits();
  DRAKE_THROW_UNLESS((q_joint_limit_upper.array() >= q_center.array()).all());
  DRAKE_THROW_UNLESS(options.scale_min >= 0);
  DRAKE_THROW_UNLESS(std::isfinite(options.scale_max));
  DRAKE_THROW_UNLESS(options.scale_min <= options.scale_max);
  DRAKE_THROW_UNLESS(options.max_iter >= 0);
  DRAKE_THROW_UNLESS(options.convergence_tol > 0);

  CspaceFreeBox::SearchResult ret;

  // geometry_pair_scale_lower_bounds[i] stores the certified lower bound on
  // the scaling factor for the i'th pair of geometries (the geometries in
  // this->separating_planes()[i]).
  std::vector<double> geometry_pair_scale_lower_bounds(
      this->separating_planes().size(), 0);

  // Determins if we can certify the scaled C-space box { q | q_box_lower <= q
  // <= q_box_upper} is collision free or not. Also update `ret` if `scale` is
  // feasible.
  auto is_scale_feasible =
      [this, &ignored_collision_pairs, &q_box_lower_init, &q_box_upper_init,
       &q_center, &options, &q_joint_limit_lower, &q_joint_limit_upper,
       &geometry_pair_scale_lower_bounds, &ret](double scale) -> bool {
    // First compute the scaled box.
    const Eigen::VectorXd q_box_lower = q_joint_limit_lower.cwiseMax(
        (q_box_lower_init - q_center) * scale + q_center);
    const Eigen::VectorXd q_box_upper = q_joint_limit_upper.cwiseMin(
        (q_box_upper_init - q_center) * scale + q_center);

    // If `scale` is smaller than geometry_pair_scale_lower_bounds[plane_index],
    // then it means that in the previous iteration of the binary search, we
    // have already certified this pair of geometries is separated for a larger
    // scale (hence a larger C-space free region), and we don't need to certify
    // the separation for this `scale` (hence we add the pair to
    // `ignored_collision_pairs_for_scale`). Only attempt to certify the
    // separating plane for this pair of geometries, if `scale` is larger than
    // geometry_pair_scale_lower_bounds[plane_index].
    CspaceFreeBox::IgnoredCollisionPairs ignored_collision_pairs_for_scale =
        ignored_collision_pairs;
    for (int i = 0; i < static_cast<int>(separating_planes().size()); ++i) {
      const auto& plane = separating_planes()[i];
      const SortedPair<geometry::GeometryId> geometry_pair =
          plane.geometry_pair();
      if (ignored_collision_pairs.count(geometry_pair) == 0 &&
          geometry_pair_scale_lower_bounds[i] >= scale) {
        ignored_collision_pairs_for_scale.insert(geometry_pair);
      }
    }
    Eigen::VectorXd q_star;
    std::unordered_map<SortedPair<geometry::GeometryId>,
                       CspaceFreeBox::SeparationCertificateResult>
        certificates;
    const bool success = this->FindSeparationCertificateGivenBox(
        q_box_lower, q_box_upper, ignored_collision_pairs_for_scale,
        options.find_lagrangian_options, &q_star, &certificates);
    for (const auto& [geometry_pair, certificate_result] : certificates) {
      // If `scale` is feasible for this pair of geometries, then update the
      // lower bound stored in geometry_pair_scale_lower_bounds.
      geometry_pair_scale_lower_bounds[certificate_result.plane_index] = scale;
    }
    if (success) {
      ret.SetBox(q_box_lower, q_box_upper);
    }
    ret.separating_planes_.Update(certificates);
    for (int plane_index = 0; plane_index < ssize(this->separating_planes());
         ++plane_index) {
      if (certificates.count(
              this->separating_planes()[plane_index].geometry_pair()) > 0) {
        ret.UpdateQStar(plane_index, q_star);
      }
    }

    return success;
  };

  const int iter = internal::BinarySearch(options.scale_min, options.scale_max,
                                          options.convergence_tol,
                                          options.max_iter, is_scale_feasible);
  if (iter < 0) {
    return std::nullopt;
  }
  ret.num_iter_ = iter;
  return ret;
}

void AddMaximizeBoxVolumeCost(solvers::MathematicalProgram* prog,
                              const VectorX<symbolic::Variable>& s_box_lower,
                              const VectorX<symbolic::Variable>& s_box_upper,
                              const Eigen::VectorXd& delta) {
  const int s_size = s_box_lower.rows();
  DRAKE_DEMAND(s_box_upper.rows() == s_size);
  DRAKE_DEMAND(delta.rows() == s_size);
  DRAKE_DEMAND((delta.array() >= 0).all());
  Eigen::MatrixXd A(s_size, 2 * s_size);
  A.leftCols(s_size) = Eigen::MatrixXd::Identity(s_size, s_size);
  A.rightCols(s_size) = -Eigen::MatrixXd::Identity(s_size, s_size);
  VectorX<symbolic::Variable> vars(2 * s_size);
  vars.head(s_size) = s_box_upper;
  vars.tail(s_size) = s_box_lower;
  prog->AddMaximizeGeometricMeanCost(A, delta, vars);
}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
