#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "drake/common/random.h"
#include "drake/geometry/optimization/cspace_free_polytope_base.h"
#include "drake/geometry/optimization/cspace_free_structs.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/solvers/mathematical_program_result.h"

namespace drake {
namespace geometry {
namespace optimization {
/**
 This class tries to find large axis-aligned bounding boxes in the configuration
 space, such that all configurations in the boxes are collision free.
 Note that we don't guarantee to find the largest box.
 */
// CspaceFreeBox "is a" CspaceFreePolytopeBase because it can do anything inside
// CspaceFreePolytopeBase. We factor out the common code in CspaceFreeBox and
// CspaceFreePolytope to CspaceFreePolytopeBase, and also rely on the access
// control (public/protected/private) in CspaceFreePolytopeBase.
class CspaceFreeBox : public CspaceFreePolytopeBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CspaceFreeBox)

  using CspaceFreePolytopeBase::IgnoredCollisionPairs;

  ~CspaceFreeBox() override;

  using CspaceFreePolytopeBase::Options;

  /**
   When searching for the separating plane, we want to certify that the
   numerator of a rational is non-negative in the C-space box q_box_lower <= q
   <= q_box_upper (or equivalently s_box_lower <= s <= s_box_upper). Hence for
   each of the rational we will introduce Lagrangian multipliers for the
   polytopic constraint s - s_box_lower >= 0, s_box_upper - s >= 0.
   */
  class SeparatingPlaneLagrangians {
   public:
    explicit SeparatingPlaneLagrangians(int s_size)
        : s_box_lower_(s_size), s_box_upper_(s_size) {}

    /** Substitutes the decision variables in each Lagrangians with its value in
     * result, returns the substitution result.
     */
    [[nodiscard]] SeparatingPlaneLagrangians GetSolution(
        const solvers::MathematicalProgramResult& result) const;

    /// The Lagrangians for s - s_box_lower >= 0.
    const VectorX<symbolic::Polynomial>& s_box_lower() const {
      return s_box_lower_;
    }

    /// The Lagrangians for s - s_box_lower >= 0.
    VectorX<symbolic::Polynomial>& mutable_s_box_lower() {
      return s_box_lower_;
    }

    /// The Lagrangians for s_box_upper - s >= 0.
    const VectorX<symbolic::Polynomial>& s_box_upper() const {
      return s_box_upper_;
    }

    /// The Lagrangians for s_box_upper - s >= 0.
    VectorX<symbolic::Polynomial>& mutable_s_box_upper() {
      return s_box_upper_;
    }

   private:
    // The Lagrangians for s - s_box_lower >= 0.
    VectorX<symbolic::Polynomial> s_box_lower_;
    // The Lagrangians for s_box_upper - s >= 0.
    VectorX<symbolic::Polynomial> s_box_upper_;
  };

  /**
   We certify that a pair of geometries is collision free in the C-space box
   {q | q_box_lower<=q<=q_box_upper} by finding the separating plane and the
   Lagrangian multipliers. This struct contains the certificate, that the
   separating plane {x | aᵀx+b=0 } separates the two geometries in
   separating_planes()[plane_index] in the C-space box.
   */
  struct SeparationCertificateResult final : SeparationCertificateResultBase {
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SeparationCertificateResult)
    SeparationCertificateResult() {}
    ~SeparationCertificateResult() override = default;

    const std::vector<SeparatingPlaneLagrangians>& lagrangians(
        PlaneSide plane_side) const {
      return plane_side == PlaneSide::kPositive
                 ? positive_side_rational_lagrangians
                 : negative_side_rational_lagrangians;
    }

    std::vector<SeparatingPlaneLagrangians> positive_side_rational_lagrangians;
    std::vector<SeparatingPlaneLagrangians> negative_side_rational_lagrangians;
  };

  /**
   This struct stores the necessary information to search for the separating
   plane for the polytopic C-space box q_box_lower <= q <= q_box_upper.
   We need to impose that N rationals are non-negative in this C-space box.
   The denominator of each rational is always positive hence we need to impose
   the N numerators are non-negative in this C-space box.
   We impose the condition
   numerator_i(s) - λ_lower(s)ᵀ * (s - s_lower)
         -λ_upper(s)ᵀ * (s_upper - s) is sos
   λ_lower(s) are sos, λ_upper(s) are sos.
   */
  struct SeparationCertificate {
    SeparationCertificate() {}

    [[nodiscard]] SeparationCertificateResult GetSolution(
        int plane_index, const Vector3<symbolic::Polynomial>& a,
        const symbolic::Polynomial& b,
        const VectorX<symbolic::Variable>& plane_decision_vars,
        const solvers::MathematicalProgramResult& result) const;

    std::vector<SeparatingPlaneLagrangians>& mutable_lagrangians(
        PlaneSide plane_side) {
      return plane_side == PlaneSide::kPositive
                 ? positive_side_rational_lagrangians
                 : negative_side_rational_lagrangians;
    }
    // positive_side_rational_lagrangians[i] is the Lagrangian multipliers for
    // PlaneSeparatesGeometries::positive_side_rationals[i].
    std::vector<SeparatingPlaneLagrangians> positive_side_rational_lagrangians;
    // negative_side_rational_lagrangians[i] is the Lagrangian multipliers for
    // PlaneSeparatesGeometries::negative_side_rationals[i].
    std::vector<SeparatingPlaneLagrangians> negative_side_rational_lagrangians;
  };

  struct SeparationCertificateProgram final : SeparationCertificateProgramBase {
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SeparationCertificateProgram)
    SeparationCertificateProgram() = default;
    ~SeparationCertificateProgram() = default;

    SeparationCertificate certificate;
  };

  struct FindBoxGivenLagrangianOptions {
    /** When we solve an optimization problem with a cost function, after we
     find the optimal solution, we will "back off" a bit to find a strictly
     feasible solution. backoff_scale=0 means no back off. backoff_scale should
     be between [0, 1]. We recommend using a small backoff_scale value during
     bilinear alternation. Without backoff, the optimal solution is only
     marginally feasible, which could make the next iteration of bilinear
     alternation infeasible.
     */
    std::optional<double> backoff_scale{std::nullopt};

    /** ID for the solver */
    solvers::SolverId solver_id{solvers::MosekSolver::id()};

    /** options for solving the MathematicalProgram */
    std::optional<solvers::SolverOptions> solver_options{std::nullopt};

    /** We can constrain the C-space box {q | q_box_lower<=q<=q_box_upper}
     to contain some sampled q. Each column of q_inner_pts is a sample of q.
     */
    std::optional<Eigen::MatrixXd> q_inner_pts;

    /** Refer to AddMaximizeBoxVolumeCost.
     Use std::nullopt for a 0-vector.
     */
    std::optional<Eigen::VectorXd> box_volume_delta;
  };

  /** Options for bilinear alternation. */
  struct BilinearAlternationOptions {
    /** The maximum number of bilinear alternation iterations. Must be
     * non-negative.
     */
    int max_iter{10};

    /** When the change of the cost function between two consecutive
     iterations in bilinear alternation is no larger than this number, stop the
     bilinear alternation. Must be non-negative.
     */
    double convergence_tol{1E-3};

    FindBoxGivenLagrangianOptions find_box_options;
    FindSeparationCertificateOptions find_lagrangian_options;
  };

  /** Result on searching the C-space box and separating planes. */
  class SearchResult {
   public:
    SearchResult() {}

    [[nodiscard]] const Eigen::VectorXd& q_box_lower() const {
      return q_box_lower_;
    }

    [[nodiscard]] const Eigen::VectorXd& q_box_upper() const {
      return q_box_upper_;
    }

    /** Maps each plane_index to q*. Note that we might certify a different box
     * for each plane, hence q* will be different. */
    [[nodiscard]] const std::unordered_map<int, Eigen::VectorXd>& q_star()
        const {
      return q_star_;
    }

    /** Maps each plane_index to a(s) of that separating plane. */
    [[nodiscard]] const std::unordered_map<int, Vector3<symbolic::Polynomial>>&
    a() const {
      return separating_planes_.a();
    }

    /** Maps each plane_index to a(s) of that separating plane. */
    [[nodiscard]] const std::unordered_map<int, symbolic::Polynomial>& b()
        const {
      return separating_planes_.b();
    }

    [[nodiscard]] int num_iter() const { return num_iter_; }

   private:
    friend class CspaceFreeBox;
    void SetBox(const Eigen::Ref<const Eigen::VectorXd>& q_box_lower,
                const Eigen::Ref<const Eigen::VectorXd>& q_box_upper);

    // Updates q_star_[i] to `q_star`.
    void UpdateQStar(int i, const Eigen::Ref<const Eigen::VectorXd>& q_star);

    Eigen::VectorXd q_box_lower_;
    Eigen::VectorXd q_box_upper_;
    // Note that the separating plane parameters a and b are polyomials of s,
    // which depends on q_star. It is possible to have different q_star for each
    // plane.
    std::unordered_map<int, Eigen::VectorXd> q_star_;

    CspaceFreePolytopeBase::SeparatingPlanesResult separating_planes_;
    // The number of iterations at termination.
    int num_iter_{};
  };

  struct BinarySearchOptions {
    /** The maximal value of the scaling factor. Must be finite and no less than
     * scale_min. */
    double scale_max{1};
    /** The minimal value of the scaling factor.
     Must be non-negative. */
    double scale_min{0.01};
    /** The maximal number of iterations in binary search.
     Must be non-negative. */
    int max_iter{10};
    /** When the gap between the upper bound and the lower bound of the scaling
     factor is below this `convergence_tol`, stops the binary search.
     Must be strictly positive.
     */
    double convergence_tol{1E-3};

    FindSeparationCertificateOptions find_lagrangian_options;
  };

  /**
   @param plant The plant for which we compute the C-space free boxes. It
   must outlive this CspaceFreeBox object.
   @param scene_graph The scene graph that has been connected with `plant`. It
   must outlive this CspaceFreeBox object.
   @param plane_order The order of the polynomials in the plane to separate a
   pair of collision geometries.

   @note CspaceFreeBox knows nothing about contexts. The plant and
   scene_graph must be fully configured before instantiating this class.
   */
  CspaceFreeBox(const multibody::MultibodyPlant<double>* plant,
                const geometry::SceneGraph<double>* scene_graph,
                SeparatingPlaneOrder plane_order,
                const Options& options = Options{});

  /** Finds the certificates that the C-space box {q | q_box_lower <= q <=
   * q_box_upper} is collision free.
   *
   * @param q_box_lower The lower bound of the C-space box.
   * @param q_box_upper The upper bound of the C-space box.
   * @param ignored_collision_pairs We ignore the pair of geometries in
   * `ignored_collision_pairs`.
   * @param[out] q_star The tangent-configuration variable s is defined as s =
   * tan((q - q_star)/2) for the revolute joint, where it depends on q_star.
   * Note that q_star depends on the box parameter q_box_lower and q_box_upper,
   * we set q_star to the center of the C-space box.
   * @param[out] certificates Contains the certificate we successfully found for
   * each pair of geometries. Notice that depending on `options`, the program
   * could search for the certificate for each geometry pair in parallel, and
   * will terminate the search once it fails to find the certificate for any
   * pair. At termination, the pair of geometries whose optimization hasn't been
   * finished will not show up in @p certificates.
   * @retval success If true, then we have certified that the C-space box
   * {q | q_box_lower<=q<=q_box_upper} is collision free. Otherwise
   * success=false.
   */
  bool FindSeparationCertificateGivenBox(
      const Eigen::Ref<const Eigen::VectorXd>& q_box_lower,
      const Eigen::Ref<const Eigen::VectorXd>& q_box_upper,
      const IgnoredCollisionPairs& ignored_collision_pairs,
      const FindSeparationCertificateOptions& options, Eigen::VectorXd* q_star,
      std::unordered_map<SortedPair<geometry::GeometryId>,
                         SeparationCertificateResult>* certificates) const;

  /**
   Given certificates certifying the C-space free box {s | s_box_lower <= s <=
   s_box_upper}, this method constructs a program to search for a new C-space
   box {s | s_box_lower' <= s <= s_box_upper'} such that this box is collision
   free. This program treats s_box_lower and s_box_upper as decision variables,
   and searches for the separating planes between each pair of geometries. Note
   that this program doesn't contain any cost yet.
   @param certificates The return of
   FindSeparationCertificateGivenBox().
   @param[out] s_box_lower The C-space box is parameterized as {s | s_box_lower
   <= s <= s_box_upper}.
   @param[out] s_box_upper The C-space box is parameterized as {s | s_box_lower
   <= s <= s_box_upper}.
   */
  [[nodiscard]] std::unique_ptr<solvers::MathematicalProgram>
  InitializeBoxSearchProgram(
      const IgnoredCollisionPairs& ignored_collision_pairs,
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const std::unordered_map<SortedPair<geometry::GeometryId>,
                               SeparationCertificateResult>& certificates,
      VectorX<symbolic::Variable>* s_box_lower,
      VectorX<symbolic::Variable>* s_box_upper) const;

  /** Searches for a collision-free C-space box {q | q_box_lower <= q <=
   q_box_upper} through bilinear alternation.
   The goal is to maximize certain measure on volume of the C-space box.
   @param ignored_collision_pairs The paris of geometries that we ignore when
   searching for separation certificates.
   @param q_box_lower_init The initial value of q_box_lower.
   @param q_box_upper_init The initial value of q_box_upper.
   @param options The options for the bilinear alternation.
   @retval results Stores the certification result in each iteration of the
   bilinear alternation.
   */
  [[nodiscard]] std::vector<SearchResult> SearchWithBilinearAlternation(
      const IgnoredCollisionPairs& ignored_collision_pairs,
      const Eigen::Ref<const Eigen::VectorXd>& q_box_lower_init,
      const Eigen::Ref<const Eigen::VectorXd>& q_box_upper_init,
      const BilinearAlternationOptions& options) const;

  /** Binary search to find the C-space box {q | q_box_lower <= q <=
   q_box_upper} being collision free.
   We scale the box {q | q_box_lower_init <= q <= q_box_upper_init} about
   `q_center` and search the scalaring factor.
   @pre q_center is inside the box {q | q_box_lower_init <= q_box_upper_init}.
   Also `q_center` is within the robot joint limits.
   */
  [[nodiscard]] std::optional<CspaceFreeBox::SearchResult> BinarySearch(
      const IgnoredCollisionPairs& ignored_collision_pairs,
      const Eigen::Ref<const Eigen::VectorXd>& q_box_lower_init,
      const Eigen::Ref<const Eigen::VectorXd>& q_box_upper_init,
      const Eigen::Ref<const Eigen::VectorXd>& q_center,
      const BinarySearchOptions& options) const;

 private:
  // Forward declare the tester class that will test the private members.
  friend class CspaceFreeBoxTester;

  struct FindBoxGivenLagrangianResult {
    Eigen::VectorXd s_box_lower;
    Eigen::VectorXd s_box_upper;
    // a[i].dot(x) + b[i] = 0 is the separation plane for
    // this->separating_planes()[i].
    std::unordered_map<int, Vector3<symbolic::Polynomial>> a;
    std::unordered_map<int, symbolic::Polynomial> b;
  };

  /*
   Computes the range of s from the box q_box_lower <= q <= q_box_upper. We also
   set q_star = 0.5(q_box_lower + q_box_upper).
   If q_box_lower is smaller than the robot position lower limit (or q_box_upper
   is larger than the robot position upper limit), then we clamp q_box_lower (or
   q_box_upper) within the robot position limits.
   @throws error if any q_box_lower is larger than q_box_upper or the robot
   position upper limit; similarly throws an error if any q_box_upper is smaller
   than the robot position lower limit.
   */
  void ComputeSBox(const Eigen::Ref<const Eigen::VectorXd>& q_box_lower,
                   const Eigen::Ref<const Eigen::VectorXd>& q_box_upper,
                   Eigen::VectorXd* s_box_lower, Eigen::VectorXd* s_box_upper,
                   Eigen::VectorXd* q_star) const;

  /*
   This class contains the polynomials that we wish to certify are non-negative.
   */
  struct PolynomialsToCertify {
    // We have the invariant plane_geometries_[i].plane_index == i.
    std::vector<PlaneSeparatesGeometries> plane_geometries;
    VectorX<symbolic::Polynomial> s_minus_s_box_lower;
    VectorX<symbolic::Polynomial> s_box_upper_minus_s;
  };

  // Generates the polynomials used for certifying the box s_box_lower <= s <=
  // s_box_upper is collision free.
  // @note The box [s_box_lower, s_box_upper] is already inside the
  // tangent-configuration space box computed from the robot position
  // lower/upper limits.
  // TODO(hongkai.dai): after we finish implementing this class, consider to
  // change the input argument to q_box_lower and q_box_upper, if ComputeSBox is
  // only used with this function.
  void GeneratePolynomialsToCertify(
      const Eigen::Ref<const Eigen::VectorXd>& s_box_lower,
      const Eigen::Ref<const Eigen::VectorXd>& s_box_upper,
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const IgnoredCollisionPairs& ignored_collision_pairs,
      PolynomialsToCertify* certify_polynomials) const;

  /*
   Generates all the PlaneSeparatesGeometries structs that we need to verify.
   @param plane_geometries_vec[out] contain the rational function that needs to
   be non-negative within the C-space box.
   */
  void GeneratePlaneGeometriesVec(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const IgnoredCollisionPairs& ignored_collision_pairs,
      std::vector<PlaneSeparatesGeometries>* plane_geometries_vec) const;

  /*
   Constructs the program which searches for the plane separating a pair of
   geometries, for all configuration in the box {q | q_box_lower <= q <=
   q_box_upper}.
   @param[in] plane_geometries Contain the conditions that need to be
   non-negative in the box q_box_lower <= q <= q_box_upper.
   @param[in] s_minus_s_lower s - s_lower.
   @param[in] s_upper_minus_s s_upper - s.
   */
  [[nodiscard]] SeparationCertificateProgram ConstructPlaneSearchProgram(
      const PlaneSeparatesGeometries& plane_geometries,
      const VectorX<symbolic::Polynomial>& s_minus_s_lower,
      const VectorX<symbolic::Polynomial>& s_upper_minus_s) const;

  /*
   Finds the certificates that the C-space box {q | q_box_lower <= q <=
   q_box_upper} is collision free.
   @retval certificates certificates[i] is the separation certificate for a pair
   of geometries. If we cannot certify or haven't certified the separation for
   this pair, then certificates[i] contains std::nullopt. Note that when we run
   this function in parallel and options.terminate_at_failure=true, we will
   terminate all the remaining certification programs that have been launched,
   so certificates[i] = std::nullopt could be either because that we have
   attempted to find the certificate for this pair of geometry but failed, or it
   could be that we fail to find the certificate for another pair and haven't
   attempted to find the certificate for this pair.
   The geometry pair which certificates[i] certifies is given by
   separating_planes()[certificates[i].plane_index].geometry_pair().
   */
  void FindSeparationCertificateGivenBoxImpl(
      const PolynomialsToCertify& polynomials_to_certify,
      const FindSeparationCertificateOptions& options,
      std::vector<std::optional<SeparationCertificateResult>>* certificates_vec)
      const;

  /*
   Adds the constraint that each column of s_inner_pts is in the box s_box_lower
   <= s <= s_box_upper.
   */
  void AddCspaceBoxContainment(solvers::MathematicalProgram* prog,
                               const VectorX<symbolic::Variable>& s_box_lower,
                               const VectorX<symbolic::Variable>& s_box_upper,
                               const Eigen::MatrixXd& s_inner_pts) const;

  /* When we fix the Lagrangian multipliers and search for the C-space box
   { s | s_box_lower <= s <= s_box_upper}, we count the total size of all Gram
   matrices in the SOS program.
   */
  [[nodiscard]] int GetGramVarSizeForBoxSearchProgram(
      const std::vector<PlaneSeparatesGeometries>& plane_geometries_vec) const;

  /*
   Overload InitializeBoxSearchProgram.
   This overloaded function use input arguments that are constructed in other
   private functions. Some of these input arguments can be re-used if we call
   this InitializeBoxSearchProgram repeatedly (for example in bilinear search).
   @param polynoials_to_certify Check the output argument of
   GeneratePolynomialsToCertify.
   @param certificates_vec This is the output of
   FindSeparationCertificateGivenBox. It contains the Lagrangian multipliers
   which will be fixed when we search for the C-space box.
   @param s_box_lower The decision variables for the box lower limits.
   @param s_box_upper The decision variables for the box upper limits.
   @param s_minus_s_box_lower The polynomial representing s - s_box_lower.
   @param s_box_upper_minus_s The polynomial representing s_box_upper - s.
   @param gram_total_size The output of GetGramVarSizeForBoxSearchProgram.
   */
  [[nodiscard]] std::unique_ptr<solvers::MathematicalProgram>
  InitializeBoxSearchProgram(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const std::vector<PlaneSeparatesGeometries>& plane_geometries_vec,
      const std::vector<std::optional<SeparationCertificateResult>>&
          certificates_vec,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& s_box_lower,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& s_box_upper,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>&
          s_minus_s_box_lower,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>&
          s_box_upper_minus_s,
      int gram_total_size) const;

  /*
   Adds the constraint s_joint_limit_lower <= s_box_lower <= s_box_upper <=
   s_joint_limit_upper.
   */
  void AddBoxInJointLimitConstraint(
      solvers::MathematicalProgram* prog,
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const VectorX<symbolic::Variable>& s_box_lower,
      const VectorX<symbolic::Variable>& s_box_upper) const;

  /*
   @param box_volume_delta Adds the cost -power(∏ᵢ(s_box_upper(i) -
   s_box_lower(i) + δ(i)), 1/n) where n is the dimensionality of s. When δ = 0,
   this cost is the volume of the box { s | s_box_lower <= s <= s_box_upper}. If
   δ(i) is larger than δ(j), then the cost will try to focus more on enlarging
   s_box_upper(j) - s_box_lower(j), compared to s_box_upper(i) - s_box_lower(i).
   δ should be elementwise non-negative.
   */
  [[nodiscard]] std::optional<FindBoxGivenLagrangianResult>
  FindBoxGivenLagrangian(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const std::vector<PlaneSeparatesGeometries>& plane_geometries_vec,
      const std::vector<std::optional<SeparationCertificateResult>>&
          certificates_vec,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& s_box_lower,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& s_box_upper,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>&
          s_minus_s_box_lower,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>&
          s_box_upper_minus_s,
      int gram_total_size, const FindBoxGivenLagrangianOptions& options) const;
};

/*
 Adds the cost -power(∏ᵢ(s_box_upper(i) - s_box_lower(i) + δ(i)), 1/n) where n
 is the dimensionality of s. When δ = 0, this cost is the volume of the box {
 s | s_box_lower <= s <= s_box_upper}.
 @param delta δ in the documentation above. If δ(i) is larger than δ(j), then
 the cost will try to focus more on enlarging s_box_upper(j) -
 s_box_lower(j), compared to s_box_upper(i) - s_box_lower(i).
 δ should be elementwise non-negative.
 */
void AddMaximizeBoxVolumeCost(solvers::MathematicalProgram* prog,
                              const VectorX<symbolic::Variable>& s_box_lower,
                              const VectorX<symbolic::Variable>& s_box_upper,
                              const Eigen::VectorXd& delta);

/**
 Given a C-space box {q | q_box_lower <= q <= q_box_upper}, find the minimal
 scaling factor, such the box scaled about a point q_scale_center touches the
 C-space obstacle region.

 Mathematically we solve the optimization problem
 min t
 s.t q is in collision
     q in ScaledBox(t)

 Where ScaledBox is to scale the box {q | q_box_lower <= q <= q_box_upper}
 about q_scale_center by a factor of t.
 */
class ScaleCspaceBoxNonlinearProgram {
 public:
  struct Options {
    /** See MinimumDistanceConstraint for more explanation.*/
    double influence_distance{0.1};
    /** Solve the NLP with multiple trials, each one takes a different initial
     guess*/
    int num_nlp_trials{10};

    std::optional<solvers::SolverOptions> solver_options;
  };

  ScaleCspaceBoxNonlinearProgram(
      const multibody::MultibodyPlant<double>& plant,
      systems::Context<double>* plant_context,
      const Eigen::Ref<const Eigen::VectorXd>& q_box_lower,
      const Eigen::Ref<const Eigen::VectorXd>& q_box_upper,
      const Eigen::Ref<const Eigen::VectorXd>& q_scale_center, Options options);

  solvers::MathematicalProgram* get_mutable_prog() { return &prog_; }

  const solvers::MathematicalProgram& prog() const { return prog_; }

  const VectorX<symbolic::Variable>& q() const { return q_; }

  /** The scaling factor of the box. */
  const symbolic::Variable& t() const { return t_; }

  /** Solve the nonlinear program with multiple trials with different initial
   guess.
   */
  [[nodiscard]] solvers::MathematicalProgramResult Solve(
      unsigned int seed = 0) const;

 private:
  const multibody::MultibodyPlant<double>* plant_;
  solvers::MathematicalProgram prog_;
  VectorX<symbolic::Variable> q_;
  // The scaling factor of the box.
  symbolic::Variable t_;
  Eigen::VectorXd q_box_lower_;
  Eigen::VectorXd q_box_upper_;
  Eigen::VectorXd q_scale_center_;
  Options options_;
};

/**
 Finds the C-space collision-free box region {q | q_box_lower <= q <=
 q_box_upper} through nonlinear optimization.

 We solve several optimization programs to find in-collision configurations, as
 the corners of the C-space collision-free box.

 Given a seed configuration q_seed, we solve an
 optimization program

 min dist(q, q_seed)
 s.t q is in collision

 to find the first corner q_corner1. We get an initial box that centered at
 q_seed, with q_corner1 as one of its vertices. We denote this box as
 box_corner1.

 We then solve an optimization program to scale box_corner1 about its vertex
 q_corner1, to find the minimal scaling until the scaled box touches the
 in-collision C-space at a point different from q_corner1.

 min s
 s.t q in ScaleBox(box_corner1, s)
     q is in collision.

 We return this scaled box.
 */
class CspaceFreeBoxNonlinearOptimization {
 public:
  struct Options {
    int num_nlp_trials{10};
    double influence_distance{0.1};
    // After we find a corner through optimization program, we will add a
    // constraint to cut that corner from the feasible set of q in the next
    // optimization program. We add a constraint sum_i w(i)*q(i) <= sum_i w(i) *
    // q_corner(i) - cut_corner_radius where w(i) = 1 if q_corner(i) >
    // q_seed(i), and w(i) = -1 if q_corner(i) < q_seed(i)
    double cut_corner_radius_{1E-3};

    // We measure the distance dist(q, q_seed) as (q-q_seed)' * diag(q_dist_Q) *
    // (q - q_seed). If q_dist_Q is nullopt, then we regard diag(q_dist_Q) as
    // the identity matrix.
    std::optional<Eigen::VectorXd> q_dist_Q{std::nullopt};
  };

  CspaceFreeBoxNonlinearOptimization(
      const multibody::MultibodyPlant<double>& plant,
      systems::Context<double>* plant_context, Options options);

  multibody::InverseKinematics* get_mutable_ik() { return &ik_; }

  const multibody::InverseKinematics& ik() const { return ik_; }

  void Solve(const Eigen::Ref<const Eigen::VectorXd>& q_seed,
             Eigen::VectorXd* q_box_lower, Eigen::VectorXd* q_box_upper);

 private:
  const multibody::MultibodyPlant<double>* plant_;
  multibody::InverseKinematics ik_;
  Options options_;
};
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
