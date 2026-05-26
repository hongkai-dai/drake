#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "drake/common/autodiff.h"
#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"
#include "drake/math/autodiff.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/test_utilities/make_kinematic_chain.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/system.h"

namespace drake {
namespace multibody {
namespace {

using Eigen::VectorXd;

constexpr int kNMin = 2;
constexpr int kNMax = 40;
constexpr int kNumSamples = 1000;
constexpr int kNumWarmup = 10;

// Samples `num_samples` random q vectors of length `nq`, each component drawn
// i.i.d. from Uniform(-pi, pi).
std::vector<VectorXd> SampleRandomPositions(int nq, int num_samples,
                                            std::mt19937* rng) {
  std::uniform_real_distribution<double> uniform(-M_PI, M_PI);
  std::vector<VectorXd> samples;
  samples.reserve(num_samples);
  for (int s = 0; s < num_samples; ++s) {
    VectorXd q(nq);
    for (int i = 0; i < nq; ++i) q[i] = uniform(*rng);
    samples.push_back(std::move(q));
  }
  return samples;
}

// For each q in `q_samples`, sets the plant positions and times exactly one
// call to CalcGravityGeneralizedForces. Returns the minimum measured time in
// microseconds.
template <typename T>
double MinCalcGravityTimeMicroseconds(
    const MultibodyPlant<T>& plant, systems::Context<T>* context,
    const std::vector<VectorX<T>>& q_samples) {
  using Clock = std::chrono::steady_clock;

  // Warmup (untimed) to stabilize caches/branch predictors.
  const int n = static_cast<int>(q_samples.size());
  for (int w = 0; w < kNumWarmup && w < n; ++w) {
    plant.SetPositions(context, q_samples[w % n]);
    VectorX<T> g = plant.CalcGravityGeneralizedForces(*context);
    // Force `g` to be observable so the call cannot be elided.
    asm volatile("" : : "r,m"(g.data()) : "memory");
  }

  double min_us = std::numeric_limits<double>::infinity();
  for (const auto& q : q_samples) {
    plant.SetPositions(context, q);
    const auto t0 = Clock::now();
    VectorX<T> g = plant.CalcGravityGeneralizedForces(*context);
    const auto t1 = Clock::now();
    asm volatile("" : : "r,m"(g.data()) : "memory");
    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (us < min_us) min_us = us;
  }
  return min_us;
}

struct AutoDiffBundle {
  std::unique_ptr<systems::System<AutoDiffXd>> system;
  MultibodyPlant<AutoDiffXd>* plant{};
  std::unique_ptr<systems::Context<AutoDiffXd>> context;
  std::vector<VectorX<AutoDiffXd>> q_samples;
};

// Converts the given double plant + context + q samples to their AutoDiffXd
// counterparts. Each q is lifted via math::InitializeAutoDiff so its
// derivatives form an identity Jacobian (size nq x nq).
AutoDiffBundle ConvertToAutoDiff(const MultibodyPlant<double>& plant_d,
                                 const systems::Context<double>& context_d,
                                 const std::vector<VectorXd>& q_samples_d) {
  AutoDiffBundle bundle;
  bundle.system = systems::System<double>::ToScalarType<AutoDiffXd>(plant_d);
  bundle.plant = dynamic_cast<MultibodyPlant<AutoDiffXd>*>(bundle.system.get());
  DRAKE_DEMAND(bundle.plant != nullptr);
  bundle.context = bundle.plant->CreateDefaultContext();
  bundle.context->SetTimeStateAndParametersFrom(context_d);
  bundle.q_samples.reserve(q_samples_d.size());
  for (const auto& q : q_samples_d) {
    bundle.q_samples.push_back(math::InitializeAutoDiff(q));
  }
  return bundle;
}

}  // namespace
}  // namespace multibody
}  // namespace drake

int main(int argc, char** argv) {
  using drake::multibody::MultibodyPlant;
  using drake::multibody::test::MakeKinematicChain;

  const std::string out_path =
      (argc >= 2) ? argv[1] : "gravity_benchmark_timings.csv";

  std::mt19937 rng(42);

  struct Row {
    std::string scalar_type;
    int N;
    double min_us;
  };
  std::vector<Row> rows;

  for (int N = drake::multibody::kNMin; N <= drake::multibody::kNMax; ++N) {
    auto plant_d = MakeKinematicChain(N);
    auto ctx_d = plant_d->CreateDefaultContext();
    auto q_samples_d = drake::multibody::SampleRandomPositions(
        plant_d->num_positions(), drake::multibody::kNumSamples, &rng);

    const double t_double =
        drake::multibody::MinCalcGravityTimeMicroseconds<double>(
            *plant_d, ctx_d.get(), q_samples_d);
    rows.push_back({"double", N, t_double});

    auto ad =
        drake::multibody::ConvertToAutoDiff(*plant_d, *ctx_d, q_samples_d);
    const double t_ad =
        drake::multibody::MinCalcGravityTimeMicroseconds<drake::AutoDiffXd>(
            *ad.plant, ad.context.get(), ad.q_samples);
    rows.push_back({"autodiff", N, t_ad});

    std::cout << "N=" << N << "  double=" << t_double << " us"
              << "  autodiff=" << t_ad << " us\n";
  }

  std::ofstream out(out_path);
  out << "scalar_type,N,min_time_microseconds\n";
  for (const auto& r : rows) {
    out << r.scalar_type << "," << r.N << "," << r.min_us << "\n";
  }
  DRAKE_DEMAND(out.good());
  std::cout << "Wrote " << rows.size() << " rows to " << out_path << "\n";
  return 0;
}
