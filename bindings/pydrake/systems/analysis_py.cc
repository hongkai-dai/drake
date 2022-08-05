#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/deprecation_pybind.h"
#include "drake/bindings/pydrake/common/serialize_pybind.h"
#include "drake/bindings/pydrake/common/wrap_pybind.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/common/scope_exit.h"
#include "drake/systems/analysis/batch_eval.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_barrier.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/integrator_base.h"
#include "drake/systems/analysis/monte_carlo.h"
#include "drake/systems/analysis/region_of_attraction.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"
#include "drake/systems/analysis/runge_kutta3_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/analysis/simulator_python_internal.h"
#include "drake/systems/analysis/test/quadrotor2d.h"

using std::unique_ptr;

namespace drake {
namespace pydrake {

namespace {
// Checks for Ctrl-C (and other signals) and invokes the Python handler,
// but only when called on the main interpreter thread. For details, see:
// https://docs.python.org/3/c-api/exceptions.html#c.PyErr_CheckSignals
// https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-properly-handle-ctrl-c-in-long-running-functions
void ThrowIfPythonHasPendingSignals() {
  py::gil_scoped_acquire guard;
  if (PyErr_CheckSignals() != 0) {
    throw py::error_already_set();
  }
}
}  // namespace

PYBIND11_MODULE(analysis, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::systems;

  m.doc() = "Bindings for the analysis portion of the Systems framework.";

  py::module::import("pydrake.systems.framework");
  py::module::import("pydrake.solvers");

  {
    using Class = SimulatorConfig;
    constexpr auto& cls_doc = pydrake_doc.drake.systems.SimulatorConfig;
    py::class_<Class> cls(m, "SimulatorConfig", cls_doc.doc);
    cls  // BR
        .def(ParamInit<Class>());
    DefAttributesUsingSerialize(&cls, cls_doc);
    DefReprUsingSerialize(&cls);
    DefCopyAndDeepCopy(&cls);
  }

  {
    constexpr auto& doc = pydrake_doc.drake.systems;
    using Class = SimulatorStatus;
    constexpr auto& cls_doc = doc.SimulatorStatus;
    py::class_<Class> cls(m, "SimulatorStatus", cls_doc.doc);

    using Enum = Class::ReturnReason;
    constexpr auto& enum_doc = cls_doc.ReturnReason;
    py::enum_<Class::ReturnReason>(cls, "ReturnReason", enum_doc.doc)
        .value("kReachedBoundaryTime", Enum::kReachedBoundaryTime,
            enum_doc.kReachedBoundaryTime.doc)
        .value("kReachedTerminationCondition",
            Enum::kReachedTerminationCondition,
            enum_doc.kReachedTerminationCondition.doc)
        .value("kEventHandlerFailed", Enum::kEventHandlerFailed,
            enum_doc.kEventHandlerFailed.doc);

    cls  // BR
         // TODO(eric.cousineau): Bind setter methods.
        .def("FormatMessage", &Class::FormatMessage, cls_doc.FormatMessage.doc)
        .def("succeeded", &Class::succeeded, cls_doc.succeeded.doc)
        .def("boundary_time", &Class::boundary_time, cls_doc.boundary_time.doc)
        .def("return_time", &Class::return_time, cls_doc.return_time.doc)
        .def("reason", &Class::reason, cls_doc.reason.doc)
        .def("system", &Class::system, py_rvp::reference, cls_doc.system.doc)
        .def("message", &Class::message, cls_doc.message.doc)
        .def("IsIdenticalStatus", &Class::IsIdenticalStatus, py::arg("other"),
            cls_doc.IsIdenticalStatus.doc);
  }

  {
    constexpr auto& cls_doc = pydrake_doc.drake.systems.InitializeParams;
    using Class = InitializeParams;
    py::class_<Class> cls(m, "InitializeParams", cls_doc.doc);
    cls  // BR
        .def(ParamInit<Class>());
    DefAttributesUsingSerialize(&cls, cls_doc);
    DefReprUsingSerialize(&cls);
    DefCopyAndDeepCopy(&cls);
  }

  auto bind_scalar_types = [&m](auto dummy) {
    constexpr auto& doc = pydrake_doc.drake.systems;
    using T = decltype(dummy);

    m.def("BatchEvalUniquePeriodicDiscreteUpdate",
        &BatchEvalUniquePeriodicDiscreteUpdate<T>, py::arg("system"),
        py::arg("context"), py::arg("times"), py::arg("states"),
        py::arg("inputs"), py::arg("num_time_steps") = 1,
        py::arg("input_port_index") =
            InputPortSelection::kUseFirstInputIfItExists,
        py::arg("parallelize") = Parallelism::Max(),
        py::call_guard<py::gil_scoped_release>(),
        doc.BatchEvalUniquePeriodicDiscreteUpdate.doc);

    m.def("BatchEvalTimeDerivatives", &BatchEvalTimeDerivatives<T>,
        py::arg("system"), py::arg("context"), py::arg("times"),
        py::arg("states"), py::arg("inputs"),
        py::arg("input_port_index") =
            InputPortSelection::kUseFirstInputIfItExists,
        py::arg("parallelize") = Parallelism::Max(),
        py::call_guard<py::gil_scoped_release>(),
        doc.BatchEvalTimeDerivatives.doc);

    {
      using Class = IntegratorBase<T>;
      constexpr auto& cls_doc = doc.IntegratorBase;
      DefineTemplateClassWithDefault<Class>(
          m, "IntegratorBase", GetPyParam<T>(), cls_doc.doc)
          .def("set_fixed_step_mode", &Class::set_fixed_step_mode,
              py::arg("flag"), cls_doc.set_fixed_step_mode.doc)
          .def("get_fixed_step_mode", &Class::get_fixed_step_mode,
              cls_doc.get_fixed_step_mode.doc)
          .def("set_target_accuracy", &Class::set_target_accuracy,
              py::arg("accuracy"), cls_doc.set_target_accuracy.doc)
          .def("get_target_accuracy", &Class::get_target_accuracy,
              cls_doc.get_target_accuracy.doc)
          .def("request_initial_step_size_target",
              &Class::request_initial_step_size_target, py::arg("step_size"),
              cls_doc.request_initial_step_size_target.doc)
          .def("get_initial_step_size_target",
              &Class::get_initial_step_size_target,
              cls_doc.get_initial_step_size_target.doc)
          .def("set_maximum_step_size", &Class::set_maximum_step_size,
              py::arg("max_step_size"), cls_doc.set_maximum_step_size.doc)
          .def("get_maximum_step_size", &Class::get_maximum_step_size,
              cls_doc.get_maximum_step_size.doc)
          .def("set_requested_minimum_step_size",
              &Class::set_requested_minimum_step_size, py::arg("min_step_size"),
              cls_doc.set_requested_minimum_step_size.doc)
          .def("get_requested_minimum_step_size",
              &Class::get_requested_minimum_step_size,
              cls_doc.get_requested_minimum_step_size.doc)
          .def("set_throw_on_minimum_step_size_violation",
              &Class::set_throw_on_minimum_step_size_violation,
              py::arg("throws"),
              cls_doc.set_throw_on_minimum_step_size_violation.doc)
          .def("get_throw_on_minimum_step_size_violation",
              &Class::get_throw_on_minimum_step_size_violation,
              cls_doc.get_throw_on_minimum_step_size_violation.doc)
          .def("Reset", &Class::Reset, cls_doc.Reset.doc)
          .def("Initialize", &Class::Initialize, cls_doc.Initialize.doc)
          .def("StartDenseIntegration", &Class::StartDenseIntegration,
              cls_doc.StartDenseIntegration.doc)
          .def("get_dense_output", &Class::get_dense_output,
              py_rvp::reference_internal, py_rvp::reference_internal,
              cls_doc.get_dense_output.doc)
          .def("StopDenseIntegration", &Class::StopDenseIntegration,
              cls_doc.StopDenseIntegration.doc)
          .def("ResetStatistics", &Class::ResetStatistics,
              cls_doc.ResetStatistics.doc)
          .def("get_num_substep_failures", &Class::get_num_substep_failures,
              cls_doc.get_num_substep_failures.doc)
          .def("get_num_step_shrinkages_from_substep_failures",
              &Class::get_num_step_shrinkages_from_substep_failures,
              cls_doc.get_num_step_shrinkages_from_substep_failures.doc)
          .def("get_num_step_shrinkages_from_error_control",
              &Class::get_num_step_shrinkages_from_error_control,
              cls_doc.get_num_step_shrinkages_from_error_control.doc)
          .def("get_num_derivative_evaluations",
              &Class::get_num_derivative_evaluations,
              cls_doc.get_num_derivative_evaluations.doc)
          .def("get_actual_initial_step_size_taken",
              &Class::get_actual_initial_step_size_taken,
              cls_doc.get_actual_initial_step_size_taken.doc)
          .def("get_smallest_adapted_step_size_taken",
              &Class::get_smallest_adapted_step_size_taken,
              cls_doc.get_smallest_adapted_step_size_taken.doc)
          .def("get_largest_step_size_taken",
              &Class::get_largest_step_size_taken,
              cls_doc.get_largest_step_size_taken.doc)
          .def("get_num_steps_taken", &Class::get_num_steps_taken,
              cls_doc.get_num_steps_taken.doc)
          // N.B. While `context` is not directly owned by this system, we
          // would still like our accessors to keep it alive (e.g. a user calls
          // `simulator.get_integrator().get_context()`.
          .def("get_context", &Class::get_context,
              // Keep alive, transitive: `return` keeps `self` alive.
              py::keep_alive<0, 1>(), cls_doc.get_context.doc)
          .def("get_mutable_context", &Class::get_mutable_context,
              // Keep alive, transitive: `return` keeps `self` alive.
              py::keep_alive<0, 1>(), cls_doc.get_mutable_context.doc)
          .def("reset_context", &Class::reset_context, py::arg("context"),
              // Keep alive, reference: `context` keeps `self` alive.
              py::keep_alive<2, 1>(), cls_doc.reset_context.doc);
    }

    DefineTemplateClassWithDefault<RungeKutta2Integrator<T>, IntegratorBase<T>>(
        m, "RungeKutta2Integrator", GetPyParam<T>(),
        doc.RungeKutta2Integrator.doc)
        .def(py::init<const System<T>&, const T&, Context<T>*>(),
            py::arg("system"), py::arg("max_step_size"),
            py::arg("context") = nullptr,
            // Keep alive, reference: `self` keeps `system` alive.
            py::keep_alive<1, 2>(),
            // Keep alive, reference: `self` keeps `context` alive.
            py::keep_alive<1, 4>(), doc.RungeKutta2Integrator.ctor.doc);

    // Systems with trignometric dynamics.
    DefineTemplateClassWithDefault<analysis::Quadrotor2dTrigPlant<T>,
        LeafSystem<T>>(m, "Quadrotor2dTrigPlant", GetPyParam<T>(),
        doc.analysis.Quadrotor2dTrigPlant.doc)
        .def(py::init<>(), doc.analysis.Quadrotor2dTrigPlant.ctor.doc)
        .def("length", &analysis::Quadrotor2dTrigPlant<T>::length,
            doc.analysis.Quadrotor2dTrigPlant.length.doc)
        .def("get_state_output_port",
            &analysis::Quadrotor2dTrigPlant<T>::get_state_output_port,
            py_rvp::reference_internal,
            doc.analysis.Quadrotor2dTrigPlant.get_state_output_port.doc)
        .def("get_actuation_input_port",
            &analysis::Quadrotor2dTrigPlant<T>::get_actuation_input_port,
            py_rvp::reference_internal,
            doc.analysis.Quadrotor2dTrigPlant.get_actuation_input_port.doc);

    DefineTemplateClassWithDefault<analysis::Quadrotor2dTrigStateConverter<T>,
        LeafSystem<T>>(m, "Quadrotor2dTrigStateConverter", GetPyParam<T>(),
        doc.analysis.Quadrotor2dTrigStateConverter.doc)
        .def(py::init<>(), doc.analysis.Quadrotor2dTrigStateConverter.ctor.doc);
  };
  type_visit(bind_scalar_types, CommonScalarPack{});

  auto bind_nonsymbolic_scalar_types = [&m](auto dummy) {
    constexpr auto& doc = pydrake_doc.drake.systems;
    using T = decltype(dummy);

    DefineTemplateClassWithDefault<RungeKutta3Integrator<T>, IntegratorBase<T>>(
        m, "RungeKutta3Integrator", GetPyParam<T>(),
        doc.RungeKutta3Integrator.doc)
        .def(py::init<const System<T>&, Context<T>*>(), py::arg("system"),
            py::arg("context") = nullptr,
            // Keep alive, reference: `self` keeps `system` alive.
            py::keep_alive<1, 2>(),
            // Keep alive, reference: `self` keeps `context` alive.
            py::keep_alive<1, 3>(), doc.RungeKutta3Integrator.ctor.doc);

    // See equivalent note about EventCallback in `framework_py_systems.cc`.
    using MonitorCallback =
        std::function<std::optional<EventStatus>(const Context<T>&)>;

    auto cls = DefineTemplateClassWithDefault<Simulator<T>>(
        m, "Simulator", GetPyParam<T>(), doc.Simulator.doc);
    cls  // BR
        .def(py::init<const System<T>&, unique_ptr<Context<T>>>(),
            py::arg("system"), py::arg("context") = nullptr,
            // Keep alive, reference: `self` keeps `system` alive.
            py::keep_alive<1, 2>(),
            // Keep alive, ownership: `context` keeps `self` alive.
            py::keep_alive<3, 1>(), doc.Simulator.ctor.doc)
        .def("Initialize", &Simulator<T>::Initialize,
            doc.Simulator.Initialize.doc,
            py::arg("params") = InitializeParams{})
        .def(
            "AdvanceTo",
            [](Simulator<T>* self, const T& boundary_time, bool interruptible) {
              if (!interruptible) {
                return self->AdvanceTo(boundary_time);
              }
              // Enable the interrupt monitor.
              using systems::internal::SimulatorPythonInternal;
              SimulatorPythonInternal<T>::set_python_monitor(
                  self, &ThrowIfPythonHasPendingSignals);
              ScopeExit guard([self]() {
                SimulatorPythonInternal<T>::set_python_monitor(self, nullptr);
              });
              return self->AdvanceTo(boundary_time);
            },
            py::arg("boundary_time"), py::arg("interruptible") = true,
            // This is a long-running function that might sleep; for both
            // reasons, we must release the GIL.
            py::call_guard<py::gil_scoped_release>(),
            // Amend the docstring with the additional parameter.
            []() {
              std::string new_doc = doc.Simulator.AdvanceTo.doc;
              auto found = new_doc.find("\nReturns");
              DRAKE_DEMAND(found != std::string::npos);
              new_doc.insert(found + 1, R"""(
Parameter ``interruptible``:
    When True, the simulator will check for ``KeyboardInterrupt``
    signals (Ctrl-C) during the call to AdvanceTo(). When False,
    the AdvanceTo() may or may not be interruptible, depending on
    what systems and/or monitors have been added to the simulator.
    The check has a very minor runtime performance cost, so can be
    disabled by passing ``False``. This is a Python-only parameter
    (not available in the C++ API).
)""");
              return new_doc;
            }()
                .c_str())
        .def("AdvancePendingEvents", &Simulator<T>::AdvancePendingEvents,
            doc.Simulator.AdvancePendingEvents.doc)
        .def("set_monitor",
            WrapCallbacks([](Simulator<T>* self, MonitorCallback monitor) {
              self->set_monitor([monitor](const Context<T>& context) {
                return monitor(context).value_or(EventStatus::DidNothing());
              });
            }),
            py::arg("monitor"), doc.Simulator.set_monitor.doc)
        .def("clear_monitor", &Simulator<T>::clear_monitor,
            doc.Simulator.clear_monitor.doc)
        .def("get_monitor", &Simulator<T>::get_monitor,
            doc.Simulator.get_monitor.doc)
        .def("get_context", &Simulator<T>::get_context,
            py_rvp::reference_internal, doc.Simulator.get_context.doc)
        .def("get_integrator", &Simulator<T>::get_integrator,
            py_rvp::reference_internal, doc.Simulator.get_integrator.doc)
        .def("get_mutable_integrator", &Simulator<T>::get_mutable_integrator,
            py_rvp::reference_internal,
            doc.Simulator.get_mutable_integrator.doc)
        .def("get_mutable_context", &Simulator<T>::get_mutable_context,
            py_rvp::reference_internal, doc.Simulator.get_mutable_context.doc)
        .def("has_context", &Simulator<T>::has_context,
            doc.Simulator.has_context.doc)
        .def("reset_context", &Simulator<T>::reset_context, py::arg("context"),
            // Keep alive, ownership: `context` keeps `self` alive.
            py::keep_alive<2, 1>(), doc.Simulator.reset_context.doc)
        // TODO(eric.cousineau): Bind `release_context` once some form of the
        // PR RobotLocomotion/pybind11#33 lands. Presently, it fails.
        .def("set_publish_every_time_step",
            &Simulator<T>::set_publish_every_time_step, py::arg("publish"),
            doc.Simulator.set_publish_every_time_step.doc)
        .def("set_publish_at_initialization",
            &Simulator<T>::set_publish_at_initialization, py::arg("publish"),
            doc.Simulator.set_publish_at_initialization.doc)
        .def("set_target_realtime_rate",
            &Simulator<T>::set_target_realtime_rate, py::arg("realtime_rate"),
            doc.Simulator.set_target_realtime_rate.doc)
        .def("get_target_realtime_rate",
            &Simulator<T>::get_target_realtime_rate,
            doc.Simulator.get_target_realtime_rate.doc)
        .def("get_actual_realtime_rate",
            &Simulator<T>::get_actual_realtime_rate,
            doc.Simulator.get_actual_realtime_rate.doc)
        .def("ResetStatistics", &Simulator<T>::ResetStatistics,
            doc.Simulator.ResetStatistics.doc)
        .def("get_num_publishes", &Simulator<T>::get_num_publishes,
            doc.Simulator.get_num_publishes.doc)
        .def("get_num_steps_taken", &Simulator<T>::get_num_steps_taken,
            doc.Simulator.get_num_steps_taken.doc)
        .def("get_num_discrete_updates",
            &Simulator<T>::get_num_discrete_updates,
            doc.Simulator.get_num_discrete_updates.doc)
        .def("get_num_unrestricted_updates",
            &Simulator<T>::get_num_unrestricted_updates,
            doc.Simulator.get_num_unrestricted_updates.doc)
        .def("get_system", &Simulator<T>::get_system, py_rvp::reference,
            doc.Simulator.get_system.doc);

    m  // BR
        .def("ApplySimulatorConfig",
            py::overload_cast<const SimulatorConfig&,
                drake::systems::Simulator<T>*>(&ApplySimulatorConfig<T>),
            py::arg("config"), py::arg("simulator"),
            pydrake_doc.drake.systems.ApplySimulatorConfig.doc_config_sim)
        .def("ExtractSimulatorConfig", &ExtractSimulatorConfig<T>,
            py::arg("simulator"),
            pydrake_doc.drake.systems.ExtractSimulatorConfig.doc);
  };
  type_visit(bind_nonsymbolic_scalar_types, NonSymbolicScalarPack{});

  // Simulator Flags
  m  // BR
      .def(
          "ResetIntegratorFromFlags",
          [](Simulator<double>* simulator, const std::string& scheme,
              const double& max_step_size) {
            IntegratorBase<double>& result =
                ResetIntegratorFromFlags(simulator, scheme, max_step_size);
            return &result;
          },
          py::arg("simulator"), py::arg("scheme"), py::arg("max_step_size"),
          py_rvp::reference,
          // Keep alive, reference: `return` keeps `simulator` alive.
          py::keep_alive<0, 1>(),
          pydrake_doc.drake.systems.ResetIntegratorFromFlags.doc)
      .def(
          "ResetIntegratorFromFlags",
          [](Simulator<AutoDiffXd>* simulator, const std::string& scheme,
              const AutoDiffXd& max_step_size) {
            IntegratorBase<AutoDiffXd>& result =
                ResetIntegratorFromFlags(simulator, scheme, max_step_size);
            return &result;
          },
          py::arg("simulator"), py::arg("scheme"), py::arg("max_step_size"),
          py_rvp::reference,
          // Keep alive, reference: `return` keeps `simulator` alive.
          py::keep_alive<0, 1>(),
          pydrake_doc.drake.systems.ResetIntegratorFromFlags.doc)
      .def("GetIntegrationSchemes", &GetIntegrationSchemes,
          pydrake_doc.drake.systems.GetIntegrationSchemes.doc);

  // Print Simulator Statistics
  m  // BR
      .def("PrintSimulatorStatistics", &PrintSimulatorStatistics<double>,
          pydrake_doc.drake.systems.PrintSimulatorStatistics.doc)
      .def("PrintSimulatorStatistics", &PrintSimulatorStatistics<AutoDiffXd>,
          pydrake_doc.drake.systems.PrintSimulatorStatistics.doc);

  // Monte Carlo Testing
  {
    // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
    using namespace drake::systems::analysis;
    constexpr auto& doc = pydrake_doc.drake.systems.analysis;

    m.def("RandomSimulation",
        WrapCallbacks([](const SimulatorFactory make_simulator,
                          const ScalarSystemFunction& output, double final_time,
                          RandomGenerator* generator) -> double {
          return RandomSimulation(
              make_simulator, output, final_time, generator);
        }),
        py::arg("make_simulator"), py::arg("output"), py::arg("final_time"),
        py::arg("generator"), doc.RandomSimulation.doc);

    py::class_<RandomSimulationResult>(
        m, "RandomSimulationResult", doc.RandomSimulationResult.doc)
        .def_readwrite("output", &RandomSimulationResult::output,
            doc.RandomSimulationResult.output.doc)
        .def_readwrite("generator_snapshot",
            &RandomSimulationResult::generator_snapshot,
            doc.RandomSimulationResult.generator_snapshot.doc);

    // Note: This hard-codes `parallelism` to be off, since parallel execution
    // of Python systems on multiple threads was thought to be unsupported. It's
    // possible that with `py::call_guard<py::gil_scoped_release>` it would
    // actually be fine, so we could revisit that decision at some point.
    m.def("MonteCarloSimulation",
        WrapCallbacks([](const SimulatorFactory make_simulator,
                          const ScalarSystemFunction& output, double final_time,
                          int num_samples, RandomGenerator* generator)
                          -> std::vector<RandomSimulationResult> {
          return MonteCarloSimulation(make_simulator, output, final_time,
              num_samples, generator, /* parallelism = */ Parallelism::None());
        }),
        py::arg("make_simulator"), py::arg("output"), py::arg("final_time"),
        py::arg("num_samples"), py::arg("generator"),
        doc.MonteCarloSimulation.doc);

    {
      using Class = RegionOfAttractionOptions;
      constexpr auto& cls_doc = doc.RegionOfAttractionOptions;
      py::class_<Class, std::shared_ptr<Class>> cls(
          m, "RegionOfAttractionOptions", cls_doc.doc);
      cls.def(py::init<>(), cls_doc.ctor.doc)
          // TODO(jeremy.nimmer): replace the def_readwrite with
          // DefAttributesUsingSerialize when we fix binding a
          // VectorX<symbolic::Variable> state_variables to a numpy array of
          // objects.
          .def_readwrite("lyapunov_candidate",
              &RegionOfAttractionOptions::lyapunov_candidate,
              doc.RegionOfAttractionOptions.lyapunov_candidate.doc)
          .def_readwrite("state_variables",
              &RegionOfAttractionOptions::state_variables,
              // dtype = object arrays must be copied, and cannot be referenced.
              py_rvp::copy, doc.RegionOfAttractionOptions.state_variables.doc)
          .def_readwrite("use_implicit_dynamics",
              &RegionOfAttractionOptions::use_implicit_dynamics,
              doc.RegionOfAttractionOptions.use_implicit_dynamics.doc)
          .def_readwrite("solver_id", &RegionOfAttractionOptions::solver_id,
              doc.RegionOfAttractionOptions.solver_id.doc)
          .def_readwrite("solver_options",
              &RegionOfAttractionOptions::solver_options,
              doc.RegionOfAttractionOptions.solver_options.doc);

      DefReprUsingSerialize(&cls);
      DefCopyAndDeepCopy(&cls);
    }

    m.def("RegionOfAttraction", &RegionOfAttraction, py::arg("system"),
        py::arg("context"), py::arg("options") = RegionOfAttractionOptions(),
        doc.RegionOfAttraction.doc);
  }

  {
    constexpr auto& cls_doc = pydrake_doc.drake.systems.analysis.ControlBarrier;
    using Class = analysis::ControlBarrier;
    auto control_barrier =
        py::class_<Class>(m, "ControlBarrier", cls_doc.doc)
            .def(
                py::init<const Eigen::Ref<const VectorX<symbolic::Polynomial>>&,
                    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>&,
                    std::optional<symbolic::Polynomial>,
                    const Eigen::Ref<const VectorX<symbolic::Variable>>&,
                    double, std::optional<double>,
                    std::vector<VectorX<symbolic::Polynomial>>,
                    const Eigen::Ref<const Eigen::MatrixXd>&,
                    const Eigen::Ref<const VectorX<symbolic::Polynomial>>&>(),
                py::arg("f"), py::arg("G"), py::arg("dynamics_denominator"),
                py::arg("x"), py::arg("beta_minus"), py::arg("beta_plus"),
                py::arg("unsafe_regions"), py::arg("u_vertices"),
                py::arg("state_eq_constraints"), cls_doc.ctor.doc)
            .def(
                "AddControlBarrierConstraint",
                [](const analysis::ControlBarrier& self,
                    solvers::MathematicalProgram* prog,
                    const symbolic::Polynomial& lambda0,
                    const std::optional<symbolic::Polynomial>& lambda1,
                    const VectorX<symbolic::Polynomial>& l,
                    const VectorX<symbolic::Polynomial>&
                        state_constraints_lagrangian,
                    const symbolic::Polynomial& h, double deriv_eps,
                    const std::optional<symbolic::Polynomial>& a) {
                  symbolic::Polynomial hdot_poly;
                  VectorX<symbolic::Monomial> monomials;
                  MatrixX<symbolic::Variable> gram;
                  self.AddControlBarrierConstraint(prog, lambda0, lambda1, l,
                      state_constraints_lagrangian, h, deriv_eps, a, &hdot_poly,
                      &monomials, &gram);
                  return std::make_tuple(hdot_poly, monomials, gram);
                },
                py::arg("prog"), py::arg("lambda0"), py::arg("lambda1"),
                py::arg("l"), py::arg("state_constraints_lagrangian"),
                py::arg("h"), py::arg("deriv_eps"), py::arg("a"),
                cls_doc.AddControlBarrierConstraint.doc);

    py::class_<Class::LagrangianReturn>(control_barrier, "LagrangianReturn")
        .def(
            "prog",
            [](const Class::LagrangianReturn& self) { return self.prog.get(); },
            pybind11::return_value_policy::reference)
        .def_readonly("lambda0", &Class::LagrangianReturn::lambda0)
        .def_readonly("lambda0_gram", &Class::LagrangianReturn::lambda0_gram)
        .def_readonly("lambda1", &Class::LagrangianReturn::lambda1)
        .def_readonly("lambda1_gram", &Class::LagrangianReturn::lambda1_gram)
        .def_readonly("l", &Class::LagrangianReturn::l)
        .def_readonly("l_grams", &Class::LagrangianReturn::l_grams)
        .def_readonly("state_constraints_lagrangian",
            &Class::LagrangianReturn::state_constraints_lagrangian)
        .def_readonly("a", &Class::LagrangianReturn::a)
        .def_readonly("a_gram", &Class::LagrangianReturn::a_gram);

    control_barrier.def("ConstructLagrangianProgram",
        &Class::ConstructLagrangianProgram, py::arg("h"), py::arg("deriv_eps"),
        py::arg("lambda0_degree"), py::arg("lambda1_degree"),
        py::arg("l_degrees"), py::arg("state_constraints_lagrangian_degrees"),
        py::arg("a_degree"), cls_doc.ConstructLagrangianProgram.doc);

    py::class_<Class::UnsafeReturn>(control_barrier, "UnsafeReturn")
        .def(
            "prog",
            [](const Class::UnsafeReturn& self) { return self.prog.get(); },
            pybind11::return_value_policy::reference)
        .def_readonly("t", &Class::UnsafeReturn::t)
        .def_readonly("t_gram", &Class::UnsafeReturn::t_gram)
        .def_readonly("s", &Class::UnsafeReturn::s)
        .def_readonly("s_grams", &Class::UnsafeReturn::s_grams)
        .def_readonly("state_constraints_lagrangian",
            &Class::UnsafeReturn::state_constraints_lagrangian)
        .def_readonly("a", &Class::UnsafeReturn::a)
        .def_readonly("a_gram", &Class::UnsafeReturn::a_gram)
        .def_readonly("sos_poly", &Class::UnsafeReturn::sos_poly)
        .def_readonly("sos_poly_gram", &Class::UnsafeReturn::sos_poly_gram);

    control_barrier.def("ConstructUnsafeRegionProgram",
        &Class::ConstructUnsafeRegionProgram, py::arg("h"),
        py::arg("region_index"), py::arg("t_degree"), py::arg("s_degrees"),
        py::arg("state_constraints_lagrangian_degrees"), py::arg("a_degree"),
        cls_doc.ConstructUnsafeRegionProgram.doc);

    py::class_<Class::BarrierReturn>(control_barrier, "BarrierReturn")
        .def(
            "prog",
            [](const Class::BarrierReturn& self) { return self.prog.get(); },
            pybind11::return_value_policy::reference)
        .def_readonly("h", &Class::BarrierReturn::h)
        .def_readonly("hdot_sos", &Class::BarrierReturn::hdot_sos)
        .def_readonly("hdot_sos_gram", &Class::BarrierReturn::hdot_sos_gram)
        .def_readonly("hdot_state_constraints_lagrangian",
            &Class::BarrierReturn::hdot_state_constraints_lagrangian)
        .def_readonly("hdot_a", &Class::BarrierReturn::hdot_a)
        .def_readonly("hdot_a_gram", &Class::BarrierReturn::hdot_a_gram)
        .def_readonly("s", &Class::BarrierReturn::s)
        .def_readonly("s_grams", &Class::BarrierReturn::s_grams)
        .def_readonly(
            "unsafe_sos_polys", &Class::BarrierReturn::unsafe_sos_polys)
        .def_readonly("unsafe_sos_poly_grams",
            &Class::BarrierReturn::unsafe_sos_poly_grams)
        .def_readonly("unsafe_state_constraints_lagrangian",
            &Class::BarrierReturn::unsafe_state_constraints_lagrangian)
        .def_readonly("unsafe_a", &Class::BarrierReturn::unsafe_a)
        .def_readonly("unsafe_a_gram", &Class::BarrierReturn::unsafe_a_gram);

    control_barrier.def("ConstructBarrierProgram",
        &Class::ConstructBarrierProgram, py::arg("lambda0"), py::arg("lambda1"),
        py::arg("l"), py::arg("hdot_state_constraints_lagrangian_degrees"),
        py::arg("hdot_a_degree"), py::arg("t"),
        py::arg("unsafe_state_constraints_lagrangian_degrees"),
        py::arg("h_degree"), py::arg("deriv_eps"), py::arg("s_degrees"),
        py::arg("unsafe_a_degrees"), cls_doc.ConstructBarrierProgram.doc);

    py::class_<Class::SearchOptions>(control_barrier, "SearchOptions")
        .def_readwrite(
            "barrier_step_solver", &Class::SearchOptions::barrier_step_solver)
        .def_readwrite("lagrangian_step_solver",
            &Class::SearchOptions::lagrangian_step_solver)
        .def_readwrite(
            "bilinear_iterations", &Class::SearchOptions::bilinear_iterations)
        .def_readwrite("lagrangian_step_solver_options",
            &Class::SearchOptions::lagrangian_step_solver_options)
        .def_readwrite("barrier_step_solver_options",
            &Class::SearchOptions::barrier_step_solver_options)
        .def_readwrite("barrier_tiny_coeff_tol",
            &Class::SearchOptions::barrier_tiny_coeff_tol)
        .def_readwrite("lagrangian_tiny_coeff_tol",
            &Class::SearchOptions::lagrangian_tiny_coeff_tol)
        .def_readwrite(
            "hsol_tiny_coeff_tol", &Class::SearchOptions::hsol_tiny_coeff_tol)
        .def_readwrite(
            "lsol_tiny_coeff_tol", &Class::SearchOptions::lsol_tiny_coeff_tol);

    py::class_<Class::SearchResult>(control_barrier, "SearchResult")
        .def_readonly("success", &Class::SearchResult::success)
        .def_readonly("h", &Class::SearchResult::h)
        .def_readonly("lambda0", &Class::SearchResult::lambda0)
        .def_readonly("lambda1", &Class::SearchResult::lambda1)
        .def_readonly("l", &Class::SearchResult::l)
        .def_readonly("hdot_state_constraints_lagrangian",
            &Class::SearchResult::hdot_state_constraints_lagrangian)
        .def_readonly("t", &Class::SearchResult::t)
        .def_readonly("s", &Class::SearchResult::s)
        .def_readonly("unsafe_state_constraints_lagrangian",
            &Class::SearchResult::unsafe_state_constraints_lagrangian);

    py::class_<Class::SearchWithSlackAOptions, Class::SearchOptions>(
        control_barrier, "SearchWithSlackAOptions")
        .def(py::init<double, double, bool, double, std::vector<double>>(),
            py::arg("hdot_a_zero_tol"), py::arg("unsafe_a_zero_tol"),
            py::arg("use_zero_a"), py::arg("hdot_a_cost_weight"),
            py::arg("unsafe_a_cost_weight"),
            cls_doc.SearchWithSlackAOptions.ctor.doc)
        .def_readwrite(
            "hdot_a_zero_tol", &Class::SearchWithSlackAOptions::hdot_a_zero_tol)
        .def_readwrite("unsafe_a_zero_tol",
            &Class::SearchWithSlackAOptions::unsafe_a_zero_tol)
        .def_readwrite(
            "use_zero_a", &Class::SearchWithSlackAOptions::use_zero_a)
        .def_readwrite("hdot_a_cost_weight",
            &Class::SearchWithSlackAOptions::hdot_a_cost_weight)
        .def_readwrite("unsafe_a_cost_weight",
            &Class::SearchWithSlackAOptions::unsafe_a_cost_weight);

    py::class_<Class::SearchWithSlackAResult, Class::SearchResult>(
        control_barrier, "SearchWithSlackAResult")
        .def_readonly("hdot_a", &Class::SearchWithSlackAResult::hdot_a)
        .def_readonly(
            "hdot_a_gram", &Class::SearchWithSlackAResult::hdot_a_gram)
        .def_readonly("unsafe_a", &Class::SearchWithSlackAResult::unsafe_a)
        .def_readonly(
            "unsafe_a_grams", &Class::SearchWithSlackAResult::unsafe_a_grams);

    control_barrier.def("SearchWithSlackA",
        &analysis::ControlBarrier::SearchWithSlackA, py::arg("h_init"),
        py::arg("h_degree"), py::arg("deriv_eps"), py::arg("lambda0_degree"),
        py::arg("lambda1_degree"), py::arg("l_degrees"),
        py::arg("hdot_state_constraints_lagrangian_degrees"),
        py::arg("hdot_a_degree"), py::arg("t_degree"), py::arg("s_degrees"),
        py::arg("unsafe_state_constraints_lagrangian_degrees"),
        py::arg("unsafe_a_degrees"), py::arg("x_safe"), py::arg("h_x_safe_min"),
        py::arg("search_options"), cls_doc.SearchWithSlackA.doc);

    py::class_<Class::SearchLagrangianResult>(
        control_barrier, "SearchLagrangianResult")
        .def_readonly("success", &Class::SearchLagrangianResult::success)
        .def_readonly("lambda0", &Class::SearchLagrangianResult::lambda0)
        .def_readonly("lambda1", &Class::SearchLagrangianResult::lambda1)
        .def_readonly("l", &Class::SearchLagrangianResult::l)
        .def_readonly("hdot_state_constraints_lagrangian",
            &Class::SearchLagrangianResult::hdot_state_constraints_lagrangian)
        .def_readonly("hdot_a", &Class::SearchLagrangianResult::hdot_a)
        .def_readonly(
            "hdot_a_gram", &Class::SearchLagrangianResult::hdot_a_gram)
        .def_readonly("t", &Class::SearchLagrangianResult::t)
        .def_readonly("s", &Class::SearchLagrangianResult::s)
        .def_readonly("unsafe_state_constraints_lagrangian",
            &Class::SearchLagrangianResult::unsafe_state_constraints_lagrangian)
        .def_readonly("unsafe_a", &Class::SearchLagrangianResult::unsafe_a)
        .def_readonly(
            "unsafe_a_grams", &Class::SearchLagrangianResult::unsafe_a_grams);

    control_barrier.def("SearchLagrangian", &Class::SearchLagrangian,
        py::arg("h"), py::arg("deriv_eps"), py::arg("lambda0_degree"),
        py::arg("lambda1_degree"), py::arg("l_degrees"),
        py::arg("hdot_state_constraints_lagrangian_degrees"),
        py::arg("hdot_a_degree"), py::arg("t_degree"), py::arg("s_degrees"),
        py::arg("unsafe_state_constraints_lagrangian_degrees"),
        py::arg("unsafe_a_degrees"), py::arg("search_options"),
        cls_doc.SearchLagrangian.doc);
  }

  {
    // clf_cbf_utils
    m.def(
        "GetPolynomialSolutions",
        [](const solvers::MathematicalProgramResult& result,
            const VectorX<symbolic::Polynomial>& p, double zero_coeff_tol) {
          VectorX<symbolic::Polynomial> p_sol;
          analysis::GetPolynomialSolutions(result, p, zero_coeff_tol, &p_sol);
          return p_sol;
        },
        py::arg("result"), py::arg("p"), py::arg("zero_coeff_tol"));
  }

  {
    // Quadrotor2d.

    AddTemplateFunction(m, "ToQuadrotor2dTrigState",
        &analysis::ToQuadrotor2dTrigState<double>, GetPyParam<double>());
    AddTemplateFunction(
        m, "TrigDynamics",
        [](const analysis::Quadrotor2dTrigPlant<double>& quadrotor,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::Vector2d>& u) {
          return analysis::TrigDynamics<double>(quadrotor, x, u);
        },
        GetPyParam<double>());

    m.def(
        "TrigPolyDynamics",
        [](const analysis::Quadrotor2dTrigPlant<double>& quadrotor,
            const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 7, 1>>&
                x) {
          Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
          Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
          analysis::TrigPolyDynamics(quadrotor, x, &f, &G);
          return std::make_tuple(f, G);
        },
        py::arg("quadrotor"), py::arg("x"),
        pydrake_doc.drake.systems.analysis.TrigPolyDynamics.doc);

    m.def(
        "TrigPolyDynamicsTwinQuadrotor",
        [](const analysis::Quadrotor2dTrigPlant<double>& quadrotor,
            const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 12, 1>>&
                x) {
          Eigen::Matrix<symbolic::Polynomial, 12, 1> f;
          Eigen::Matrix<symbolic::Polynomial, 12, 4> G;
          analysis::TrigPolyDynamicsTwinQuadrotor(quadrotor, x, &f, &G);
          return std::make_tuple(f, G);
        },
        py::arg("quadrotor"), py::arg("x"),
        pydrake_doc.drake.systems.analysis.TrigPolyDynamicsTwinQuadrotor.doc);

    m.def(
        "EquilibriumThrust",
        [](const analysis::Quadrotor2dTrigPlant<double>& quadrotor) {
          return analysis::EquilibriumThrust(quadrotor);
        },
        py::arg("quadrotor"),
        pydrake_doc.drake.systems.analysis.EquilibriumThrust.doc);

    m.def("Quadrotor2dStateEqConstraint",
        &analysis::Quadrotor2dStateEqConstraint, py::arg("x"),
        pydrake_doc.drake.systems.analysis.Quadrotor2dStateEqConstraint.doc);

    m.def("TwinQuadrotor2dStateEqConstraint",
        &analysis::TwinQuadrotor2dStateEqConstraint, py::arg("x"),
        pydrake_doc.drake.systems.analysis.TwinQuadrotor2dStateEqConstraint
            .doc);
  }
}

}  // namespace pydrake
}  // namespace drake
