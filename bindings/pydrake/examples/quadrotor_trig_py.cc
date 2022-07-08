#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/examples/quadrotor_trig/quadrotor.h"
// #include "drake/examples/quadrotor_trig/quadrotor3d_trig_clf_demo.h"
#include "drake/systems/primitives/affine_system.h"

namespace drake {
namespace pydrake {

PYBIND11_MODULE(quadrotor_trig, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::systems;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::examples::quadrotor_trig;
  constexpr auto& doc = pydrake_doc.drake.examples.quadrotor_trig;

  m.doc() = "Bindings for the QuadrotorTrigPlant example.";

  py::module::import("pydrake.systems.framework");
  py::module::import("pydrake.systems.primitives");

  // TODO(eric.cousineau): At present, we only bind doubles.
  // In the future, we will bind more scalar types, and enable scalar
  // conversion. Issue #7660.
  using T = double;

  py::class_<QuadrotorTrigPlant<T>, LeafSystem<T>>(
      m, "QuadrotorTrigPlant", doc.QuadrotorTrigPlant.doc)
      .def(py::init<>(), doc.QuadrotorTrigPlant.ctor.doc)
      .def("m", &QuadrotorTrigPlant<T>::mass, doc.QuadrotorTrigPlant.mass.doc)
      .def("g", &QuadrotorTrigPlant<T>::gravity, doc.QuadrotorTrigPlant.gravity.doc)
      .def("length", &QuadrotorTrigPlant<T>::length, doc.QuadrotorTrigPlant.length.doc)
      .def("kF", &QuadrotorTrigPlant<T>::kF, doc.QuadrotorTrigPlant.kF.doc)
      .def("kM", &QuadrotorTrigPlant<T>::kM, doc.QuadrotorTrigPlant.kM.doc)
      .def("inertia", &QuadrotorTrigPlant<T>::inertia, py_rvp::reference_internal,
          doc.QuadrotorTrigPlant.inertia.doc)
      .def("SynthesizeTrigLqr", &QuadrotorTrigPlant<T>::SynthesizeTrigLqr, doc.QuadrotorTrigPlant.SynthesizeTrigLqr.doc);

  // m.def("SynthesizeTrigLqr", &SynthesizeTrigLqr, doc.SynthesizeTrigLqr.doc);
}

}  // namespace pydrake
}  // namespace drake
