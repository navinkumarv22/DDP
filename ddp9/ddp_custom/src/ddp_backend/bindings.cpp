// ddp_backend_bindings.cpp  (your pybind file)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstddef>
#include <cstring>
#include "process_group_nccl.h"
#include "macros.h"

namespace py = pybind11;

// C linkage kernel if you use it elsewhere
extern "C" void launch_scale_f32(float*, float, size_t, cudaStream_t);

PYBIND11_MODULE(ddp_backend, m) {
  py::class_<ncclUniqueId>(m, "NcclId");

  m.def("nccl_get_unique_id", [](){
    ncclUniqueId id; NCCL_CHECK(ncclGetUniqueId(&id));
    return py::bytes(reinterpret_cast<char*>(&id), sizeof(ncclUniqueId));
  });

  m.def("nccl_id_from_bytes", [](py::bytes b){
    ncclUniqueId id{};
    std::string s = b;
    if (s.size() != sizeof(ncclUniqueId)) throw std::runtime_error("bad id size");
    std::memcpy(&id, s.data(), s.size());
    return id;
  });

  py::class_<PGOptions>(m, "PGOptions")
    .def(py::init<>())
    .def_readwrite("device", &PGOptions::device)
    .def_readwrite("rank", &PGOptions::rank)
    .def_readwrite("world", &PGOptions::world)
    .def_readwrite("stream_priority", &PGOptions::stream_priority);

  py::class_<ProcessGroupNCCL>(m, "ProcessGroupNCCL")
    .def(py::init<const ncclUniqueId&, const PGOptions&>())
    .def_property_readonly("rank", &ProcessGroupNCCL::rank)
    .def_property_readonly("world_size", &ProcessGroupNCCL::world_size)

    // Pointer-based (float*) API
    .def("allreduce_float32", &ProcessGroupNCCL::allreduce_float32)
    .def("broadcast_float32", &ProcessGroupNCCL::broadcast_float32)
    .def("barrier", &ProcessGroupNCCL::barrier)

    // Address-based (size_t) API — easiest to call from Python
    .def("allreduce_addr_f32", [](ProcessGroupNCCL& self, size_t addr, size_t count){
        auto* p = reinterpret_cast<float*>(addr);
        self.allreduce_float32(p, count);
      })
    .def("broadcast_addr_f32", [](ProcessGroupNCCL& self, size_t addr, size_t count, int root){
        auto* p = reinterpret_cast<float*>(addr);
        self.broadcast_float32(p, count, root);
      })

    // Short aliases (some wrappers search for these simple names):
    .def("allreduce_addr", [](ProcessGroupNCCL& self, size_t addr, size_t count){
        auto* p = reinterpret_cast<float*>(addr);
        self.allreduce_float32(p, count);
      })
    .def("broadcast_addr", [](ProcessGroupNCCL& self, size_t addr, size_t count, int root){
        auto* p = reinterpret_cast<float*>(addr);
        self.broadcast_float32(p, count, root);
      });
}
