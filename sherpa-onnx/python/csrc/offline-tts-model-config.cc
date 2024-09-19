// sherpa-onnx/python/csrc/offline-tts-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-tts-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tts-vits-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsModelConfig(py::module *m) {
  PybindOfflineTtsVitsModelConfig(m);

  using PyClass = OfflineTtsModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsModelConfig")
      .def(py::init<>())
      .def(py::init<const OfflineTtsVitsModelConfig &, const std::string, const int32_t, int32_t, bool,
                    const std::string &>(),
           py::arg("vits"), py::arg("bmodel_path"), py::arg("devid") = 0, py::arg("num_threads") = 1,
           py::arg("debug") = false, py::arg("provider") = "cpu")
      .def_readwrite("vits", &PyClass::vits)
      .def_readwrite("bmodel_path", &PyClass::bmodel_path)
      .def_readwrite("devid", &PyClass::devid)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
