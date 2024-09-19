// sherpa-onnx/csrc/offline-tts-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/offline-tts-vits-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsModelConfig {
  OfflineTtsVitsModelConfig vits;
  std::string bmodel_path;
  int32_t devid = 0;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  OfflineTtsModelConfig() = default;

  OfflineTtsModelConfig(const OfflineTtsVitsModelConfig &vits, const std::string bmodel_path, const int devid,
                        int32_t num_threads, bool debug,
                        const std::string &provider)
      : vits(vits),
        bmodel_path(bmodel_path),
        devid(devid),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_
