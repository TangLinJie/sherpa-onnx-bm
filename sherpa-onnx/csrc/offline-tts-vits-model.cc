// sherpa-onnx/csrc/offline-tts-vits-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-vits-model.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "bmruntime_cpp.h"
#include "bmruntime_interface.h"

using namespace bmruntime;

namespace sherpa_onnx {

class OfflineTtsVitsModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config.vits.model);
    Init(buf.data(), buf.size());
    // context
    dev_id = config_.devid;
    m_ctx = std::make_unique<bmruntime::Context>(dev_id);
    std::string bmodel_path = config_.bmodel_path;
    assert(BM_SUCCESS == m_ctx->load_bmodel(bmodel_path.c_str()));

    // network
    std::vector<const char*> net_names;
    m_ctx->get_network_names(&net_names);
    m_net = std::make_unique<bmruntime::Network>(*m_ctx, net_names[0], 0);
    m_handle = m_ctx->handle();
    m_inputs = m_net->Inputs();
    m_outputs = m_net->Outputs();
    m_netinfo = m_ctx->get_network_info(net_names[0]);
    struct bm_misc_info misc_info;
    assert(BM_SUCCESS == bm_get_misc_info(m_handle, &misc_info));
    is_soc = misc_info.pcie_soc_mode == 1;
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.vits.model);
    Init(buf.data(), buf.size());
  }
#endif

  Ort::Value Run(Ort::Value x, int64_t sid, float speed) {
    if (meta_data_.is_piper || meta_data_.is_coqui) {
      return RunVitsPiperOrCoqui(std::move(x), sid, speed);
    }

    return RunVits(std::move(x), sid, speed);
  }

  Ort::Value Run(Ort::Value x, Ort::Value tones, int64_t sid, float speed) {
    // For MeloTTS, we hardcode sid to the one contained in the meta data
    sid = meta_data_.speaker_id;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> x_shape = x.GetTensorTypeAndShapeInfo().GetShape();
    if (x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Support only batch_size == 1. Given: %d",
                       static_cast<int32_t>(x_shape[0]));
      exit(-1);
    }

    int64_t len = x_shape[1];
    int64_t len_shape = 1;

    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, &len, 1, &len_shape, 1);

    int64_t scale_shape = 1;
    float noise_scale = config_.vits.noise_scale;
    float length_scale = config_.vits.length_scale;
    float noise_scale_w = config_.vits.noise_scale_w;

    if (speed != 1 && speed > 0) {
      length_scale = 1. / speed;
    }

    Ort::Value noise_scale_tensor =
        Ort::Value::CreateTensor(memory_info, &noise_scale, 1, &scale_shape, 1);

    Ort::Value length_scale_tensor = Ort::Value::CreateTensor(
        memory_info, &length_scale, 1, &scale_shape, 1);

    Ort::Value noise_scale_w_tensor = Ort::Value::CreateTensor(
        memory_info, &noise_scale_w, 1, &scale_shape, 1);

    Ort::Value sid_tensor =
        Ort::Value::CreateTensor(memory_info, &sid, 1, &scale_shape, 1);

    std::vector<Ort::Value> inputs;
    inputs.reserve(7);
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(x_length));
    inputs.push_back(std::move(tones));
    inputs.push_back(std::move(sid_tensor));
    inputs.push_back(std::move(noise_scale_tensor));
    inputs.push_back(std::move(length_scale_tensor));
    inputs.push_back(std::move(noise_scale_w_tensor));

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    return std::move(out[0]);
  }

  const OfflineTtsVitsModelMetaData &GetMetaData() const { return meta_data_; }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---vits model---\n";
      PrintModelMetadata(os, meta_data);

      os << "----------input names----------\n";
      int32_t i = 0;
      for (const auto &s : input_names_) {
        os << i << " " << s << "\n";
        ++i;
      }
      os << "----------output names----------\n";
      i = 0;
      for (const auto &s : output_names_) {
        os << i << " " << s << "\n";
        ++i;
      }

      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_data_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.add_blank, "add_blank",
                                            0);

    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.speaker_id, "speaker_id",
                                            0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.version, "version", 0);
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_speakers, "n_speakers");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.punctuations,
                                                "punctuation", "");
    SHERPA_ONNX_READ_META_DATA_STR(meta_data_.language, "language");

    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.voice, "voice", "");

    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.frontend, "frontend",
                                                "");

    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.jieba, "jieba", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.blank_id, "blank_id", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.bos_id, "bos_id", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.eos_id, "eos_id", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.use_eos_bos,
                                            "use_eos_bos", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.pad_id, "pad_id", 0);

    std::string comment;
    SHERPA_ONNX_READ_META_DATA_STR(comment, "comment");

    if (comment.find("piper") != std::string::npos) {
      meta_data_.is_piper = true;
    }

    if (comment.find("coqui") != std::string::npos) {
      meta_data_.is_coqui = true;
    }

    if (comment.find("icefall") != std::string::npos) {
      meta_data_.is_icefall = true;
    }

    if (comment.find("melo") != std::string::npos) {
      meta_data_.is_melo_tts = true;
      int32_t expected_version = 2;
      if (meta_data_.version < expected_version) {
        SHERPA_ONNX_LOGE(
            "Please download the latest MeloTTS model and retry. Current "
            "version: %d. Expected version: %d",
            meta_data_.version, expected_version);
        exit(-1);
      }

      // NOTE(fangjun):
      // version 0 is the first version
      // version 2: add jieba=1 to the metadata
    }
  }

  Ort::Value RunVitsPiperOrCoqui(Ort::Value x, int64_t sid, float speed) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> x_shape = x.GetTensorTypeAndShapeInfo().GetShape();
    if (x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Support only batch_size == 1. Given: %d",
                       static_cast<int32_t>(x_shape[0]));
      exit(-1);
    }

    int64_t len = x_shape[1];
    int64_t len_shape = 1;

    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, &len, 1, &len_shape, 1);

    float noise_scale = config_.vits.noise_scale;
    float length_scale = config_.vits.length_scale;
    float noise_scale_w = config_.vits.noise_scale_w;

    if (speed != 1 && speed > 0) {
      length_scale = 1. / speed;
    }
    std::array<float, 3> scales = {noise_scale, length_scale, noise_scale_w};

    int64_t scale_shape = 3;

    Ort::Value scales_tensor = Ort::Value::CreateTensor(
        memory_info, scales.data(), scales.size(), &scale_shape, 1);

    int64_t sid_shape = 1;
    Ort::Value sid_tensor =
        Ort::Value::CreateTensor(memory_info, &sid, 1, &sid_shape, 1);

    int64_t lang_id_shape = 1;
    int64_t lang_id = 0;
    Ort::Value lang_id_tensor =
        Ort::Value::CreateTensor(memory_info, &lang_id, 1, &lang_id_shape, 1);

    std::vector<Ort::Value> inputs;
    inputs.reserve(5);
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(x_length));
    inputs.push_back(std::move(scales_tensor));

    if (input_names_.size() >= 4 && input_names_[3] == "sid") {
      inputs.push_back(std::move(sid_tensor));
    }

    if (input_names_.size() >= 5 && input_names_[4] == "langid") {
      inputs.push_back(std::move(lang_id_tensor));
    }

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());
    // bmrt
    int64_t* floatarr = nullptr;
    floatarr = inputs[0].GetTensorMutableData<int64_t>();
    int64_t output_tensor_size = 1;
    for (auto& it : x_shape)
    {
      output_tensor_size *= it;
    }
    std::vector<int>x_array(output_tensor_size);
    for (unsigned i = 0; i < output_tensor_size; i++)
    {
      x_array[i] = floatarr[i];
    }
    int int_len = len;
    m_inputs[0]->CopyFrom((void *)x_array.data());
    m_inputs[1]->CopyFrom((void *)&int_len);
    m_inputs[2]->CopyFrom((void *)scales.data());
    assert(BM_SUCCESS == m_net->Forward(true));
    int last_dim = m_outputs[0]->tensor()->shape.dims[3];
    std::vector<float>bmrt_out(last_dim);
    m_outputs[0]->CopyTo(bmrt_out.data());

    /*
    float* out_floatarr = out[0].GetTensorMutableData<float>();
    std::vector<float>output_array(439552);
    for (unsigned i = 0; i < 439552; i++)
    {
      output_array[i] = out_floatarr[i];
    }
    */
    std::vector<int64_t> bmrt_out_shape{1, 1, 1, last_dim};
    out[0] = Ort::Value::CreateTensor(
        memory_info, bmrt_out.data(), last_dim, bmrt_out_shape.data(), 4);

    return std::move(out[0]);
  }

  Ort::Value RunVits(Ort::Value x, int64_t sid, float speed) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> x_shape = x.GetTensorTypeAndShapeInfo().GetShape();
    if (x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Support only batch_size == 1. Given: %d",
                       static_cast<int32_t>(x_shape[0]));
      exit(-1);
    }

    int64_t len = x_shape[1];
    int64_t len_shape = 1;

    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, &len, 1, &len_shape, 1);

    int64_t scale_shape = 1;
    float noise_scale = config_.vits.noise_scale;
    float length_scale = config_.vits.length_scale;
    float noise_scale_w = config_.vits.noise_scale_w;

    if (speed != 1 && speed > 0) {
      length_scale = 1. / speed;
    }

    Ort::Value noise_scale_tensor =
        Ort::Value::CreateTensor(memory_info, &noise_scale, 1, &scale_shape, 1);

    Ort::Value length_scale_tensor = Ort::Value::CreateTensor(
        memory_info, &length_scale, 1, &scale_shape, 1);

    Ort::Value noise_scale_w_tensor = Ort::Value::CreateTensor(
        memory_info, &noise_scale_w, 1, &scale_shape, 1);

    Ort::Value sid_tensor =
        Ort::Value::CreateTensor(memory_info, &sid, 1, &scale_shape, 1);

    std::vector<Ort::Value> inputs;
    inputs.reserve(6);
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(x_length));
    inputs.push_back(std::move(noise_scale_tensor));
    inputs.push_back(std::move(length_scale_tensor));
    inputs.push_back(std::move(noise_scale_w_tensor));

    if (input_names_.size() == 6 &&
        (input_names_.back() == "sid" || input_names_.back() == "speaker")) {
      inputs.push_back(std::move(sid_tensor));
    }

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    return std::move(out[0]);
  }

 private:
  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  // inference
  int dev_id;
  bm_handle_t m_handle;
  const bm_net_info_t* m_netinfo;
  bool is_soc;
  std::unique_ptr<bmruntime::Context> m_ctx;
  std::unique_ptr<bmruntime::Network> m_net;
  std::vector<bmruntime::Tensor*> m_inputs;
  std::vector<bmruntime::Tensor*> m_outputs;

  OfflineTtsVitsModelMetaData meta_data_;
};

OfflineTtsVitsModel::OfflineTtsVitsModel(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineTtsVitsModel::OfflineTtsVitsModel(AAssetManager *mgr,
                                         const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineTtsVitsModel::~OfflineTtsVitsModel() = default;

Ort::Value OfflineTtsVitsModel::Run(Ort::Value x, int64_t sid /*=0*/,
                                    float speed /*= 1.0*/) {
  return impl_->Run(std::move(x), sid, speed);
}

Ort::Value OfflineTtsVitsModel::Run(Ort::Value x, Ort::Value tones,
                                    int64_t sid /*= 0*/,
                                    float speed /*= 1.0*/) {
  return impl_->Run(std::move(x), std::move(tones), sid, speed);
}

const OfflineTtsVitsModelMetaData &OfflineTtsVitsModel::GetMetaData() const {
  return impl_->GetMetaData();
}

}  // namespace sherpa_onnx
