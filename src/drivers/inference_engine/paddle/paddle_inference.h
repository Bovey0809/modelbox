/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MODELBOX_ENGINE_PADDLE_INFERENCE_H_
#define MODELBOX_ENGINE_PADDLE_INFERENCE_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>

#include <memory>
#include <string>
#include <vector>

namespace paddle_infer {
class Predictor;
}

namespace modelbox {

struct PaddleInferenceParams {
  std::string model_file;
  std::string params_file;
  std::string device = "gpu";
  int gpu_id = 0;
  bool enable_trt = false;
  int trt_workspace_mb = 256;
  std::string trt_precision = "fp32";
  bool enable_mkldnn = false;
  int cpu_threads = 4;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

class PaddleInference {
 public:
  Status Init(const PaddleInferenceParams& p);
  Status Infer(const std::vector<std::shared_ptr<Buffer>>& inputs,
               std::vector<std::shared_ptr<Buffer>>& outputs,
               const std::shared_ptr<Device>& output_device);
  const std::vector<std::string>& InputNames() const { return input_names_; }
  const std::vector<std::string>& OutputNames() const { return output_names_; }

 private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string device_{"gpu"};
};

}  // namespace modelbox
#endif  // MODELBOX_ENGINE_PADDLE_INFERENCE_H_
