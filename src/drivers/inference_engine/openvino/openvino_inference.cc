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

#include "openvino_inference.h"

#include <modelbox/base/log.h>

#include <utility>

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "virtualdriver_inference.h"

OpenVINOInference::OpenVINOInference(std::string target_device)
    : target_device_(std::move(target_device)) {}

OpenVINOInference::~OpenVINOInference() = default;

modelbox::ModelBoxDataType OpenVINOInference::OvElementToModelBox(
    ov::element::Type type) {
  if (type == ov::element::f32) return modelbox::MODELBOX_FLOAT;
  if (type == ov::element::f16) return modelbox::MODELBOX_HALF;
  if (type == ov::element::f64) return modelbox::MODELBOX_DOUBLE;
  if (type == ov::element::i8) return modelbox::MODELBOX_INT8;
  if (type == ov::element::u8) return modelbox::MODELBOX_UINT8;
  if (type == ov::element::i16) return modelbox::MODELBOX_INT16;
  if (type == ov::element::u16) return modelbox::MODELBOX_UINT16;
  if (type == ov::element::i32) return modelbox::MODELBOX_INT32;
  if (type == ov::element::u32) return modelbox::MODELBOX_UINT32;
  if (type == ov::element::i64) return modelbox::MODELBOX_INT64;
  if (type == ov::element::u64) return modelbox::MODELBOX_UINT64;
  if (type == ov::element::boolean) return modelbox::MODELBOX_BOOL;
  return modelbox::MODELBOX_TYPE_INVALID;
}

size_t OpenVINOInference::OvElementSize(ov::element::Type type) {
  return type.size();
}

modelbox::Status OpenVINOInference::GetFlowUnitIO(
    const std::shared_ptr<modelbox::FlowUnitDesc> &flowunit_desc) {
  auto unit_desc =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(flowunit_desc);
  if (unit_desc == nullptr) {
    return {modelbox::STATUS_BADCONF,
            "openvino: cast to VirtualInferenceFlowUnitDesc failed"};
  }

  for (auto &input : unit_desc->GetFlowUnitInput()) {
    io_list_.input_name_list.push_back(input.GetPortName());
    io_list_.input_type_list.push_back(input.GetPortType());
  }
  for (auto &output : unit_desc->GetFlowUnitOutput()) {
    io_list_.output_name_list.push_back(output.GetPortName());
    io_list_.output_type_list.push_back(output.GetPortType());
  }

  if (io_list_.input_name_list.empty() || io_list_.output_name_list.empty()) {
    return {modelbox::STATUS_BADCONF,
            "openvino: flowunit has no inputs or no outputs"};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status OpenVINOInference::Open(
    const std::shared_ptr<modelbox::Configuration> &opts,
    std::shared_ptr<modelbox::FlowUnitDesc> flowunit_desc) {
  auto status = GetFlowUnitIO(flowunit_desc);
  if (!status) {
    return status;
  }

  auto unit_desc =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(flowunit_desc);
  model_entry_ = unit_desc->GetModelEntry();
  if (model_entry_.empty()) {
    return {modelbox::STATUS_BADCONF, "openvino: model entry is empty"};
  }

  // Optional built-in preprocessing baked into the model graph.
  preprocess_mode_ = opts->GetString("preprocess", "");

  try {
    auto model = core_.read_model(model_entry_);

    if (preprocess_mode_ == "ultralytics_image") {
      // Bake BGR uint8 NHWC at any resolution -> resize -> /255 -> NCHW float
      // into the compiled graph itself, so the host pushes raw decoded frames
      // and OpenVINO does the rest on whichever device runs inference.
      ov::preprocess::PrePostProcessor ppp(model);
      auto& in = ppp.input();
      in.tensor()
          .set_element_type(ov::element::u8)
          .set_layout("NHWC")
          .set_spatial_dynamic_shape();
      in.preprocess()
          .convert_element_type(ov::element::f32)
          .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
          .scale(255.0F);
      in.model().set_layout("NCHW");
      model = ppp.build();
      MBLOG_INFO << "openvino: preprocess=ultralytics_image (resize + /255 + "
                    "NHWC->NCHW baked into graph)";
    }

    compiled_model_ = core_.compile_model(model, target_device_);
    infer_request_ = compiled_model_.create_infer_request();
  } catch (const std::exception &e) {
    auto msg = std::string("openvino: failed to load model '") + model_entry_ +
               "' on device '" + target_device_ + "': " + e.what();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  MBLOG_INFO << "openvino: loaded " << model_entry_ << " on device "
             << target_device_ << " (" << io_list_.input_name_list.size()
             << " inputs, " << io_list_.output_name_list.size() << " outputs)";
  return modelbox::STATUS_OK;
}

modelbox::Status OpenVINOInference::SetInputs(
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
  const auto &inputs = compiled_model_.inputs();
  if (inputs.size() != io_list_.input_name_list.size()) {
    auto msg = "openvino: model expects " + std::to_string(inputs.size()) +
               " inputs but flowunit has " +
               std::to_string(io_list_.input_name_list.size());
    return {modelbox::STATUS_BADCONF, msg};
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &port_name = io_list_.input_name_list[i];
    auto buf_list = data_ctx->Input(port_name);
    if (buf_list == nullptr || buf_list->Size() == 0) {
      return {modelbox::STATUS_FAULT,
              "openvino: missing input buffer list for port " + port_name};
    }
    auto buffer = buf_list->At(0);
    if (buffer == nullptr || buffer->ConstData() == nullptr) {
      return {modelbox::STATUS_FAULT,
              "openvino: input buffer is null for port " + port_name};
    }

    // Wrap the host-resident input buffer as an ov::Tensor without copy.
    // OpenVINO's GPU plugin internally uploads to the Arc device.
    auto element_type = inputs[i].get_element_type();
    ov::Shape shape;

    if (preprocess_mode_ == "ultralytics_image") {
      // Buffer is BGR uint8 HWC at the source frame's native resolution; the
      // graph has been wrapped to accept that and resize/normalize/transpose
      // on the device. Build a [1, H, W, 3] u8 tensor over the buffer bytes.
      int32_t width = 0, height = 0, channel = 3;
      buffer->Get("width", width);
      buffer->Get("height", height);
      buffer->Get("channel", channel);
      if (width <= 0 || height <= 0 || channel != 3) {
        return {modelbox::STATUS_FAULT,
                "openvino: ultralytics_image preprocess needs width/height "
                "image meta with 3 channels on input port " + port_name};
      }
      shape = ov::Shape{1, static_cast<size_t>(height),
                        static_cast<size_t>(width),
                        static_cast<size_t>(channel)};
    } else if (inputs[i].get_partial_shape().is_static()) {
      shape = inputs[i].get_shape();
    } else {
      // Generic dynamic path: 1D over the buffer bytes.
      size_t total_bytes = buffer->GetBytes();
      size_t element_bytes = OvElementSize(element_type);
      if (element_bytes == 0) {
        return {modelbox::STATUS_FAULT,
                "openvino: zero-size element for dynamic input"};
      }
      shape = ov::Shape{total_bytes / element_bytes};
    }

    void *data = const_cast<void *>(buffer->ConstData());
    ov::Tensor tensor(element_type, shape, data);
    try {
      infer_request_.set_input_tensor(i, tensor);
    } catch (const std::exception &e) {
      return {modelbox::STATUS_FAULT,
              std::string("openvino: set_input_tensor failed: ") + e.what()};
    }
  }
  return modelbox::STATUS_OK;
}

modelbox::Status OpenVINOInference::WriteOutputs(
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
  const auto &outputs = compiled_model_.outputs();
  if (outputs.size() != io_list_.output_name_list.size()) {
    auto msg = "openvino: model has " + std::to_string(outputs.size()) +
               " outputs but flowunit declares " +
               std::to_string(io_list_.output_name_list.size());
    return {modelbox::STATUS_BADCONF, msg};
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto tensor = infer_request_.get_output_tensor(i);
    auto bytes = tensor.get_byte_size();
    auto element_type = tensor.get_element_type();
    auto mb_type = OvElementToModelBox(element_type);

    auto out_buf_list = data_ctx->Output(io_list_.output_name_list[i]);
    if (out_buf_list == nullptr) {
      return {modelbox::STATUS_FAULT, "openvino: missing output buffer list"};
    }
    auto status = out_buf_list->Build({bytes});
    if (!status) {
      return status;
    }
    auto out_buf = out_buf_list->At(0);

    // Copy result bytes from OpenVINO's internal tensor into the modelbox
    // buffer. OpenVINO owns the inference output tensor's lifetime.
    auto *dst = out_buf->MutableData();
    if (dst == nullptr) {
      return {modelbox::STATUS_FAULT, "openvino: output buffer data is null"};
    }
    std::memcpy(dst, tensor.data(), bytes);

    // Stamp shape + type metadata for downstream flow units.
    std::vector<size_t> shape_vec;
    for (auto d : tensor.get_shape()) {
      shape_vec.push_back(d);
    }
    out_buf->Set("shape", shape_vec);
    out_buf->Set("type", mb_type);
  }
  return modelbox::STATUS_OK;
}

modelbox::Status OpenVINOInference::Infer(
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto status = SetInputs(data_ctx);
  if (!status) {
    return status;
  }

  try {
    infer_request_.infer();
  } catch (const std::exception &e) {
    auto msg = std::string("openvino: infer() threw: ") + e.what();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  return WriteOutputs(data_ctx);
}
