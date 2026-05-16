/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "ort_inference.h"

#include <modelbox/base/log.h>

#include <algorithm>
#include <cstring>
#include <utility>

#include "virtualdriver_inference.h"

#ifdef MODELBOX_ORT_WITH_CUDA
#include <cuda_runtime.h>
#endif

OrtInference::OrtInference(std::string backend)
    : backend_(std::move(backend)),
      env_(ORT_LOGGING_LEVEL_WARNING, "modelbox-ort") {}

OrtInference::~OrtInference() = default;

modelbox::ModelBoxDataType OrtInference::OrtTypeToModelBox(
    ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return modelbox::MODELBOX_FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return modelbox::MODELBOX_HALF;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return modelbox::MODELBOX_DOUBLE;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return modelbox::MODELBOX_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return modelbox::MODELBOX_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return modelbox::MODELBOX_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return modelbox::MODELBOX_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return modelbox::MODELBOX_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return modelbox::MODELBOX_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return modelbox::MODELBOX_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return modelbox::MODELBOX_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return modelbox::MODELBOX_BOOL;
    default:                                    return modelbox::MODELBOX_TYPE_INVALID;
  }
}

size_t OrtInference::OrtTypeSize(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return 8;
    default:                                    return 0;
  }
}

modelbox::Status OrtInference::GetFlowUnitIO(
    const std::shared_ptr<modelbox::FlowUnitDesc> &flowunit_desc) {
  auto unit_desc =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(flowunit_desc);
  if (unit_desc == nullptr) {
    return {modelbox::STATUS_BADCONF,
            "onnxruntime: cast to VirtualInferenceFlowUnitDesc failed"};
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
            "onnxruntime: flowunit has no inputs or no outputs"};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status OrtInference::BuildSessionOptions(Ort::SessionOptions &opts) {
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Best-effort EP selection. If the requested provider wasn't built into this
  // ORT binary, AppendExecutionProvider* throws — we log and fall through to
  // the default CPU provider.
  if (backend_ == "CUDA") {
    try {
      OrtCUDAProviderOptions cuda{};
      opts.AppendExecutionProvider_CUDA(cuda);
      MBLOG_INFO << "onnxruntime: enabled execution provider CUDA";
    } catch (const std::exception &e) {
      MBLOG_WARN << "onnxruntime: CUDA EP not available: " << e.what();
    }
  } else if (backend_ == "CoreML") {
    try {
      std::unordered_map<std::string, std::string> coreml_options;
      coreml_options["MLComputeUnits"] = "ALL";
      opts.AppendExecutionProvider("CoreML", coreml_options);
      MBLOG_INFO << "onnxruntime: enabled execution provider CoreML";
    } catch (const std::exception &e) {
      MBLOG_WARN << "onnxruntime: CoreML EP not available: " << e.what();
    }
  } else if (backend_ == "ROCM") {
    try {
      OrtROCMProviderOptions rocm{};
      opts.AppendExecutionProvider_ROCM(rocm);
      MBLOG_INFO << "onnxruntime: enabled execution provider ROCM";
    } catch (const std::exception &e) {
      MBLOG_WARN << "onnxruntime: ROCM EP not available: " << e.what();
    }
  } else if (backend_ != "CPU") {
    MBLOG_WARN << "onnxruntime: unknown backend '" << backend_
               << "', falling back to CPU";
  }
  return modelbox::STATUS_OK;
}

modelbox::Status OrtInference::Open(
    const std::shared_ptr<modelbox::Configuration> &opts,
    std::shared_ptr<modelbox::FlowUnitDesc> flowunit_desc) {
  auto status = GetFlowUnitIO(flowunit_desc);
  if (!status) return status;

  auto unit_desc =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(flowunit_desc);
  model_entry_ = unit_desc->GetModelEntry();
  if (model_entry_.empty()) {
    return {modelbox::STATUS_BADCONF, "onnxruntime: model entry is empty"};
  }

  try {
    Ort::SessionOptions session_options;
    status = BuildSessionOptions(session_options);
    if (!status) return status;
    session_ = std::unique_ptr<Ort::Session>(
        new Ort::Session(env_, model_entry_.c_str(), session_options));
  } catch (const std::exception &e) {
    auto msg = std::string("onnxruntime: failed to load model '") +
               model_entry_ + "' on backend '" + backend_ + "': " + e.what();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  // Snapshot model IO names + types up front so Infer() is allocation-free
  // outside the tensor bodies.
  size_t n_in = session_->GetInputCount();
  size_t n_out = session_->GetOutputCount();
  model_input_names_owned_.reserve(n_in);
  model_output_names_owned_.reserve(n_out);
  model_input_names_.reserve(n_in);
  model_output_names_.reserve(n_out);
  model_input_shapes_.reserve(n_in);
  model_input_types_.reserve(n_in);
  model_output_types_.reserve(n_out);
  for (size_t i = 0; i < n_in; ++i) {
    auto name = session_->GetInputNameAllocated(i, allocator_);
    model_input_names_.push_back(name.get());
    model_input_names_owned_.push_back(std::move(name));
    Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
    auto info = type_info.GetTensorTypeAndShapeInfo();
    model_input_shapes_.push_back(info.GetShape());
    model_input_types_.push_back(info.GetElementType());
  }
  for (size_t i = 0; i < n_out; ++i) {
    auto name = session_->GetOutputNameAllocated(i, allocator_);
    model_output_names_.push_back(name.get());
    model_output_names_owned_.push_back(std::move(name));
    Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
    auto info = type_info.GetTensorTypeAndShapeInfo();
    model_output_types_.push_back(info.GetElementType());
  }

  if (n_in != io_list_.input_name_list.size()) {
    return {modelbox::STATUS_BADCONF,
            "onnxruntime: model has " + std::to_string(n_in) +
                " inputs but flowunit declares " +
                std::to_string(io_list_.input_name_list.size())};
  }
  if (n_out != io_list_.output_name_list.size()) {
    return {modelbox::STATUS_BADCONF,
            "onnxruntime: model has " + std::to_string(n_out) +
                " outputs but flowunit declares " +
                std::to_string(io_list_.output_name_list.size())};
  }

  MBLOG_INFO << "onnxruntime: loaded " << model_entry_ << " backend="
             << backend_ << " (" << n_in << " inputs, " << n_out
             << " outputs)";
  return modelbox::STATUS_OK;
}

modelbox::Status OrtInference::RunOnce(
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
#ifdef MODELBOX_ORT_WITH_CUDA
  const bool on_cuda = (backend_ == "CUDA");
#else
  const bool on_cuda = false;
#endif
  Ort::MemoryInfo mem_info = on_cuda
      ? Ort::MemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault)
      : Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> inputs;
  inputs.reserve(io_list_.input_name_list.size());

  for (size_t i = 0; i < io_list_.input_name_list.size(); ++i) {
    const auto &port = io_list_.input_name_list[i];
    auto bufs = data_ctx->Input(port);
    if (bufs == nullptr || bufs->Size() == 0) {
      return {modelbox::STATUS_FAULT,
              "onnxruntime: missing input buffer for port " + port};
    }
    auto buf = bufs->At(0);
    if (buf == nullptr || buf->ConstData() == nullptr) {
      return {modelbox::STATUS_FAULT,
              "onnxruntime: input buffer is null for port " + port};
    }
    auto type = model_input_types_[i];
    size_t elem_bytes = OrtTypeSize(type);
    if (elem_bytes == 0) {
      return {modelbox::STATUS_FAULT,
              "onnxruntime: unsupported element type for input " + port};
    }
    // Compute concrete shape: take model's static shape, fill any dynamic
    // dimension by dividing remaining bytes equally across them. The common
    // case is a single dynamic batch dim → batch = bytes / (static_elems *
    // elem_bytes).
    auto shape = model_input_shapes_[i];
    size_t static_elems = 1;
    int dynamic_dims = 0;
    for (auto d : shape) {
      if (d < 0) ++dynamic_dims;
      else static_elems *= static_cast<size_t>(d);
    }
    size_t total_bytes = buf->GetBytes();
    if (dynamic_dims == 1 && static_elems > 0) {
      size_t derived = total_bytes / (static_elems * elem_bytes);
      for (auto &d : shape) if (d < 0) { d = static_cast<int64_t>(derived); break; }
    } else if (dynamic_dims > 1) {
      return {modelbox::STATUS_BADCONF,
              "onnxruntime: input " + port + " has multiple dynamic dims; "
              "cannot derive shape from buffer size"};
    } else if (total_bytes != static_elems * elem_bytes) {
      return {modelbox::STATUS_FAULT,
              "onnxruntime: input " + port + " bytes mismatch (got " +
                  std::to_string(total_bytes) + ", expected " +
                  std::to_string(static_elems * elem_bytes) + ")"};
    }

    void *data = const_cast<void *>(buf->ConstData());
    inputs.emplace_back(Ort::Value::CreateTensor(
        mem_info, data, total_bytes, shape.data(), shape.size(), type));
  }

  // Bind outputs to CPU memory so that EPs that allocate on device (e.g. CUDA)
  // copy results back before we read them. Without this the downstream
  // std::memcpy reads a device pointer from CPU code and segfaults.
  std::vector<Ort::Value> outputs;
  try {
    Ort::IoBinding binding(*session_);
    for (size_t i = 0; i < inputs.size(); ++i) {
      binding.BindInput(model_input_names_[i], inputs[i]);
    }
    for (size_t i = 0; i < model_output_names_.size(); ++i) {
      binding.BindOutput(model_output_names_[i], mem_info);
    }
    session_->Run(Ort::RunOptions{nullptr}, binding);
    outputs = binding.GetOutputValues();
  } catch (const std::exception &e) {
    return {modelbox::STATUS_FAULT,
            std::string("onnxruntime: session->Run threw: ") + e.what()};
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto info = outputs[i].GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    auto type = info.GetElementType();
    size_t total_bytes =
        info.GetElementCount() * OrtTypeSize(type);

    auto out_list = data_ctx->Output(io_list_.output_name_list[i]);
    if (out_list == nullptr) {
      return {modelbox::STATUS_FAULT, "onnxruntime: missing output buffer list"};
    }
    auto status = out_list->Build({total_bytes});
    if (!status) return status;
    auto out_buf = out_list->At(0);
    auto *dst = out_buf->MutableData();
    if (dst == nullptr) {
      return {modelbox::STATUS_FAULT, "onnxruntime: output buffer null"};
    }
#ifdef MODELBOX_ORT_WITH_CUDA
    if (on_cuda) {
      auto cuda_ret = cudaMemcpy(dst, outputs[i].GetTensorMutableData<void>(),
                                 total_bytes, cudaMemcpyDeviceToDevice);
      if (cuda_ret != cudaSuccess) {
        return {modelbox::STATUS_FAULT,
                std::string("onnxruntime: cudaMemcpy(D2D) for output ") +
                    io_list_.output_name_list[i] + " failed: " +
                    cudaGetErrorString(cuda_ret)};
      }
    } else {
      std::memcpy(dst, outputs[i].GetTensorMutableData<void>(), total_bytes);
    }
#else
    std::memcpy(dst, outputs[i].GetTensorMutableData<void>(), total_bytes);
#endif

    std::vector<size_t> shape_vec;
    shape_vec.reserve(shape.size());
    for (auto d : shape) shape_vec.push_back(static_cast<size_t>(d));
    out_buf->Set("shape", shape_vec);
    out_buf->Set("type", OrtTypeToModelBox(type));
  }
  return modelbox::STATUS_OK;
}

modelbox::Status OrtInference::Infer(
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
  if (session_ == nullptr) {
    return {modelbox::STATUS_FAULT, "onnxruntime: session not initialized"};
  }
  return RunOnce(data_ctx);
}
