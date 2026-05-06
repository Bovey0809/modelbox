/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
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

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "coreml_inference.h"

#include <modelbox/base/log.h>

#include <cstring>
#include <utility>

#include "virtualdriver_inference.h"

namespace {

modelbox::ModelBoxDataType MlTypeToModelBox(MLMultiArrayDataType t) {
  switch (t) {
    case MLMultiArrayDataTypeFloat32:
      return modelbox::MODELBOX_FLOAT;
    case MLMultiArrayDataTypeFloat16:
      return modelbox::MODELBOX_HALF;
    case MLMultiArrayDataTypeDouble:
      return modelbox::MODELBOX_DOUBLE;
    case MLMultiArrayDataTypeInt32:
      return modelbox::MODELBOX_INT32;
    default:
      return modelbox::MODELBOX_TYPE_INVALID;
  }
}

size_t MlElementSize(MLMultiArrayDataType t) {
  switch (t) {
    case MLMultiArrayDataTypeFloat32:
      return 4;
    case MLMultiArrayDataTypeFloat16:
      return 2;
    case MLMultiArrayDataTypeDouble:
      return 8;
    case MLMultiArrayDataTypeInt32:
      return 4;
    default:
      return 0;
  }
}

NSString *NS(const std::string &s) {
  return [NSString stringWithUTF8String:s.c_str()];
}

}  // namespace

// Implementation holder. Kept opaque to the C++ header so the rest of the
// codebase doesn't need an Objective-C++ compilation flag.
class CoreMLInferenceImpl {
 public:
  // Strong refs (ARC retains under -fobjc-arc).
  MLModel *model = nil;
  // Per-input/output cached MLFeatureDescription so we don't re-resolve on
  // every Infer() call.
  NSArray<NSString *> *model_input_names = nil;
  NSArray<NSString *> *model_output_names = nil;
};

CoreMLInference::CoreMLInference(std::string target_device)
    : target_device_(std::move(target_device)),
      impl_(new CoreMLInferenceImpl()) {}

CoreMLInference::~CoreMLInference() = default;

modelbox::Status CoreMLInference::GetFlowUnitIO(
    const std::shared_ptr<modelbox::FlowUnitDesc> &flowunit_desc) {
  auto unit_desc =
      std::dynamic_pointer_cast<VirtualInferenceFlowUnitDesc>(flowunit_desc);
  if (unit_desc == nullptr) {
    return {modelbox::STATUS_BADCONF,
            "coreml: cast to VirtualInferenceFlowUnitDesc failed"};
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
            "coreml: flowunit has no inputs or no outputs"};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status CoreMLInference::Open(
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
    return {modelbox::STATUS_BADCONF, "coreml: model entry is empty"};
  }

  @autoreleasepool {
    NSError *err = nil;
    NSURL *url = [NSURL fileURLWithPath:NS(model_entry_)];

    // .mlpackage (Ultralytics export) and .mlmodel both load via this API;
    // CoreML auto-compiles to .mlmodelc on first load. For repeated runs
    // a future commit can cache the compiled URL via
    // [MLModel compileModelAtURL:].
    MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsAll;  // CPU + GPU + ANE

    impl_->model = [MLModel modelWithContentsOfURL:url
                                     configuration:cfg
                                             error:&err];
    if (impl_->model == nil) {
      auto msg = std::string("coreml: failed to load '") + model_entry_ +
                 "': " + (err ? err.localizedDescription.UTF8String
                              : "unknown error");
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    impl_->model_input_names =
        impl_->model.modelDescription.inputDescriptionsByName.allKeys;
    impl_->model_output_names =
        impl_->model.modelDescription.outputDescriptionsByName.allKeys;
  }

  MBLOG_INFO << "coreml: loaded " << model_entry_ << " on " << target_device_
             << " (" << io_list_.input_name_list.size() << " inputs, "
             << io_list_.output_name_list.size() << " outputs)";
  return modelbox::STATUS_OK;
}

modelbox::Status CoreMLInference::Infer(
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
  if (impl_->model == nil) {
    return {modelbox::STATUS_FAULT, "coreml: not opened"};
  }

  @autoreleasepool {
    NSMutableDictionary<NSString *, MLFeatureValue *> *features =
        [NSMutableDictionary dictionary];

    // Map each declared modelbox input port to a CoreML feature. Wrap the
    // host-resident buffer as an MLMultiArray pointing at the same memory
    // (no copy) using initWithDataPointer:shape:dataType:strides:...
    NSDictionary<NSString *, MLFeatureDescription *> *inputs_desc =
        impl_->model.modelDescription.inputDescriptionsByName;

    for (size_t i = 0; i < io_list_.input_name_list.size(); ++i) {
      const auto &port = io_list_.input_name_list[i];
      // Map modelbox port name -> model feature name. Default: same name. If
      // the model has only one input, use it regardless of port name.
      NSString *feature_name = NS(port);
      if (inputs_desc[feature_name] == nil) {
        if (impl_->model_input_names.count == 1) {
          feature_name = impl_->model_input_names[0];
        } else {
          return {modelbox::STATUS_BADCONF,
                  std::string("coreml: input port '") + port +
                      "' not found in model"};
        }
      }

      MLFeatureDescription *fd = inputs_desc[feature_name];
      if (fd.type != MLFeatureTypeMultiArray) {
        return {modelbox::STATUS_BADCONF,
                std::string("coreml: input '") + feature_name.UTF8String +
                    "' is not an MLMultiArray (only multiarray inputs are "
                    "supported in this driver)"};
      }

      auto buf_list = data_ctx->Input(port);
      if (buf_list == nullptr || buf_list->Size() == 0) {
        return {modelbox::STATUS_FAULT,
                "coreml: missing input buffer list for port " + port};
      }
      auto buffer = buf_list->At(0);
      if (buffer == nullptr || buffer->ConstData() == nullptr) {
        return {modelbox::STATUS_FAULT,
                "coreml: input buffer is null for port " + port};
      }

      MLMultiArrayConstraint *con = fd.multiArrayConstraint;
      MLMultiArrayDataType dtype = con.dataType;
      NSArray<NSNumber *> *shape_arr = con.shape;

      // Build row-major strides matching the declared shape. CoreML's API
      // wants element strides (not byte strides). For shape [s0, s1, ..., sN]
      // strides are [s1*s2*...*sN, s2*...*sN, ..., sN, 1].
      NSMutableArray<NSNumber *> *strides =
          [NSMutableArray arrayWithCapacity:shape_arr.count];
      NSInteger acc = 1;
      for (NSInteger k = (NSInteger)shape_arr.count - 1; k >= 0; --k) {
        [strides insertObject:@(acc) atIndex:0];
        acc *= shape_arr[k].integerValue;
      }

      NSError *err = nil;
      MLMultiArray *arr = [[MLMultiArray alloc]
          initWithDataPointer:const_cast<void *>(buffer->ConstData())
                        shape:shape_arr
                     dataType:dtype
                      strides:strides
                  deallocator:^(void *){
                      // Modelbox owns the buffer; no-op.
                  }
                        error:&err];
      if (arr == nil) {
        auto msg = std::string("coreml: MLMultiArray init failed: ") +
                   (err ? err.localizedDescription.UTF8String : "unknown");
        return {modelbox::STATUS_FAULT, msg};
      }

      [features setObject:[MLFeatureValue featureValueWithMultiArray:arr]
                   forKey:feature_name];
    }

    NSError *err = nil;
    MLDictionaryFeatureProvider *provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:features
                                                          error:&err];
    if (provider == nil) {
      return {modelbox::STATUS_FAULT,
              std::string("coreml: feature provider init failed: ") +
                  (err ? err.localizedDescription.UTF8String : "unknown")};
    }

    id<MLFeatureProvider> result = [impl_->model predictionFromFeatures:provider
                                                                  error:&err];
    if (result == nil) {
      auto msg = std::string("coreml: predict failed: ") +
                 (err ? err.localizedDescription.UTF8String : "unknown");
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    // Drain outputs into modelbox buffers. Map declared output ports 1:1 onto
    // model output features by name; if the model has a single output and the
    // names differ, route it to the first declared port.
    NSDictionary<NSString *, MLFeatureDescription *> *outputs_desc =
        impl_->model.modelDescription.outputDescriptionsByName;
    for (size_t i = 0; i < io_list_.output_name_list.size(); ++i) {
      const auto &port = io_list_.output_name_list[i];
      NSString *feature_name = NS(port);
      if (outputs_desc[feature_name] == nil) {
        if (impl_->model_output_names.count == 1) {
          feature_name = impl_->model_output_names[0];
        } else if (i < impl_->model_output_names.count) {
          feature_name = impl_->model_output_names[i];
        } else {
          return {modelbox::STATUS_BADCONF,
                  std::string("coreml: output port '") + port +
                      "' not found in model"};
        }
      }

      MLFeatureValue *val = [result featureValueForName:feature_name];
      if (val == nil || val.type != MLFeatureTypeMultiArray) {
        return {modelbox::STATUS_FAULT,
                std::string("coreml: output '") + feature_name.UTF8String +
                    "' missing or not multiarray"};
      }
      MLMultiArray *arr = val.multiArrayValue;
      MLMultiArrayDataType dtype = arr.dataType;
      size_t element_size = MlElementSize(dtype);
      if (element_size == 0) {
        return {modelbox::STATUS_FAULT,
                "coreml: unsupported output dtype"};
      }

      size_t total_elements = (size_t)arr.count;
      size_t total_bytes = total_elements * element_size;

      auto out_buf_list = data_ctx->Output(port);
      if (out_buf_list == nullptr) {
        return {modelbox::STATUS_FAULT,
                "coreml: missing output buffer list for port " + port};
      }
      auto status = out_buf_list->Build({total_bytes});
      if (!status) {
        return status;
      }
      auto out_buf = out_buf_list->At(0);
      auto *dst = out_buf->MutableData();
      if (dst == nullptr) {
        return {modelbox::STATUS_FAULT,
                "coreml: output buffer data is null"};
      }

      // Use getBytesWithHandler: which gives us a contiguous read pointer;
      // fall back to a per-element copy if Core ML can't expose one (e.g.
      // non-contiguous strides).
      __block bool copied = false;
      [arr getBytesWithHandler:^(const void *bytes, NSInteger size) {
        if (bytes != nullptr && (NSInteger)total_bytes <= size) {
          std::memcpy(dst, bytes, total_bytes);
          copied = true;
        }
      }];
      if (!copied) {
        // Fallback: walk via NSNumber accessors. Slow but correct.
        if (dtype == MLMultiArrayDataTypeFloat32) {
          float *dstf = reinterpret_cast<float *>(dst);
          for (size_t k = 0; k < total_elements; ++k) {
            dstf[k] = [arr objectAtIndexedSubscript:k].floatValue;
          }
        } else {
          return {modelbox::STATUS_FAULT,
                  "coreml: contiguous output read failed and dtype fallback "
                  "not implemented"};
        }
      }

      // Stamp shape + type metadata so downstream flowunits (e.g. the yolo
      // postprocessor) can reshape without re-introspecting the model.
      std::vector<size_t> shape_vec;
      for (NSNumber *d in arr.shape) {
        shape_vec.push_back((size_t)d.integerValue);
      }
      out_buf->Set("shape", shape_vec);
      out_buf->Set("type", MlTypeToModelBox(dtype));
    }
  }

  return modelbox::STATUS_OK;
}
