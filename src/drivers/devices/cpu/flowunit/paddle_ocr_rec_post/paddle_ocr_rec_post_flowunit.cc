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

#include "paddle_ocr_rec_post_flowunit.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

modelbox::Status PaddleOcrRecPostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto dict_path = opts->GetString("char_dict_file", "");
  if (dict_path.empty()) {
    return {modelbox::STATUS_BADCONF,
            "paddle_ocr_rec_post: required option 'char_dict_file' missing"};
  }
  std::ifstream ifs(dict_path);
  if (!ifs.is_open()) {
    return {modelbox::STATUS_BADCONF,
            "paddle_ocr_rec_post: cannot open char_dict_file: " + dict_path};
  }
  charset_.clear();
  charset_.emplace_back("");  // CTC blank at index 0
  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    charset_.push_back(line);
  }
  ifs.close();

  bool use_space = opts->GetBool("use_space_char", true);
  if (use_space) {
    charset_.emplace_back(" ");
  }

  score_thresh_ = opts->GetFloat("score_thresh", 0.5F);
  MBLOG_INFO << "paddle_ocr_rec_post: charset_size=" << charset_.size()
             << " (blank+chars" << (use_space ? "+space" : "") << ")"
             << " score_thresh=" << score_thresh_;
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrRecPostFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrRecPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_logits = ctx->Input("in_logits");
  auto in_crop = ctx->Input("in_crop");
  auto out_result = ctx->Output("out_result");
  if (!in_logits || !in_crop || !out_result) {
    return {modelbox::STATUS_FAULT, "paddle_ocr_rec_post: missing port"};
  }

  const size_t n_crop = in_crop->Size();
  const size_t n_logit = in_logits->Size();
  if (n_crop == 0) {
    return modelbox::STATUS_OK;
  }

  // Resolve T, C and per-crop logits stride. paddle_inference_flowunit packs
  // the whole batch into one output buffer (it only feeds ins[0] to Infer and
  // emits one output buffer per port). So the expected scheme is
  // n_logit == 1 with shape {N, T, C}. Tolerate the alternative of
  // n_logit == n_crop with shape {1, T, C} per buffer for defensiveness.
  const int kClasses = static_cast<int>(charset_.size());
  size_t T = 0;
  size_t C = 0;
  bool single_buffer = false;
  const float *batch_ptr = nullptr;

  if (n_logit == 1) {
    auto lbuf = in_logits->At(0);
    std::vector<size_t> shape;
    if (!lbuf->Get("shape", shape) || shape.size() < 2) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_rec_post: logits buffer missing shape meta"};
    }
    // Accept {N,T,C} or {T,C} (latter only when n_crop == 1).
    if (shape.size() == 3) {
      if (shape[0] != n_crop) {
        return {modelbox::STATUS_FAULT,
                "paddle_ocr_rec_post: logits N=" + std::to_string(shape[0]) +
                    " != in_crop N=" + std::to_string(n_crop)};
      }
      T = shape[1];
      C = shape[2];
    } else if (shape.size() == 2 && n_crop == 1) {
      T = shape[0];
      C = shape[1];
    } else {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_rec_post: unsupported logits shape (rank " +
                  std::to_string(shape.size()) + ")"};
    }
    const size_t need = n_crop * T * C * sizeof(float);
    if (lbuf->GetBytes() != need) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_rec_post: logits bytes " +
                  std::to_string(lbuf->GetBytes()) + " != " +
                  std::to_string(need)};
    }
    batch_ptr = static_cast<const float *>(lbuf->ConstData());
    single_buffer = true;
  } else if (n_logit == n_crop) {
    auto lbuf0 = in_logits->At(0);
    std::vector<size_t> shape;
    if (!lbuf0->Get("shape", shape)) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_rec_post: per-crop logits missing shape meta"};
    }
    if (shape.size() == 3 && shape[0] == 1) {
      T = shape[1];
      C = shape[2];
    } else if (shape.size() == 2) {
      T = shape[0];
      C = shape[1];
    } else {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_rec_post: unsupported per-crop logits shape"};
    }
    single_buffer = false;
  } else {
    return {modelbox::STATUS_FAULT,
            "paddle_ocr_rec_post: in_logits size " + std::to_string(n_logit) +
                " incompatible with in_crop size " + std::to_string(n_crop)};
  }

  if (T == 0 || C == 0) {
    return {modelbox::STATUS_FAULT, "paddle_ocr_rec_post: zero T or C"};
  }
  if (static_cast<int>(C) != kClasses) {
    MBLOG_WARN << "paddle_ocr_rec_post: logits C=" << C
               << " != charset size=" << kClasses
               << " (decoded indices may be off)";
  }

  // Dummy placeholder payload: 1 byte per result; metadata carries the answer.
  std::vector<size_t> out_sizes(n_crop, 1);
  auto bs = out_result->Build(out_sizes);
  if (!bs) return bs;

  const size_t stride = T * C;

  for (size_t i = 0; i < n_crop; ++i) {
    const float *p = nullptr;
    if (single_buffer) {
      p = batch_ptr + i * stride;
    } else {
      auto lbuf = in_logits->At(i);
      if (lbuf->GetBytes() != stride * sizeof(float)) {
        return {modelbox::STATUS_FAULT,
                "paddle_ocr_rec_post: per-crop logits buffer size mismatch"};
      }
      p = static_cast<const float *>(lbuf->ConstData());
    }

    std::string text;
    double score_sum = 0.0;
    int score_n = 0;
    int prev_idx = -1;
    for (size_t t = 0; t < T; ++t) {
      const float *row = p + t * C;
      int best = 0;
      float best_v = row[0];
      for (size_t c = 1; c < C; ++c) {
        if (row[c] > best_v) {
          best_v = row[c];
          best = static_cast<int>(c);
        }
      }
      // Softmax probability of the argmax: 1 / sum(exp(row[c] - best_v)).
      double sum_exp = 0.0;
      for (size_t c = 0; c < C; ++c) {
        sum_exp += std::exp(static_cast<double>(row[c]) - best_v);
      }
      const double prob = sum_exp > 0.0 ? 1.0 / sum_exp : 0.0;

      // CTC collapse: emit only non-blank (idx != 0) and non-repeat
      // (idx != prev_idx). Accumulate score for emitted timesteps.
      if (best != 0 && best != prev_idx) {
        if (best >= 0 && best < static_cast<int>(charset_.size())) {
          text += charset_[best];
        }
        score_sum += prob;
        ++score_n;
      }
      prev_idx = best;
    }

    float score = score_n > 0 ? static_cast<float>(score_sum / score_n) : 0.0F;
    if (score < score_thresh_) {
      text.clear();
    }

    auto src_buf = in_crop->At(i);
    auto dst_buf = out_result->At(i);
    auto *raw = static_cast<uint8_t *>(dst_buf->MutableData());
    raw[0] = 0;  // placeholder byte

    dst_buf->Set("text", text);
    dst_buf->Set("score", score);

    int64_t frame_id = 0;
    int32_t box_id = 0;
    int32_t box_count = 0;
    int32_t angle = 0;
    std::vector<float> polygon;
    if (src_buf->Get("frame_id", frame_id)) {
      dst_buf->Set("frame_id", frame_id);
    }
    if (src_buf->Get("box_id", box_id)) {
      dst_buf->Set("box_id", box_id);
    }
    if (src_buf->Get("box_count", box_count)) {
      dst_buf->Set("box_count", box_count);
    }
    if (src_buf->Get("polygon", polygon)) {
      dst_buf->Set("polygon", polygon);
    }
    if (src_buf->Get("angle", angle)) {
      dst_buf->Set("angle", angle);
    }
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(PaddleOcrRecPostFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_logits"});
  desc.AddFlowUnitInput({"in_crop"});
  desc.AddFlowUnitOutput({"out_result"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "char_dict_file", "string", true, "",
      "path to PaddleOCR charset dictionary file (one char per line)"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "score_thresh", "float", false, "0.5",
      "minimum mean per-char softmax probability to keep decoded text"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "use_space_char", "bool", false, "true",
      "append space character to the charset"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
