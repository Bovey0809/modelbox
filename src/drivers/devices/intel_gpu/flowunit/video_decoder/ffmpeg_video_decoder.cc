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

#include "ffmpeg_video_decoder.h"

#include <video_decode_common.h>

#include "modelbox/base/log.h"

namespace {

// Map a codec id to the FFmpeg-built-in QSV decoder name. Returns nullptr if
// no QSV variant exists (we fall back to the software decoder in that case).
const char *QsvDecoderName(AVCodecID codec_id) {
  switch (codec_id) {
    case AV_CODEC_ID_H264:       return "h264_qsv";
    case AV_CODEC_ID_HEVC:       return "hevc_qsv";
    case AV_CODEC_ID_MPEG2VIDEO: return "mpeg2_qsv";
    case AV_CODEC_ID_VP8:        return "vp8_qsv";
    case AV_CODEC_ID_VP9:        return "vp9_qsv";
    case AV_CODEC_ID_AV1:        return "av1_qsv";
    case AV_CODEC_ID_VC1:        return "vc1_qsv";
    case AV_CODEC_ID_MJPEG:      return "mjpeg_qsv";
    default:                     return nullptr;
  }
}

}  // namespace

modelbox::Status FfmpegVideoDecoder::Init(AVCodecID codec_id) {
  codec_id_ = codec_id;

  // Prefer the Intel Quick Sync Video hardware decoder; fall back to the
  // generic software decoder if QSV is not available for this codec.
  const AVCodec *codec_ptr = nullptr;
  if (const char *qsv_name = QsvDecoderName(codec_id_)) {
    codec_ptr = avcodec_find_decoder_by_name(qsv_name);
    if (codec_ptr != nullptr) {
      MBLOG_INFO << "intel_gpu video_decoder: using QSV codec '" << qsv_name
                 << "'";
    }
  }
  if (codec_ptr == nullptr) {
    codec_ptr = avcodec_find_decoder(codec_id_);
    if (codec_ptr != nullptr) {
      MBLOG_WARN << "intel_gpu video_decoder: no QSV codec for codec_id "
                 << codec_id_ << ", falling back to software '"
                 << codec_ptr->name << "'";
    }
  }
  if (codec_ptr == nullptr) {
    MBLOG_ERROR << "Find decoder for codec[" << codec_id_ << "] failed";
    return modelbox::STATUS_FAULT;
  }

  auto *av_ctx_ptr = avcodec_alloc_context3(codec_ptr);
  if (av_ctx_ptr == nullptr) {
    MBLOG_ERROR << "avcodec_alloc_context3 return, codec_id " << codec_id_;
    return modelbox::STATUS_FAULT;
  }

  AVDictionary *opts = nullptr;
  av_dict_set(&opts, "refcounted_frames", "1", 0);
  auto ret = avcodec_open2(av_ctx_ptr, codec_ptr, &opts);
  av_dict_free(&opts);
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "avcodec_open2 failed, code_id " << codec_id_ << ", err "
                << err_str;
    avcodec_free_context(&av_ctx_ptr);
    return modelbox::STATUS_FAULT;
  }

  av_ctx_.reset(av_ctx_ptr,
                [](AVCodecContext *ctx) { avcodec_free_context(&ctx); });

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoDecoder::Decode(
    const std::shared_ptr<const AVPacket> &av_packet,
    std::list<std::shared_ptr<AVFrame>> &av_frame_list) {
  auto ret = avcodec_send_packet(av_ctx_.get(), av_packet.get());
  if (ret == AVERROR_EOF) {
    return modelbox::STATUS_NODATA;
  }

  if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "avcodec_send_packet failed, err " << err_str;
    return modelbox::STATUS_FAULT;
  }

  do {
    auto *av_frame_ptr = av_frame_alloc();
    if (av_frame_ptr == nullptr) {
      MBLOG_ERROR << "av frame alloc failed";
      return modelbox::STATUS_FAULT;
    }

    std::shared_ptr<AVFrame> av_frame(
        av_frame_ptr, [](AVFrame *frame) { av_frame_free(&frame); });
    ret = avcodec_receive_frame(av_ctx_.get(), av_frame.get());
    if (ret == AVERROR(EAGAIN)) {
      return modelbox::STATUS_SUCCESS;
    }

    if (ret == AVERROR_EOF) {
      return modelbox::STATUS_NODATA;
    }

    if (ret < 0) {
      GET_FFMPEG_ERR(ret, err_str);
      MBLOG_ERROR << "avcodec_receive_frame failed, err " << err_str;
      return modelbox::STATUS_FAULT;
    }

    av_frame_list.push_back(av_frame);
  } while (ret >= 0);

  return modelbox::STATUS_SUCCESS;
}