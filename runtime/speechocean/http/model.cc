#include <stdio.h>
#include <boost/shared_ptr.hpp>
#include <iostream>

#include "decoder/params.h"
#include "frontend/wav.h"
#include "gflags/gflags.h"
#include "model.h"
#include "utils/flags.h"
#include "utils/string.h"
#include "utils/thread_pool.h"
#include "utils/timer.h"
#include "utils/utils.h"

DEFINE_bool(simulate_streaming, false, "simulate streaming input");
DEFINE_bool(output_nbest, false, "output n-best of decode result");
DEFINE_string(wav_path, "", "single wave path");
DEFINE_string(result, "", "result output file");
DEFINE_bool(continuous_decoding, false, "continuous decoding mode");
DEFINE_int32(thread_num, 1, "num of decode thread");
DEFINE_int32(warmup, 0, "num of warmup decode, 0 means no warmup");

class Model {
 public:
  Model();
  ~Model();
  int load();
  DecoderResult predict(std::string wav_path);

 private:
  DecoderResult decode(std::pair<std::string, std::string> wav);
  int decode_task(std::pair<std::string, std::string> wav,
                  DecoderResult* deocder_result);

  std::shared_ptr<wenet::DecodeOptions> decode_options = nullptr;
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config = nullptr;
  std::shared_ptr<wenet::DecodeResource> decode_resource = nullptr;
};

// Wenet核心解码函数
DecoderResult Model::decode(std::pair<std::string, std::string> wav) {
  wenet::WavReader wav_reader(wav.second);
  int num_samples = wav_reader.num_samples();
  CHECK_EQ(wav_reader.sample_rate(), FLAGS_sample_rate);

  auto feature_pipeline =
      std::make_shared<wenet::FeaturePipeline>(*feature_config);
  feature_pipeline->AcceptWaveform(wav_reader.data(), num_samples);
  feature_pipeline->set_input_finished();
  LOG(INFO) << "num frames " << feature_pipeline->num_frames();

  wenet::AsrDecoder decoder(feature_pipeline, decode_resource, *decode_options);

  int wave_dur = static_cast<int>(static_cast<float>(num_samples) /
                                  wav_reader.sample_rate() * 1000);

  int decode_time = 0;
  std::string final_result;
  while (true) {
    wenet::Timer timer;
    wenet::DecodeState state = decoder.Decode();
    if (state == wenet::DecodeState::kEndFeats) {
      decoder.Rescoring();
    }
    int chunk_decode_time = timer.Elapsed();
    decode_time += chunk_decode_time;
    if (decoder.DecodedSomething()) {
      //   LOG(INFO) << "Partial result: " << decoder.result()[0].sentence;
    }

    if (FLAGS_continuous_decoding && state == wenet::DecodeState::kEndpoint) {
      if (decoder.DecodedSomething()) {
        decoder.Rescoring();
        LOG(INFO) << "Final result (continuous decoding): "
                  << decoder.result()[0].sentence;
        final_result.append(decoder.result()[0].sentence);
      }
      decoder.ResetContinuousDecoding();
    }

    if (state == wenet::DecodeState::kEndFeats) {
      break;
    } else if (FLAGS_chunk_size > 0 && FLAGS_simulate_streaming) {
      float frame_shift_in_ms =
          static_cast<float>(feature_config->frame_shift) /
          wav_reader.sample_rate() * 1000;
      auto wait_time =
          decoder.num_frames_in_current_chunk() * frame_shift_in_ms -
          chunk_decode_time;
      if (wait_time > 0) {
        LOG(INFO) << "Simulate streaming, waiting for " << wait_time << "ms";
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(wait_time)));
      }
    }
  }
  if (decoder.DecodedSomething()) {
    final_result.append(decoder.result()[0].sentence);
  }

  //   LOG(INFO) << "Decode wav " << wav.first << " duration " << wave_dur
  //             << "ms audio token, elapse " << decode_time << "ms.";

  DecoderResult result = {
    transcript : final_result,
    duration : wave_dur,
    decode_time : decode_time,
  };

  return result;
}

int Model::decode_task(std::pair<std::string, std::string> wav,
                       DecoderResult* final_result) {
  DecoderResult result;

  result = decode(wav);
  final_result->transcript = result.transcript;
  final_result->duration = result.duration;
  final_result->decode_time = result.decode_time;

  return 0;
}

Model::Model() {
  char* params[] = {"speechocean",
                    "--model_path",
                    "/data/yangyang/models/wenet/zh-cn/final.zip",
                    "--unit_path",
                    "/data/yangyang/models/wenet/zh-cn/words.txt",
                    "--chunk_size",
                    "-1",
                    "--op_thread_num",
                    "12"};
  char** argv = params;
  int argc = 9;

  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging("speechocean");

  decode_options = wenet::InitDecodeOptionsFromFlags();
  feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  decode_resource = wenet::InitDecodeResourceFromFlags();
}

Model::~Model() {}

int Model::load() { return 0; }

DecoderResult Model::predict(std::string wav_path) {
  std::thread thread;
  DecoderResult decoder_result;
  std::pair<std::string, std::string> wav = make_pair(wav_path, wav_path);

  thread = std::thread(&Model::decode_task, this, wav, &decoder_result);
  thread.join();

  return decoder_result;
}

std::shared_ptr<Model> g_model;

int init() {
  g_model = std::make_shared<Model>();
  return 0;
}

int load() { return 0; }

DecoderResult predict(std::string wav_path) {
  DecoderResult result;
  result = g_model->predict(wav_path);
  return result;
}

int main(int argc, char* argv[]) {
  // 初始化
  init();

  // 加载模型
  load();

  std::string wav_path = "zh-cn-demo.wav";
  // warmup
  // 根据实验结果，前两次解码时间较长，且不稳定，后续解码时间比较稳定。
  for (int i = 0; i < 3; i++) {
    predict(wav_path);
  }

  int count = 3;
  DecoderResult result;
  for (int i = 0; i < count; i++) {
    result = predict(wav_path);
    LOG(INFO) << "Decode wav " << wav_path << " duration: " << result.duration
              << " transcript: " << result.transcript << ", elpase "
              << result.decode_time << "ms";
  }

  return 0;
}