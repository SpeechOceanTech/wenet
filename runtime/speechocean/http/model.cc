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

std::shared_ptr<wenet::DecodeOptions> g_decode_config;
std::shared_ptr<wenet::FeaturePipelineConfig> g_feature_config;
std::shared_ptr<wenet::DecodeResource> g_decode_resource;

std::ofstream g_result;
std::mutex g_mutex;
int g_total_waves_dur = 0;
int g_total_decode_time = 0;

struct DecoderResult {
  std::string result;
  int duration;
  int decode_time;
};

DecoderResult decode(std::pair<std::string, std::string> wav) {
  wenet::WavReader wav_reader(wav.second);
  int num_samples = wav_reader.num_samples();
  CHECK_EQ(wav_reader.sample_rate(), FLAGS_sample_rate);

  auto feature_pipeline =
      std::make_shared<wenet::FeaturePipeline>(*g_feature_config);
  feature_pipeline->AcceptWaveform(wav_reader.data(), num_samples);
  feature_pipeline->set_input_finished();
  LOG(INFO) << "num frames " << feature_pipeline->num_frames();

  wenet::AsrDecoder decoder(feature_pipeline, g_decode_resource,
                            *g_decode_config);

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
      LOG(INFO) << "Partial result: " << decoder.result()[0].sentence;
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
          static_cast<float>(g_feature_config->frame_shift) /
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

  LOG(INFO) << "Decode wav " << wav.first << " duration " << wave_dur
            << "ms audio token, elapse " << decode_time << "ms.";

  DecoderResult result = {
    result : final_result,
    duration : wave_dur,
    decode_time : decode_time,
  };

  return result;
}

class Model {
 public:
  Model();
  ~Model();
  int load();
  int predict();

 private:
  void* config;
};

Model::Model() {}
Model::~Model() {}
int Model::load() { return 0; }
int Model::predict() { return 0; }

std::shared_ptr<Model> g_model;

int init() { return 0; }

int load() { return 0; }

int predict() { return 0; }

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  g_decode_config = wenet::InitDecodeOptionsFromFlags();
  g_feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  g_decode_resource = wenet::InitDecodeResourceFromFlags();

  if (FLAGS_wav_path.empty()) {
    LOG(FATAL) << "Please provide the wave path or the wav scp!";
  }

  std::string result;
  std::pair<std::string, std::string> wav =
      make_pair(FLAGS_wav_path, FLAGS_wav_path);

  // warmup
  // 根据实验结果，前两次解码时间较长，且不稳定，后续解码时间比较稳定。
  for (int i = 0; i < 3; i++) {
    decode(wav);
  }

  DecoderResult decode_result;
  int total_decode_time = 0;

  for (int i = 0; i < 10; i++) {
    decode_result = decode(wav);
    total_decode_time += decode_result.decode_time;
  }

  LOG(INFO) << "Decode average time " << total_decode_time / 10 << "ms";
  LOG(INFO) << "Decode wav " << wav.first << "result " << decode_result.result
            << std::endl;
}