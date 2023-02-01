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

typedef struct {
  std::string transcript;
  int duration;
  int decode_time;
} DecodeResult;

class Model {
 public:
  Model(std::string model_name, std::string model_verion,
        std::string model_path);
  ~Model();
  int load();
  DecodeResult predict(std::string wav_path);

 private:
  DecodeResult decode(std::pair<std::string, std::string> wav);
  int decode_task(std::pair<std::string, std::string> wav,
                  DecodeResult* deocder_result);

  std::string model_name;
  std::string model_version;
  std::string model_path;

  std::shared_ptr<wenet::DecodeOptions> decode_options = nullptr;
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config = nullptr;
  std::shared_ptr<wenet::DecodeResource> decode_resource = nullptr;
};

// Wenet核心解码函数
DecodeResult Model::decode(std::pair<std::string, std::string> wav) {
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

  DecodeResult result = {
    transcript : final_result,
    duration : wave_dur,
    decode_time : decode_time,
  };

  return result;
}

int Model::decode_task(std::pair<std::string, std::string> wav,
                       DecodeResult* final_result) {
  DecodeResult result;

  result = decode(wav);
  final_result->transcript = result.transcript;
  final_result->duration = result.duration;
  final_result->decode_time = result.decode_time;

  return 0;
}

Model::Model(std::string model_name, std::string model_version,
             std::string model_path)
    : model_name(model_name),
      model_version(model_version),
      model_path(model_path) {
  std::vector<std::string> cmd_argv;
  cmd_argv.push_back(std::string("speechocean"));
  cmd_argv.push_back(std::string("--model_path"));
  cmd_argv.push_back(model_path + std::string("final.zip"));
  cmd_argv.push_back(std::string("--unit_path"));
  cmd_argv.push_back(model_path + std::string("/words.txt"));
  cmd_argv.push_back(std::string("--chunk_size"));
  cmd_argv.push_back(std::string("-1"));
  cmd_argv.push_back(std::string("--op_thread_num"));
  cmd_argv.push_back(std::string("8"));

  int argc = static_cast<int>(cmd_argv.size());
  char** argv = new char*[argc + 1];

  for (int i = 0; i < argc; i++) {
    int length = static_cast<int>(cmd_argv[i].size());
    argv[i] = new char[length + 1];
    strncpy(argv[i], cmd_argv[i].c_str(), length);
    argv[i][length] = '\0';
  }

  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging("speechocean");

  decode_options = wenet::InitDecodeOptionsFromFlags();
  feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  decode_resource = wenet::InitDecodeResourceFromFlags();
}

Model::~Model() {}

int Model::load() { return 0; }

DecodeResult Model::predict(std::string wav_path) {
  std::thread thread;
  DecodeResult decoder_result;
  std::pair<std::string, std::string> wav = make_pair(wav_path, wav_path);

  // 线程函数通过传入decoder_result指针得到解码结果；
  // 传入Model::decode_task函数指针，修复invalid use of non-static member
  // function报错
  thread = std::thread(&Model::decode_task, this, wav, &decoder_result);
  thread.join();

  return decoder_result;
}

std::shared_ptr<Model> g_model;

int model_load(const char* model_name, const char* model_version,
               const char* model_path) {
  g_model = std::make_shared<Model>(std::string(model_name),
                                    std::string(model_version),
                                    std::string(model_path));
  return 0;
}

CDecodeResult model_predict(const char* wav_path) {
  DecodeResult ori_result;
  CDecodeResult result;

  std::string c_wav_path(wav_path);
  ori_result = g_model->predict(c_wav_path);

  result.transcript = ori_result.transcript.c_str();
  result.duration = ori_result.duration;
  result.decode_time = ori_result.decode_time;

  return result;
}

int main(int argc, char* argv[]) {
  const char* model_name = "wenet";
  const char* model_version = "v1";
  const char* model_path = "/data/yangyang/models/wenet/zh-cn/";

  // 加载模型
  model_load(model_name, model_version, model_path);

  const char* wav_path = "zh-cn-demo.wav";
  // warmup
  // 根据实验结果，前两次解码时间较长，且不稳定，后续解码时间比较稳定。
  for (int i = 0; i < 3; i++) {
    model_predict(wav_path);
  }

  int count = 3;
  CDecodeResult result;
  for (int i = 0; i < count; i++) {
    result = model_predict(wav_path);
    LOG(INFO) << "Decode wav " << wav_path << " duration: " << result.duration
              << " transcript: " << result.transcript << ", elpase "
              << result.decode_time << "ms";
  }

  return 0;
}