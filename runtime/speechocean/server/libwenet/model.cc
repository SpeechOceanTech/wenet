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
  std::string sentence;
  int duration;
  int decode_time;

  std::vector<wenet::DecodeResult> decode_results;
} FinalDecodeResult;

class Model {
 public:
  Model(std::string model_name, std::string model_verion,
        std::string model_path);
  ~Model();
  int load();
  FinalDecodeResult predict(std::string wav_path);

 private:
  FinalDecodeResult decode(std::pair<std::string, std::string> wav);
  int decode_task(std::pair<std::string, std::string> wav,
                  FinalDecodeResult* deocder_result);

  std::string model_name;
  std::string model_version;
  std::string model_path;

  std::shared_ptr<wenet::DecodeOptions> decode_options = nullptr;
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config = nullptr;
  std::shared_ptr<wenet::DecodeResource> decode_resource = nullptr;
};

// Wenet核心解码函数
FinalDecodeResult Model::decode(std::pair<std::string, std::string> wav) {
  FinalDecodeResult final_decode_result;
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
  //   std::string final_result;
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

        for (int i = 0; i < decoder.result().size(); i++) {
          //   final_decode_result.decode_results.emplace_back(decoder.result()[i]);
        }
        // final_result.append(decoder.result()[0].sentence);
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
    for (int i = 0; i < decoder.result().size(); i++) {
      final_decode_result.decode_results.emplace_back(decoder.result()[i]);
    }

    // final_result.append(decoder.result()[0].sentence);
  }

  //   LOG(INFO) << "Decode wav " << wav.first << " duration " << wave_dur
  //             << "ms audio token, elapse " << decode_time << "ms.";

  final_decode_result.sentence = final_decode_result.decode_results[0].sentence;

  final_decode_result.duration = wave_dur;
  final_decode_result.decode_time = decode_time;
  return final_decode_result;
}

int Model::decode_task(std::pair<std::string, std::string> wav,
                       FinalDecodeResult* final_result) {
  FinalDecodeResult result;

  result = decode(wav);

  for (int i = 0; i < result.decode_results.size(); i++) {
    final_result->decode_results.emplace_back(result.decode_results[i]);
  }
  final_result->sentence = result.sentence;
  final_result->duration = result.duration;
  final_result->decode_time = result.decode_time;

  LOG(INFO) << "Decode sentence: " << final_result->sentence
            << ", duration: " << final_result->duration
            << "ms, elpase: " << final_result->decode_time << "ms";

  return 0;
}

Model::Model(std::string model_name, std::string model_version,
             std::string model_path)
    : model_name(model_name),
      model_version(model_version),
      model_path(model_path) {
#ifdef WENET_LIB
  std::vector<std::string> cmd_argv;
  cmd_argv.push_back(std::string("speechocean"));
  cmd_argv.push_back(std::string("--model_path"));
  cmd_argv.push_back(model_path + std::string("final.zip"));
  cmd_argv.push_back(std::string("--unit_path"));
  cmd_argv.push_back(model_path + std::string("words.txt"));
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
  google::InitGoogleLogging(argv[0]);
#endif

  load();
}

Model::~Model() {}

int Model::load() {
  decode_options = wenet::InitDecodeOptionsFromFlags();
  feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  decode_resource = wenet::InitDecodeResourceFromFlags();

  return 0;
}

FinalDecodeResult Model::predict(std::string wav_path) {
  std::thread thread;
  FinalDecodeResult final_result;
  std::pair<std::string, std::string> wav = make_pair(wav_path, wav_path);

  // 线程函数通过传入decoder_result指针得到解码结果；
  // 传入Model::decode_task函数指针，修复invalid use of non-static member
  // function报错
  thread = std::thread(&Model::decode_task, this, wav, &final_result);
  thread.join();

  return final_result;
}

std::shared_ptr<Model> g_model;

int model_load(const char* model_name, const char* model_version,
               const char* model_path) {
  g_model = std::make_shared<Model>(std::string(model_name),
                                    std::string(model_version),
                                    std::string(model_path));
  return 0;
}

ModelResponse model_predict(ModelRequest request) {
  FinalDecodeResult result;
  ModelResponse response;

  std::string wav_path(request.wav_path);
  result = g_model->predict(wav_path);

  response.text = result.sentence.c_str();
  response.duration = result.duration;
  response.decode_time = result.decode_time;

  return response;
}

int main(int argc, char* argv[]) {
  const char* model_name = "wenet";
  const char* model_version = "v1";

  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_wav_path.empty()) {
    LOG(FATAL) << "Please provide the wav path";
  }

  std::string model_path = FLAGS_wav_path;

  // 加载模型
  model_load(model_name, model_version, model_path.c_str());

  ModelRequest request;
  request.wav_path = "zh-cn-demo.wav";
  // warmup
  // 根据实验结果，前两次解码时间较长，且不稳定，后续解码时间比较稳定。
  for (int i = 0; i < 3; i++) {
    model_predict(request);
  }

  int count = 3;
  ModelResponse response;
  for (int i = 0; i < count; i++) {
    response = model_predict(request);
    LOG(INFO) << "Decode wav " << request.wav_path
              << " duration: " << response.duration
              << " transcript: " << response.text << ", elpase "
              << response.decode_time << "ms";
  }

  return 0;
}