#ifdef __cplusplus
extern "C" {
#endif

struct DecoderResult {
  std::string transcript;
  int duration;
  int decode_time;
};

int init();
int load();
DecoderResult predict(std::string wav_path);

#ifdef __cplusplus
}
#endif
