#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char* wav_path;
} ModelRequest;

typedef struct {
  const char* text;
  int duration;
  int decode_time;
} ModelResponse;

int model_load(const char* model_name, const char* model_version,
               const char* model_path);
ModelResponse model_predict(ModelRequest request);

#ifdef __cplusplus
}
#endif
