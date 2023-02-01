#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char* transcript;
  int duration;
  int decode_time;
} CDecodeResult;

int model_load(const char* model_name, const char* model_version,
               const char* model_path);
CDecodeResult model_predict(const char* wav_path);

#ifdef __cplusplus
}
#endif
