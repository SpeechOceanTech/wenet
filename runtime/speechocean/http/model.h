#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char* transcript;
  int duration;
  int decode_time;
} CDecodeResult;

int init();
int load();
CDecodeResult predict(char* wav_path);

#ifdef __cplusplus
}
#endif
