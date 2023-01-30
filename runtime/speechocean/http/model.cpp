#include "model.h"
#include <stdio.h>

class Model {
 public:
  Model();
  ~Model();
  int load(void);
  int predict(void);
};

Model::Model() {}
Model::~Model() {}
int Model::load(void) { return 0; }
int Model::predict(void) { return 0; }

int load(void) {
  printf("Call C.load function.");
  printf("Load model");

  return 0;
}

int predict(void) {
  Model* model = NULL;

  model = new Model();
  model->predict();
  printf("Model inference");

  return 0;
}
