#include <iostream>
#include <vector>
#include "../include/LightGBM/c_api.h"
#include "booster_str.h"

int main(int argc, char** argv) {
  std::vector<double> data(180, 0);

  int num_iters = -1;
  BoosterHandle handle;
  LGBM_BoosterLoadModelFromString(model_txt.c_str(),  &num_iters, &handle);

  int num_classes = -1;
  LGBM_BoosterGetNumClasses(handle, &num_classes);
  std::cout << "NUMCLASSES: " << num_classes << std::endl;

  std::vector<double> res(1 * num_classes, 0);
  int64_t res_len = -1;
  LGBM_BoosterPredictForMat(handle, data.data(), C_API_DTYPE_FLOAT64, 1, 180, 1, C_API_PREDICT_NORMAL, num_iters, "", &res_len, res.data());

  std::cout << "RESLEN: " << res_len << std::endl;

}
