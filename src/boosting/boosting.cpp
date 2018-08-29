#include <LightGBM/boosting.h>
#include "gbdt.h"

namespace LightGBM {

Boosting* Boosting::CreateBoosting(const std::string& type, const char* filename) {
  if (filename == nullptr || filename[0] == '\0') {
    if (type == std::string("gbdt")) {
      return new GBDT();
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

}  // namespace LightGBM
