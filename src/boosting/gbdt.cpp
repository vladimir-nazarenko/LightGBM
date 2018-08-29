#include "gbdt.h"

#include <LightGBM/utils/openmp_wrapper.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>

#include <ctime>

#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <utility>

namespace LightGBM {

#ifdef TIMETAG
std::chrono::duration<double, std::milli> boosting_time;
std::chrono::duration<double, std::milli> train_score_time;
std::chrono::duration<double, std::milli> out_of_bag_score_time;
std::chrono::duration<double, std::milli> valid_score_time;
std::chrono::duration<double, std::milli> metric_time;
std::chrono::duration<double, std::milli> bagging_time;
std::chrono::duration<double, std::milli> tree_time;
#endif // TIMETAG

GBDT::GBDT() : iter_(0),
train_data_(nullptr),
objective_function_(nullptr),
early_stopping_round_(0),
max_feature_idx_(0),
num_tree_per_iteration_(1),
num_class_(1),
num_iteration_for_pred_(0),
shrinkage_rate_(0.1f),
num_init_iteration_(0),
need_re_bagging_(false) {

  #pragma omp parallel
  #pragma omp master
  {
    num_threads_ = omp_get_num_threads();
  }
  average_output_ = false;
}

GBDT::~GBDT() {
  #ifdef TIMETAG
  Log::Info("GBDT::boosting costs %f", boosting_time * 1e-3);
  Log::Info("GBDT::train_score costs %f", train_score_time * 1e-3);
  Log::Info("GBDT::out_of_bag_score costs %f", out_of_bag_score_time * 1e-3);
  Log::Info("GBDT::valid_score costs %f", valid_score_time * 1e-3);
  Log::Info("GBDT::metric costs %f", metric_time * 1e-3);
  Log::Info("GBDT::bagging costs %f", bagging_time * 1e-3);
  Log::Info("GBDT::tree costs %f", tree_time * 1e-3);
  #endif
}


/* If the custom "average" is implemented it will be used inplace of the label average (if enabled)
*
* An improvement to this is to have options to explicitly choose
* (i) standard average
* (ii) custom average if available
* (iii) any user defined scalar bias (e.g. using a new option "init_score" that overrides (i) and (ii) )
*
* (i) and (ii) could be selected as say "auto_init_score" = 0 or 1 etc..
*
*/
double ObtainAutomaticInitialScore(const ObjectiveFunction* fobj) {
  double init_score = 0.0f;
  if (fobj != nullptr) {
    init_score = fobj->BoostFromScore();
  }
  return init_score;
}


void GBDT::ResetConfig(const Config* config) {
  auto new_config = std::unique_ptr<Config>(new Config(*config));
  early_stopping_round_ = new_config->early_stopping_round;
  shrinkage_rate_ = new_config->learning_rate;
  config_.reset(new_config.release());
}

}  // namespace LightGBM
