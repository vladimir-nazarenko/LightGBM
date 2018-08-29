#include <LightGBM/utils/openmp_wrapper.h>

#include <LightGBM/utils/common.h>
//#include <LightGBM/utils/random.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/c_api.h>
#include <LightGBM/boosting.h>
#include <LightGBM/config.h>
#include <LightGBM/prediction_early_stop.h>

#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <functional>

#include "./application/predictor.hpp"

namespace LightGBM {

class Booster {
public:
  explicit Booster(const char* filename) {
    boosting_.reset(Boosting::CreateBoosting("gbdt", filename));
  }

  ~Booster() {

  }

  void Predict(int num_iteration, int predict_type, int nrow,
               std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun,
               const Config& config,
               double* out_result, int64_t* out_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool is_predict_leaf = false;
    bool is_raw_score = false;
    bool predict_contrib = false;
    if (predict_type == C_API_PREDICT_LEAF_INDEX) {
      is_predict_leaf = true;
    } else if (predict_type == C_API_PREDICT_RAW_SCORE) {
      is_raw_score = true;
    } else if (predict_type == C_API_PREDICT_CONTRIB) {
      predict_contrib = true;
    } else {
      is_raw_score = false;
    }

    Predictor predictor(boosting_.get(), num_iteration, is_raw_score, is_predict_leaf, predict_contrib,
                        config.pred_early_stop, config.pred_early_stop_freq, config.pred_early_stop_margin);
    int64_t num_pred_in_one_row = boosting_->NumPredictOneRow(num_iteration, is_predict_leaf, predict_contrib);
    auto pred_fun = predictor.GetPredictFunction();
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nrow; ++i) {
      OMP_LOOP_EX_BEGIN();
      auto one_row = get_row_fun(i);
      auto pred_wrt_ptr = out_result + static_cast<size_t>(num_pred_in_one_row) * i;
      pred_fun(one_row, pred_wrt_ptr);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    *out_len = nrow * num_pred_in_one_row;
  }

  void LoadModelFromString(const char* model_str) {
    size_t len = std::strlen(model_str);
    boosting_->LoadModelFromString(model_str, len);
  }

  double GetLeafValue(int tree_idx, int leaf_idx) const {
    return dynamic_cast<GBDTBase*>(boosting_.get())->GetLeafValue(tree_idx, leaf_idx);
  }

  void SetLeafValue(int tree_idx, int leaf_idx, double val) {
    std::lock_guard<std::mutex> lock(mutex_);
    dynamic_cast<GBDTBase*>(boosting_.get())->SetLeafValue(tree_idx, leaf_idx, val);
  }

  const Boosting* GetBoosting() const { return boosting_.get(); }

private:
  std::unique_ptr<Boosting> boosting_;
  /*! \brief All configs */
  Config config_;
  /*! \brief mutex for threading safe call */
  std::mutex mutex_;
};

}

using namespace LightGBM;

// some help functions used to convert data

std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

// start of c_api functions

const char* LGBM_GetLastError() {
  return LastErrorMsg();
}

// ---- start of booster

int LGBM_BoosterLoadModelFromString(
  const char* model_str,
  int* out_num_iterations,
  BoosterHandle* out) {
  API_BEGIN();
  auto ret = std::unique_ptr<Booster>(new Booster(0));
  ret->LoadModelFromString(model_str);
  *out_num_iterations = ret->GetBoosting()->GetCurrentIteration();
  *out = ret.release();
  API_END();
}

#pragma warning(disable : 4702)
int LGBM_BoosterFree(BoosterHandle handle) {
  API_BEGIN();
  delete reinterpret_cast<Booster*>(handle);
  API_END();
}

int LGBM_BoosterGetNumClasses(BoosterHandle handle, int* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetBoosting()->NumberOfClasses();
  API_END();
}

int LGBM_BoosterGetNumFeature(BoosterHandle handle, int* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetBoosting()->MaxFeatureIdx() + 1;
  API_END();
}

int LGBM_BoosterPredictForMat(BoosterHandle handle,
                              const void* data,
                              int data_type,
                              int32_t nrow,
                              int32_t ncol,
                              int is_row_major,
                              int predict_type,
                              int num_iteration,
                              const char* parameter,
                              int64_t* out_len,
                              double* out_result) {
  API_BEGIN();
  auto param = Config::Str2Map(parameter);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto get_row_fun = RowPairFunctionFromDenseMatric(data, nrow, ncol, data_type, is_row_major);
  ref_booster->Predict(num_iteration, predict_type, nrow, get_row_fun,
                       config, out_result, out_len);
  API_END();
}

// ---- start of some help functions

template <typename PTR_T>
class row_functor_row_major {
	const PTR_T *data_ptr_;
	const int num_col_;
	const int num_row_;
public:
	row_functor_row_major(const PTR_T *data_ptr, const int num_col, const int num_row)
			: data_ptr_(data_ptr), num_col_(num_col), num_row_(num_row)
	{}

	std::vector<double> operator() (const int row_idx)
	{
		std::vector<double> ret(num_col_);
		auto tmp_ptr = data_ptr_ + static_cast<size_t>(num_col_) * row_idx;
		for (int i = 0; i < num_col_; ++i) {
			ret[i] = static_cast<double>(*(tmp_ptr + i));
		}
		return ret;
	}
};

template <typename PTR_T>
class row_functor_col_major {
	const PTR_T *data_ptr_;
	const int num_col_;
	const int num_row_;
public:
	row_functor_col_major(const PTR_T *data_ptr, const int num_col, const int num_row)
			: data_ptr_(data_ptr), num_col_(num_col), num_row_(num_row)
	{}

	std::vector<double> operator() (const int row_idx)
	{
		std::vector<double> ret(num_col_);
		for (int i = 0; i < num_col_; ++i) {
			ret[i] = static_cast<double>(*(data_ptr_ + static_cast<size_t>(num_row_) * i + row_idx));
		}
		return ret;
	}
};

std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major) {
  if (data_type == C_API_DTYPE_FLOAT32) {
    const float* data_ptr = reinterpret_cast<const float*>(data);
    if (is_row_major) {
      return row_functor_row_major<float>(data_ptr, num_col, num_row);
    } else {
      return row_functor_col_major<float>(data_ptr, num_col, num_row);
    }
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    const double* data_ptr = reinterpret_cast<const double*>(data);
    if (is_row_major) {
       return row_functor_row_major<double>(data_ptr, num_col, num_row);
    } else {
       return row_functor_col_major<double>(data_ptr, num_col, num_row);
    }
  }
  throw std::runtime_error("Unknown data type in RowFunctionFromDenseMatric");
}

struct _outer_func_ftor {

	_outer_func_ftor(const std::function<std::vector<double>(int row_idx)> &inner_func)
			: inner_func_(inner_func)
	{}

	std::vector<std::pair<int, double>> operator() (const int row_idx)
	{
		auto raw_values = inner_func_(row_idx);
		std::vector<std::pair<int, double>> ret;
		for (int i = 0; i < static_cast<int>(raw_values.size()); ++i) {
			if (std::fabs(raw_values[i]) > kZeroThreshold || std::isnan(raw_values[i])) {
				ret.emplace_back(i, raw_values[i]);
			}
		}
		return ret;
	};

private:
	const std::function<std::vector<double>(int row_idx)> inner_func_;
};

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major) {
  auto inner_function = RowFunctionFromDenseMatric(data, num_row, num_col, data_type, is_row_major);
  if (inner_function != 0) {
    return _outer_func_ftor(inner_function);
  }
  return NULL;
}
