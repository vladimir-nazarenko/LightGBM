#ifndef LIGHTGBM_PREDICTOR_HPP_
#define LIGHTGBM_PREDICTOR_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/boosting.h>
//#include <LightGBM/dataset.h>

#include <LightGBM/utils/openmp_wrapper.h>

#include <map>
#include <cstring>
#include <cstdio>
#include <vector>
#include <utility>
#include <functional>
#include <string>
#include <memory>

namespace LightGBM {
	class Predictor;
}

namespace {
	namespace {
		class predict_ftor {
			LightGBM::Predictor *predictor_;
			const int kFeatureThreshold_;
			const size_t KSparseThreshold_;
		public:
			predict_ftor(LightGBM::Predictor *predictor, const int kFeatureThreshold, const size_t KSparseThreshold);

			void operator() (const std::vector<std::pair<int, double>>& features, double* output);
		};
	}
}

namespace LightGBM {

/*!
* \brief Used to predict data with input model
*/
class Predictor {
	friend class ::predict_ftor;
public:
  /*!
  * \brief Constructor
  * \param boosting Input boosting model
  * \param num_iteration Number of boosting round
  * \param is_raw_score True if need to predict result with raw score
  * \param predict_leaf_index True to output leaf index instead of prediction score
  * \param predict_contrib True to output feature contributions instead of prediction score
  */
  Predictor(Boosting* boosting, int num_iteration,
            bool is_raw_score, bool predict_leaf_index, bool predict_contrib,
            bool early_stop, int early_stop_freq, double early_stop_margin) {

    early_stop_ = CreatePredictionEarlyStopInstance("none", LightGBM::PredictionEarlyStopConfig());
    if (early_stop && !boosting->NeedAccuratePrediction()) {
      PredictionEarlyStopConfig pred_early_stop_config;
      CHECK(early_stop_freq > 0);
      CHECK(early_stop_margin >= 0);
      pred_early_stop_config.margin_threshold = early_stop_margin;
      pred_early_stop_config.round_period = early_stop_freq;
      if (boosting->NumberOfClasses() == 1) {
        early_stop_ = CreatePredictionEarlyStopInstance("binary", pred_early_stop_config);
      } else {
        early_stop_ = CreatePredictionEarlyStopInstance("multiclass", pred_early_stop_config);
      }
    }

    #pragma omp parallel
    #pragma omp master
    {
      num_threads_ = omp_get_num_threads();
    }
    boosting->InitPredict(num_iteration, predict_contrib);
    boosting_ = boosting;
    num_pred_one_row_ = boosting_->NumPredictOneRow(num_iteration, predict_leaf_index, predict_contrib);
    num_feature_ = boosting_->MaxFeatureIdx() + 1;
    predict_buf_ = std::vector<std::vector<double>>(num_threads_, std::vector<double>(num_feature_, 0.0f));
    const int kFeatureThreshold = 100000;
    const size_t KSparseThreshold = static_cast<size_t>(0.01 * num_feature_);
    if (predict_leaf_index || is_raw_score) {
    	throw std::runtime_error("This prediction type is not implmented");
    }
    predict_fun_ = predict_ftor(this, kFeatureThreshold, KSparseThreshold);
  }

  /*!
  * \brief Destructor
  */
  ~Predictor() {
  }

  inline const PredictFunction& GetPredictFunction() const {
    return predict_fun_;
  }

private:

  void CopyToPredictBuffer(double* pred_buf, const std::vector<std::pair<int, double>>& features) {
    int loop_size = static_cast<int>(features.size());
    for (int i = 0; i < loop_size; ++i) {
      if (features[i].first < num_feature_) {
        pred_buf[features[i].first] = features[i].second;
      }
    }
  }

  void ClearPredictBuffer(double* pred_buf, size_t buf_size, const std::vector<std::pair<int, double>>& features) {
    if (features.size() < static_cast<size_t>(buf_size / 2)) {
      std::memset(pred_buf, 0, sizeof(double)*(buf_size));
    } else {
      int loop_size = static_cast<int>(features.size());
      for (int i = 0; i < loop_size; ++i) {
        if (features[i].first < num_feature_) {
          pred_buf[features[i].first] = 0.0f;
        }
      }
    }
  }

  std::unordered_map<int, double> CopyToPredictMap(const std::vector<std::pair<int, double>>& features) {
    std::unordered_map<int, double> buf;
    int loop_size = static_cast<int>(features.size());
    for (int i = 0; i < loop_size; ++i) {
      if (features[i].first < num_feature_) {
        buf[features[i].first] = features[i].second;
      }
    }
    return std::move(buf);
  }

  /*! \brief Boosting model */
  const Boosting* boosting_;
  /*! \brief function for prediction */
  PredictFunction predict_fun_;
  PredictionEarlyStopInstance early_stop_;
  int num_feature_;
  int num_pred_one_row_;
  int num_threads_;
  std::vector<std::vector<double>> predict_buf_;
};

}  // namespace LightGBM

namespace {
	predict_ftor::predict_ftor(LightGBM::Predictor *predictor, const int kFeatureThreshold, const size_t KSparseThreshold)
			: predictor_(predictor), kFeatureThreshold_(kFeatureThreshold), KSparseThreshold_(KSparseThreshold)
	{}

	void predict_ftor::operator() (const std::vector<std::pair<int, double>>& features, double* output) {
		int tid = omp_get_thread_num();
		if (predictor_->num_feature_ > kFeatureThreshold_ && features.size() < KSparseThreshold_) {
			auto buf = predictor_->CopyToPredictMap(features);
			predictor_->boosting_->PredictByMap(buf, output, &predictor_->early_stop_);
		} else {
			predictor_->CopyToPredictBuffer(predictor_->predict_buf_[tid].data(), features);
			predictor_->boosting_->Predict (predictor_->predict_buf_[tid].data(), output, &predictor_->early_stop_);
			predictor_->ClearPredictBuffer (predictor_->predict_buf_[tid].data(), predictor_->predict_buf_[tid].size(), features);
		}
	}
}

#endif   // LightGBM_PREDICTOR_HPP_
