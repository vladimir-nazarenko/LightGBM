#ifndef LIGHTGBM_BOOSTING_GBDT_H_
#define LIGHTGBM_BOOSTING_GBDT_H_

#include <LightGBM/boosting.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/tree.h>

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <mutex>
#include <map>

namespace LightGBM {

/*!
* \brief GBDT algorithm implementation. including Training, prediction, bagging.
*/
class GBDT : public GBDTBase {
public:

  /*!
  * \brief Constructor
  */
  GBDT();

  /*!
  * \brief Destructor
  */
  ~GBDT();


  /*!
  * \brief Reset Boosting Config
  * \param gbdt_config Config for boosting
  */
  void ResetConfig(const Config* gbdt_config) override;

    /*!
  * \brief Get current iteration
  */
    int GetCurrentIteration() const override { return static_cast<int>(models_.size()) / num_tree_per_iteration_; }

  /*!
  * \brief Can use early stopping for prediction or not
  * \return True if cannot use early stopping for prediction
  */
  bool NeedAccuratePrediction() const override {
    if (objective_function_ == nullptr) {
      return true;
    } else {
      return objective_function_->NeedAccuratePrediction();
    }
  }

  /*!
  * \brief Get number of prediction for one data
  * \param num_iteration number of used iterations
  * \param is_pred_leaf True if predicting  leaf index
  * \param is_pred_contrib True if predicting feature contribution
  * \return number of prediction
  */
  inline int NumPredictOneRow(int , bool is_pred_leaf, bool is_pred_contrib) const override {
    int num_preb_in_one_row = num_class_;
    if (is_pred_contrib || is_pred_leaf) {
      throw std::runtime_error("Invalid mode!");
    }
    return num_preb_in_one_row;
  }

  void PredictRaw(const double* features, double* output,
                  const PredictionEarlyStopInstance* earlyStop) const override;

  void PredictRawByMap(const std::unordered_map<int, double>& features, double* output,
                       const PredictionEarlyStopInstance* early_stop) const override;

  void Predict(const double* features, double* output,
               const PredictionEarlyStopInstance* earlyStop) const override;

  void PredictByMap(const std::unordered_map<int, double>& features, double* output,
                    const PredictionEarlyStopInstance* early_stop) const override;

  void PredictLeafIndex(const double* features, double* output) const override;

  void PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const override;

  /*!
  * \brief Restore from a serialized buffer
  */
  bool LoadModelFromString(const char* buffer, size_t len) override;

  /*!
  * \brief Get max feature index of this model
  * \return Max feature index of this model
  */
  inline int MaxFeatureIdx() const override { return max_feature_idx_; }

  /*!
  * \brief Get feature names of this model
  * \return Feature names of this model
  */
  inline std::vector<std::string> FeatureNames() const override { return feature_names_; }

  /*!
  * \brief Get index of label column
  * \return index of label column
  */
  inline int LabelIdx() const override { return label_idx_; }

  /*!
  * \brief Get number of weak sub-models
  * \return Number of weak sub-models
  */
  inline int NumberOfTotalModel() const override { return static_cast<int>(models_.size()); }

  /*!
  * \brief Get number of tree per iteration
  * \return number of tree per iteration
  */
  inline int NumModelPerIteration() const override { return num_tree_per_iteration_; }

  /*!
  * \brief Get number of classes
  * \return Number of classes
  */
  inline int NumberOfClasses() const override { return num_class_; }

  inline void InitPredict(int num_iteration, bool is_pred_contrib) override {
    num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
    if (num_iteration > 0) {
      num_iteration_for_pred_ = std::min(num_iteration, num_iteration_for_pred_);
    }
    if (is_pred_contrib) {
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(models_.size()); ++i) {
        models_[i]->RecomputeMaxDepth();
      }
    }
  }

  inline double GetLeafValue(int tree_idx, int leaf_idx) const override {
    CHECK(tree_idx >= 0 && static_cast<size_t>(tree_idx) < models_.size());
    CHECK(leaf_idx >= 0 && leaf_idx < models_[tree_idx]->num_leaves());
    return models_[tree_idx]->LeafOutput(leaf_idx);
  }

  inline void SetLeafValue(int tree_idx, int leaf_idx, double val) override {
    CHECK(tree_idx >= 0 && static_cast<size_t>(tree_idx) < models_.size());
    CHECK(leaf_idx >= 0 && leaf_idx < models_[tree_idx]->num_leaves());
    models_[tree_idx]->SetLeafOutput(leaf_idx, val);
  }

  /*!
  * \brief Get Type name of this boosting object
  */
  virtual const char* SubModelName() const override { return "tree"; }

protected:

  /*! \brief current iteration */
  int iter_;
  /*! \brief Pointer to training data */
  const Dataset* train_data_;
  /*! \brief Config of gbdt */
  std::unique_ptr<Config> config_;
  /*! \brief Objective function */
  const ObjectiveFunction* objective_function_;
  /*! \brief Metrics for training data */
  std::vector<const Metric*> training_metrics_;
  /*! \brief Metric for validation data */
  std::vector<std::vector<const Metric*>> valid_metrics_;
  /*! \brief Number of rounds for early stopping */
  int early_stopping_round_;
  /*! \brief Best iteration(s) for early stopping */
  std::vector<std::vector<int>> best_iter_;
  /*! \brief Best score(s) for early stopping */
  std::vector<std::vector<double>> best_score_;
  /*! \brief output message of best iteration */
  std::vector<std::vector<std::string>> best_msg_;
  /*! \brief Trained models(trees) */
  std::vector<std::unique_ptr<Tree>> models_;
  /*! \brief Max feature index of training data*/
  int max_feature_idx_;
  /*! \brief First order derivative of training data */
  std::vector<score_t> gradients_;
  /*! \brief Secend order derivative of training data */
  std::vector<score_t> hessians_;
  /*! \brief Store the indices of in-bag data */
  std::vector<data_size_t> bag_data_indices_;
  /*! \brief Number of in-bag data */
  data_size_t bag_data_cnt_;
  /*! \brief Store the indices of in-bag data */
  std::vector<data_size_t> tmp_indices_;
  /*! \brief Number of training data */
  data_size_t num_data_;
  /*! \brief Number of trees per iterations */
  int num_tree_per_iteration_;
  /*! \brief Number of class */
  int num_class_;
  /*! \brief Index of label column */
  data_size_t label_idx_;
  /*! \brief number of used model */
  int num_iteration_for_pred_;
  /*! \brief Shrinkage rate for one iteration */
  double shrinkage_rate_;
  /*! \brief Number of loaded initial models */
  int num_init_iteration_;
  /*! \brief Feature names */
  std::vector<std::string> feature_names_;
  std::vector<std::string> feature_infos_;
  /*! \brief number of threads */
  int num_threads_;
  /*! \brief Buffer for multi-threading bagging */
  std::vector<data_size_t> offsets_buf_;
  /*! \brief Buffer for multi-threading bagging */
  std::vector<data_size_t> left_cnts_buf_;
  /*! \brief Buffer for multi-threading bagging */
  std::vector<data_size_t> right_cnts_buf_;
  /*! \brief Buffer for multi-threading bagging */
  std::vector<data_size_t> left_write_pos_buf_;
  /*! \brief Buffer for multi-threading bagging */
  std::vector<data_size_t> right_write_pos_buf_;
  bool is_use_subset_;
  std::vector<bool> class_need_train_;
  std::vector<double> class_default_output_;
  bool is_constant_hessian_;
  std::unique_ptr<ObjectiveFunction> loaded_objective_;
  bool average_output_;
  bool need_re_bagging_;
  std::string loaded_parameter_;

};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_GBDT_H_
