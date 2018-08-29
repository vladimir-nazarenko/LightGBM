#ifndef LIGHTGBM_BOOSTING_H_
#define LIGHTGBM_BOOSTING_H_

#include <LightGBM/meta.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/config.h>

#include <vector>
#include <string>
#include <map>

namespace LightGBM {

/*! \brief forward declaration */
class Dataset;
class ObjectiveFunction;
class Metric;
struct PredictionEarlyStopInstance;

/*!
* \brief The interface for Boosting
*/
class LIGHTGBM_EXPORT Boosting {
public:
  /*! \brief virtual destructor */
  virtual ~Boosting() {}

  virtual void ResetConfig(const Config* config) = 0;

  /*!
  * \brief return current iteration
  */
  virtual int GetCurrentIteration() const = 0;

  virtual int NumPredictOneRow(int num_iteration, bool is_pred_leaf, bool is_pred_contrib) const = 0;

  /*!
  * \brief Prediction for one record, not sigmoid transform
  * \param feature_values Feature value on this record
  * \param output Prediction result for this record
  * \param early_stop Early stopping instance. If nullptr, no early stopping is applied and all models are evaluated.
  */
  virtual void PredictRaw(const double* features, double* output,
                          const PredictionEarlyStopInstance* early_stop) const = 0;

  virtual void PredictRawByMap(const std::unordered_map<int, double>& features, double* output,
                               const PredictionEarlyStopInstance* early_stop) const = 0;


  /*!
  * \brief Prediction for one record, sigmoid transformation will be used if needed
  * \param feature_values Feature value on this record
  * \param output Prediction result for this record
  * \param early_stop Early stopping instance. If nullptr, no early stopping is applied and all models are evaluated.
  */
  virtual void Predict(const double* features, double* output,
                       const PredictionEarlyStopInstance* early_stop) const = 0;

  virtual void PredictByMap(const std::unordered_map<int, double>& features, double* output,
                            const PredictionEarlyStopInstance* early_stop) const = 0;


  /*!
  * \brief Prediction for one record with leaf index
  * \param feature_values Feature value on this record
  * \param output Prediction result for this record
  */
  virtual void PredictLeafIndex(
    const double* features, double* output) const = 0;

  virtual void PredictLeafIndexByMap(
    const std::unordered_map<int, double>& features, double* output) const = 0;

  /*!
  * \brief Restore from a serialized string
  * \param buffer The content of model
  * \param len The length of buffer
  * \return true if succeeded
  */
  virtual bool LoadModelFromString(const char* buffer, size_t len) = 0;

  /*!
  * \brief Get max feature index of this model
  * \return Max feature index of this model
  */
  virtual int MaxFeatureIdx() const = 0;

  /*!
  * \brief Get feature names of this model
  * \return Feature names of this model
  */
  virtual std::vector<std::string> FeatureNames() const = 0;

  /*!
  * \brief Get index of label column
  * \return index of label column
  */
  virtual int LabelIdx() const = 0;

  /*!
  * \brief Get number of weak sub-models
  * \return Number of weak sub-models
  */
  virtual int NumberOfTotalModel() const = 0;

  /*!
  * \brief Get number of models per iteration
  * \return Number of models per iteration
  */
  virtual int NumModelPerIteration() const = 0;

  /*!
  * \brief Get number of classes
  * \return Number of classes
  */
  virtual int NumberOfClasses() const = 0;

  /*! \brief The prediction should be accurate or not. True will disable early stopping for prediction. */
  virtual bool NeedAccuratePrediction() const = 0;

  /*!
  * \brief Initial work for the prediction
  * \param num_iteration number of used iteration
  * \param is_pred_contrib
  */
  virtual void InitPredict(int num_iteration, bool is_pred_contrib) = 0;

  /*!
  * \brief Name of submodel
  */
  virtual const char* SubModelName() const = 0;

  Boosting() = default;
  /*! \brief Disable copy */
  Boosting& operator=(const Boosting&) = delete;
  /*! \brief Disable copy */
  Boosting(const Boosting&) = delete;

  static bool LoadFileToBoosting(Boosting* boosting, const char* filename);

  /*!
  * \brief Create boosting object
  * \param type Type of boosting
  * \param format Format of model
  * \param config config for boosting
  * \param filename name of model file, if existing will continue to train from this model
  * \return The boosting object
  */
  static Boosting* CreateBoosting(const std::string& type, const char* filename);

};

class GBDTBase : public Boosting {
public:
  virtual double GetLeafValue(int tree_idx, int leaf_idx) const = 0;
  virtual void SetLeafValue(int tree_idx, int leaf_idx, double val) = 0;
};

}  // namespace LightGBM

#endif   // LightGBM_BOOSTING_H_
