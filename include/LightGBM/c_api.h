#ifndef LIGHTGBM_C_API_H_
#define LIGHTGBM_C_API_H_

// #include <LightGBM/meta.h>
#include <LightGBM/utils/log.h>

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <cstring>
#include <string>

/*!
* To avoid type conversion on large data, most of our expose interface support both for float_32 and float_64.
* Except following:
* 1. gradients and hessians.
* 2. Get current score for training data and validation
* The reason is because they are called frequently, the type-conversion on them maybe time cost.
*/

#include <LightGBM/export.h>

typedef void* DatasetHandle;
typedef void* BoosterHandle;

#define C_API_DTYPE_FLOAT32 (0)
#define C_API_DTYPE_FLOAT64 (1)
#define C_API_DTYPE_INT32   (2)
#define C_API_DTYPE_INT64   (3)

#define C_API_PREDICT_NORMAL     (0)
#define C_API_PREDICT_RAW_SCORE  (1)
#define C_API_PREDICT_LEAF_INDEX (2)
#define C_API_PREDICT_CONTRIB    (3)

/*!
* \brief get string message of the last error
*  all function in this file will return 0 when succeed
*  and -1 when an error occured,
* \return const char* error inforomation
*/
LIGHTGBM_C_EXPORT const char* LGBM_GetLastError();

// --- start Booster interfaces

/*!
* \brief load an existing boosting from string
* \param model_str model string
* \param out_num_iterations number of iterations of this booster
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterLoadModelFromString(
  const char* model_str,
  int* out_num_iterations,
  BoosterHandle* out);

/*!
* \brief free obj in handle
* \param handle handle to be freed
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterFree(BoosterHandle handle);

/*!
* \brief Get number of class
* \param handle handle
* \param out_len number of class
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumClasses(BoosterHandle handle, int* out_len);

/*!
* \brief Get number of features
* \param out_len total number of features
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumFeature(BoosterHandle handle, int* out_len);

/*!
* \brief make prediction for an new data set
*        Note:  should pre-allocate memory for out_result,
*               for noraml and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param data pointer to the data space
* \param data_type type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
* \param nrow number of rows
* \param ncol number columns
* \param is_row_major 1 for row major, 0 for column major
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter Other parameters for the parameters, e.g. early stopping for prediction.
* \param out_len len of output result
* \param out_result used to set a pointer to array, should allocate memory before call this function
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForMat(BoosterHandle handle,
                                                const void* data,
                                                int data_type,
                                                int32_t nrow,
                                                int32_t ncol,
                                                int is_row_major,
                                                int predict_type,
                                                int num_iteration,
                                                const char* parameter,
                                                int64_t* out_len,
                                                double* out_result);

// exception handle and error msg
static char* LastErrorMsg() { static THREAD_LOCAL char err_msg[512] = "Everything is fine"; return err_msg; }

#pragma warning(disable : 4996)
inline void LGBM_SetLastError(const char* msg) {
  std::strcpy(LastErrorMsg(), msg);
}

inline int LGBM_APIHandleException(const std::exception& ex) {
  LGBM_SetLastError(ex.what());
  return -1;
}
inline int LGBM_APIHandleException(const std::string& ex) {
  LGBM_SetLastError(ex.c_str());
  return -1;
}

#define API_BEGIN() try {

#define API_END() } \
catch(std::exception& ex) { return LGBM_APIHandleException(ex); } \
catch(std::string& ex) { return LGBM_APIHandleException(ex); } \
catch(...) { return LGBM_APIHandleException("unknown exception"); } \
return 0;

#endif // LIGHTGBM_C_API_H_
