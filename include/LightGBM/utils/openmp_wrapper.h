#ifndef LIGHTGBM_OPENMP_WRAPPER_H_
#define LIGHTGBM_OPENMP_WRAPPER_H_
#ifdef _OPENMP

#include <omp.h>
#include <exception>
#include <stdexcept>
#include <mutex>
#include <vector>
#include <memory>
#include <boost/thread/mutex.hpp>
#include <boost/exception_ptr.hpp>
#include "log.h"

#if __GNUC__
#define nullptr ((void *)0)
#endif

class ThreadExceptionHelper {
public:
  ThreadExceptionHelper() { 
    ex_ptr_ = empty_ptr_;
  }

  ~ThreadExceptionHelper() { 
    ReThrow();
  }
  void ReThrow() {
    if (ex_ptr_ != empty_ptr_) {
      boost::rethrow_exception(ex_ptr_);
    }
  }
  void CaptureException() {
    // only catch first exception.
    if (ex_ptr_ != empty_ptr_) { return; }
    boost::unique_lock<boost::mutex> guard(lock_);
    if (ex_ptr_ != empty_ptr_) { return; }
    ex_ptr_ = boost::current_exception();
  }
private:
  boost::exception_ptr ex_ptr_;
  const boost::exception_ptr empty_ptr_;
  boost::mutex lock_;
};

#define OMP_INIT_EX() ThreadExceptionHelper omp_except_helper
#define OMP_LOOP_EX_BEGIN() try {

#define OMP_LOOP_EX_END() } \
catch(std::exception& ex) { Log::Warning(ex.what()); omp_except_helper.CaptureException(); } \
catch(...) { omp_except_helper.CaptureException();  }
#define OMP_THROW_EX() omp_except_helper.ReThrow()

#else

#ifdef _MSC_VER
  #pragma warning( disable : 4068 ) // disable unknown pragma warning
#endif

#ifdef __cplusplus
  extern "C" {
#endif
  /** Fall here if no OPENMP support, so just
      simulate a single thread running.
      All #pragma omp should be ignored by the compiler **/
  inline void omp_set_num_threads(int) {}
  inline void omp_set_nested(int) {}
  inline int omp_get_num_threads() {return 1;}
  inline int omp_get_thread_num() {return 0;}
#ifdef __cplusplus
}; // extern "C"
#endif

#define OMP_INIT_EX()
#define OMP_LOOP_EX_BEGIN()
#define OMP_LOOP_EX_END()
#define OMP_THROW_EX()

#endif



#endif /* LIGHTGBM_OPENMP_WRAPPER_H_ */
