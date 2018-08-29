#include "gbdt.h"

#include <LightGBM/utils/common.h>
#include <LightGBM/objective_function.h>

#include <sstream>
#include <string>
#include <vector>

namespace LightGBM {

const std::string kModelVersion = "v2";

bool GBDT::LoadModelFromString(const char* buffer, size_t len) {
  // use serialized string to restore this object
  models_.clear();
  auto c_str = buffer;
  auto p = c_str;
  auto end = p + len;
  std::unordered_map<std::string, std::string> key_vals;
  while (p < end) {
    auto line_len = Common::GetLine(p);
    std::string cur_line(p, line_len);
    if (line_len > 0) {
      if (!Common::StartsWith(cur_line, "Tree=")) {
        auto strs = Common::Split(cur_line.c_str(), '=');
        if (strs.size() == 1) {
          key_vals[strs[0]] = "";
        }
        else if (strs.size() == 2) {
          key_vals[strs[0]] = strs[1];
        }
        else if (strs.size() > 2) {
          if (strs[0] == "feature_names") {
            key_vals[strs[0]] = cur_line.substr(std::strlen("feature_names="));
          } else {
            // Use first 128 chars to avoid exceed the message buffer.
            Log::Fatal("Wrong line at model file: %s", cur_line.substr(0, std::min<size_t>(128, cur_line.size())).c_str());
          }
        }
      }
      else {
        break;
      }
    }
    p += line_len;
    p = Common::SkipNewLine(p);
  }

  // get number of classes
  if (key_vals.count("num_class")) {
    Common::Atoi(key_vals["num_class"].c_str(), &num_class_);
  } else {
    Log::Fatal("Model file doesn't specify the number of classes");
    return false;
  }

  if (key_vals.count("num_tree_per_iteration")) {
    Common::Atoi(key_vals["num_tree_per_iteration"].c_str(), &num_tree_per_iteration_);
  } else {
    num_tree_per_iteration_ = num_class_;
  }

  // get index of label
  if (key_vals.count("label_index")) {
    Common::Atoi(key_vals["label_index"].c_str(), &label_idx_);
  } else {
    Log::Fatal("Model file doesn't specify the label index");
    return false;
  }

  // get max_feature_idx first
  if (key_vals.count("max_feature_idx")) {
    Common::Atoi(key_vals["max_feature_idx"].c_str(), &max_feature_idx_);
  } else {
    Log::Fatal("Model file doesn't specify max_feature_idx");
    return false;
  }

  // get average_output
  if (key_vals.count("average_output")) {
    average_output_ = true;
  }

  // get feature names
  if (key_vals.count("feature_names")) {
    feature_names_ = Common::Split(key_vals["feature_names"].c_str(), ' ');
    if (feature_names_.size() != static_cast<size_t>(max_feature_idx_ + 1)) {
      Log::Fatal("Wrong size of feature_names");
      return false;
    }
  } else {
    Log::Fatal("Model file doesn't contain feature_names");
    return false;
  }

  if (key_vals.count("feature_infos")) {
    feature_infos_ = Common::Split(key_vals["feature_infos"].c_str(), ' ');
    if (feature_infos_.size() != static_cast<size_t>(max_feature_idx_ + 1)) {
      Log::Fatal("Wrong size of feature_infos");
      return false;
    }
  } else {
    Log::Fatal("Model file doesn't contain feature_infos");
    return false;
  }

  if (key_vals.count("objective")) {
    auto str = key_vals["objective"];
    loaded_objective_.reset(ObjectiveFunction::CreateObjectiveFunction(str));
    objective_function_ = loaded_objective_.get();
  }
  if (!key_vals.count("tree_sizes")) {
    while (p < end) {
      auto line_len = Common::GetLine(p);
      std::string cur_line(p, line_len);
      if (line_len > 0) {
        if (Common::StartsWith(cur_line, "Tree=")) {
          p += line_len;
          p = Common::SkipNewLine(p);
          size_t used_len = 0;
          models_.emplace_back(new Tree(p, &used_len));
          p += used_len;
        }
        else {
          break;
        }
      }
      p = Common::SkipNewLine(p);
    }
  } else {
    std::vector<size_t> tree_sizes = Common::StringToArray<size_t>(key_vals["tree_sizes"].c_str(), ' ');
    std::vector<size_t> tree_boundries(tree_sizes.size() + 1, 0);
    int num_trees = static_cast<int>(tree_sizes.size());
    for (int i = 0; i < num_trees; ++i) {
      tree_boundries[i + 1] = tree_boundries[i] + tree_sizes[i];
      models_.emplace_back(nullptr);
    }
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_trees; ++i) {
      OMP_LOOP_EX_BEGIN();
      auto cur_p = p + tree_boundries[i];
      auto line_len = Common::GetLine(cur_p);
      std::string cur_line(cur_p, line_len);
      if (Common::StartsWith(cur_line, "Tree=")) {
        cur_p += line_len;
        cur_p = Common::SkipNewLine(cur_p);
        size_t used_len = 0;
        models_[i].reset(new Tree(cur_p, &used_len));
      } else {
        Log::Fatal("Model format error, expect a tree here. met %s", cur_line.c_str());
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }
  num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
  num_init_iteration_ = num_iteration_for_pred_;
  iter_ = 0;
  bool is_inparameter = false;
  std::stringstream ss;
  while (p < end) {
    auto line_len = Common::GetLine(p);
    std::string cur_line(p, line_len);
    if (line_len > 0) {
      if (cur_line == std::string("parameters:")) {
        is_inparameter = true;
      } else if (cur_line == std::string("end of parameters")) {
        break;
      } else if (is_inparameter) {
        ss << cur_line << "\n";
      }
    }
    p += line_len;
    p = Common::SkipNewLine(p);
  }
  if (!ss.str().empty()) {
    loaded_parameter_ = ss.str();
  }
  return true;
}

}  // namespace LightGBM
