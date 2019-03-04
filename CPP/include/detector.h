//
// Created by andylei77@qq.com
//

#ifndef TF_EXAMPLE_DETECTOR_H
#define TF_EXAMPLE_DETECTOR_H

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;

namespace detector {

// detect_roi
struct ROI {
  int cls;
  float score;
  cv::Rect box;
  cv::Mat mask;
  bool is_valid = false;
  int id; // origin detect_id

  friend ostream &operator<<(ostream &stream, const ROI &roi) {
    stream
        << " id:" << roi.id
        << " is_valid:" << roi.is_valid
        << " cls:" << roi.cls
        << " score:" << roi.score
        << " box:" << roi.box
        << std::endl;
    return stream;
  }
};

class Detector{
  public:

      Detector(const std::string& graph_path, const std::string& image_path);

      void Detect(const std::string& image_path, std::vector<ROI>& rois);

  private:

      //Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
      //                      size_t* found_label_count);

      //Status ReadEntireFile(tensorflow::Env* env, const string& filename,
      //                                       Tensor* output);

      //Status ReadTensorFromImageFile(const string& file_name,
      //                         const float input_mean,
      //                         const float input_std);

      //Status ParseMaskTensor(const Tensor& detection_boxes_input,
      //                 const Tensor& detection_masks_input,
      //                 const int real_num_detection,
      //                 const int image_height,
      //                 const int image_width,
      //                 std::vector<Tensor>* out_tensors);

      Status LoadGraph(const std::string& graph_path,
                                 std::unique_ptr<tensorflow::Session>* session);

  private:

      std::unique_ptr<tensorflow::Session> session;

  };
}

#endif //TF_EXAMPLE_DETECTOR_H
