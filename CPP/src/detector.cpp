//
// Created by andylei77@qq.com
//

#include "detector.h"

using namespace detector;

/*
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
void Print(cv::Mat image, std::string name){
      std::cout << name
              << " rows:" << image.rows
              << " cols:" << image.cols
              << " channels:" << image.channels()
              << " type:" << type2str(image.type())
              << std::endl;
}
*/


/*
// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status Detector::ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
    std::ifstream file(file_name);
    if (!file) {
        return tensorflow::errors::NotFound("Labels file ", file_name,
                                            " not found.");
    }
    result->clear();
    string line;
    while (std::getline(file, line)) {
        result->push_back(line);
    }
    *found_label_count = result->size();
    const int padding = 16;
    while (result->size() % padding) {
        result->emplace_back();
    }
    return Status::OK();
}

Status Detector::ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                            "' expected ", file_size, " got ",
                                            data.size());
    }
    output->scalar<string>()() = data.ToString();
    return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status Detector::ReadTensorFromImageFile(const string& file_name,
                                         const float input_mean,
                                         const float input_std) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    // use a placeholder to read input data
    auto file_reader =
            Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::ops::Output image_reader;

    std::vector<string> vec_strs = tensorflow::str_util::Split(file_name, '.');
    std::string subfix = vec_strs[vec_strs.size()-1];
    //if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    //if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    if ( subfix == "png") {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                                 DecodePng::Channels(wanted_channels));

    //} else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    //} else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    } else if (subfix == "gif") {
        // gif decoder returns 4-D tensor, remove the first dim
        image_reader =
                Squeeze(root.WithOpName("squeeze_first_dim"),
                        DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else {
        // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
        image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                  DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    // auto float_caster =
    //     Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

    auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);

    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

    // Bilinearly resize the image to fit the required dimensions.
    // auto resized = ResizeBilinear(
    //     root, dims_expander,
    //     Const(root.WithOpName("size"), {input_height, input_width}));


    // Subtract the mean and divide by the scale.
    // auto div =  Div(root.WithOpName(output_name), Sub(root, dims_expander, {input_mean}),
    //     {input_std});


    //cast to int
    //auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), div, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    //TF_RETURN_IF_ERROR(session->Extend(graph));
    Status res = session->Extend(graph);
    if(!res.ok()){
      LOG(ERROR) << res.error_message();
      return res;
    }

    //std::unique_ptr<tensorflow::Session> session(
    //        tensorflow::NewSession(tensorflow::SessionOptions()));
    //TF_RETURN_IF_ERROR(session->Create(graph));

    //Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    //TF_RETURN_IF_ERROR(
    //  ReadEntireFile(tensorflow::Env::Default(), file_name, &input));
    //std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    //        {"input", input},
    //};
    //TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
    return Status::OK();
}

Status Detector::ParseMaskTensor(const Tensor& detection_boxes_input,
                       const Tensor& detection_masks_input,
                       //const Tensor& num_detections_input,
                       const int real_num_detection,
                       const int image_height,
                       const int image_width,
                       std::vector<Tensor>* out_tensors) {


    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    auto detection_boxes_raw =
            Placeholder(root.WithOpName("detection_boxes_raw"), tensorflow::DataType::DT_FLOAT);
    auto detection_masks_raw =
            Placeholder(root.WithOpName("detection_masks_raw"), tensorflow::DataType::DT_FLOAT);
    //auto num_detections_raw =
    //        Placeholder(root.WithOpName("num_detections_raw"), tensorflow::DataType::DT_FLOAT);


    // the following processing is only for single image
    tensorflow::ops::Output detection_boxes;
    detection_boxes = Squeeze(root.WithOpName("detection_boxes_squeezed"), detection_boxes_raw, Squeeze::Axis({0}));
    tensorflow::Output detection_masks;
    detection_masks = Squeeze(root.WithOpName("detection_masks_squeezed"), detection_masks_raw, Squeeze::Axis({0}));

    // reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    // real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    //auto real_num_detection =
    //    Cast(root.WithOpName("real_num_detection"),
    //        Slice(root.WithOpName("real_num_detection_slice"),
    //              num_detections_raw, {0},{1}),
    //    tensorflow::DT_INT32);

    //detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    auto detection_boxes_filted = Slice(root.WithOpName("detection_boxes_filted"), detection_boxes, {0,0},
                                        {real_num_detection, -1});

    //detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    auto detection_masks_filted = Slice(root.WithOpName("detection_masks_filted"), detection_masks, {0,0,0}, {real_num_detection, -1, -1});

    //detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
    //    detection_masks_filted, detection_boxes_filted, image_height, image_width)
    // todo check num_detection == 0
    // box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    auto box_masks_expanded = ExpandDims(root.WithOpName("box_masks_expanded"), detection_masks_filted, 3);
    //num_boxes = tf.shape(box_masks_expanded)[0]
    //auto num_boxes = Slice(root.WithOpName("num_boxes"),  Shape(root.WithOpName("box_masks_expanded_shape"), box_masks_expanded), {0}, {1});

    //unit_boxes = tf.concat( [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    tensorflow::Input zeros = Fill(root.WithOpName("zeros_num_boxes_2"),{real_num_detection,2}, 0.0);
    tensorflow::Input ones = Fill(root.WithOpName("ones_num_boxes_2"), {real_num_detection,2}, 1.0);
    auto unit_boxes =
        Cast(root.WithOpName("unit_boxes_float"),
            Concat(root.WithOpName("unit_boxes"),
                             tensorflow::InputList({zeros,ones}),
                             1),
        tensorflow::DT_FLOAT
        );

    //reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, detection_boxes_filted)
    //def transform_boxes_relative_to_boxes(unit_boxes, detection_boxes_filted):
    //  unit_boxes_reshape = tf.reshape(unit_boxes, [-1, 2, 2])
    auto unit_boxes_reshape = Reshape(root.WithOpName("unit_boxes_reshape"), unit_boxes, {-1,2,2});

    //min_corner = tf.expand_dims(detection_boxes_filted[:, 0:2], 1)
    auto min_corner = ExpandDims(root.WithOpName("min_corner"), Slice(root.WithOpName("min_corner_raw"),detection_boxes_filted, {0,0}, {-1,2}), 1);

    //max_corner = tf.expand_dims(detection_boxes_filted[:, 2:4], 1)
    auto max_corner = ExpandDims(root.WithOpName("max_corner"), Slice(root.WithOpName("max_corner_raw"),detection_boxes_filted, {0,2}, {-1,2}), 1);

    //transformed_boxes = (unit_boxes_reshape - min_corner) / (max_corner - min_corner)
    auto transformed_boxes = Div(root.WithOpName("transformed_boxes"),
        Subtract(root.WithOpName("min_corner_convert"), unit_boxes_reshape, min_corner),
        Subtract(root.WithOpName("max_corner_convert"), max_corner, min_corner));

    //return tf.reshape(transformed_boxes, [-1, 4])
    auto transformed_boxes_reshape = Reshape(root.WithOpName("transformed_boxes_reshape"), transformed_boxes, {-1,4});


    //return tf.image.crop_and_resize(
    //    image=box_masks_expanded,
    //    boxes=transformed_boxes_reshape,
    //    box_ind=tf.range(num_boxes),
    //    crop_size=[image_height, image_width],
    //    extrapolation_value=0.0)
    auto detection_masks_reframed_raw = CropAndResize(root.WithOpName("detection_masks_reframed_raw"),
        box_masks_expanded,
        transformed_boxes_reshape,
        //Range(root.WithOpName("boxes_index"),0, num_boxes,1),
        Range(root.WithOpName("boxes_index"), 0, real_num_detection, 1),
        {image_height, image_width},
        CropAndResize::ExtrapolationValue(0.0));

    //detection_masks_reframed = tf.cast(
    //    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    // follow the convention by adding back the batch dimension
    //tensor_dict['detection_masks'] = tf.expand_dims(
    //    detection_masks_reframed, 0)

  auto detection_masks_reframed_greated =
      Greater(root.WithOpName("detection_masks_reframed_greated"),
              detection_masks_reframed_raw,
              .5f);
  //auto detection_masks_reframed_any =
  //    Any(root.WithOpName("detection_masks_reframed_any"),
  //        detection_masks_reframed_greated,
  //        0,
  //        Any::KeepDims(false));

  auto detection_masks_reframed =
      Cast(root.WithOpName("detection_masks_reframed"),
          detection_masks_reframed_greated,
          tensorflow::DataType::DT_FLOAT);



    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    //TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
    Status to_graph_res = root.ToGraphDef(&graph);
    if(!to_graph_res.ok()){
        LOG(ERROR) << to_graph_res.error_message();
    }

    std::unique_ptr<tensorflow::Session> session(
            tensorflow::NewSession(tensorflow::SessionOptions()));
    //TF_RETURN_IF_ERROR(session->Create(graph));
    Status session_create_res = session->Create(graph);
    if(!session_create_res.ok()){
        LOG(ERROR) << session_create_res.error_message();
    }

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            {"detection_boxes_raw", detection_boxes_input},
            {"detection_masks_raw", detection_masks_input}
            //{"num_detections_raw",num_detections_input}
    };
    //TF_RETURN_IF_ERROR(session->Run({inputs}, {"detection_masks_reframed"}, {}, out_tensors));
  Status session_run_res = session->Run(
      {inputs},
      {"detection_masks_reframed"},
      //{"detection_masks_reframed_raw"},
      {},
      out_tensors);
    if(!session_run_res.ok()){
        LOG(ERROR) << session_run_res.error_message();
    }

    return Status::OK();
}
*/

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status Detector::LoadGraph(const std::string& graph_path,
                 std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_path, "'");
    }

    for(int i=0; i< graph_def.node_size(); ++i){
        std::string keyword = "Postprocessor";
        if(graph_def.node(i).name().compare(0,keyword.size(), keyword) == 0){
            graph_def.mutable_node(i)->set_device("/cpu:0");
            std::cout << "cccccccccccccccccccccccccccc set cpu " << graph_def.node(i).name() <<  std::endl;
        }else{
            graph_def.mutable_node(i)->set_device("/gpu:0");
            std::cout << "gggggggggggggggggggggggggggg set gpu " << graph_def.node(i).name() <<  std::endl;
        }
    }
    tensorflow::SessionOptions session_options;
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
    session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(1.0);
    session_options.config.set_allow_soft_placement(true);

    //session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    session->reset(tensorflow::NewSession(session_options));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}


Detector::Detector(const std::string& graph_path, const std::string& image_type){
  // First we load and initialize the model.
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(FATAL) << "LoadGraph ERROR!!!!"<< load_graph_status;
  }

  //// Read image subgraph
  //float input_mean = 0;
  //float input_std = 255;
  //Status read_tensor_status = ReadTensorFromImageFile(image_type,
  //                            input_mean,
  //                            input_std);
  //if (!read_tensor_status.ok()) {
  //  LOG(FATAL) << read_tensor_status;
  //}
  // Parse mask subgraph

}

void Detector::Detect(const std::string& image_path, std::vector<ROI>& rois){

  auto start1 = std::chrono::high_resolution_clock::now();

  float input_mean = 0;
  float input_std = 255;

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  const int image_height = image.rows;
  const int image_width = image.cols;
  if(image.empty()){
    LOG(ERROR) << "Could not open or find the image !";
    return;
  }

  // Prepare image_tensor
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  tensorflow::Tensor image_tensor(tensorflow::DT_UINT8,
    tensorflow::TensorShape({1, image_rgb.rows,image_rgb.cols,image_rgb.channels()}));
  tensorflow::uint8 * image_point = image_tensor.flat<tensorflow::uint8>().data();
  cv::Mat fake_image(image_rgb.rows, image_rgb.cols, CV_8UC3, image_point);
  image_rgb.convertTo(fake_image, CV_8UC3);

      auto end1 = std::chrono::high_resolution_clock::now();
  auto duration1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end1-start1);
  std::cout << " prepare image_tensor: " << duration1.count() / 1.e6 << " s" <<std::endl;


  /*
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  const int image_height = image.rows;
  const int image_width = image.cols;

  // Read image
  std::vector<Tensor> resized_tensors;
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  Status res = ReadEntireFile(tensorflow::Env::Default(), image_path, &input);
  if(!res.ok()){
    LOG(FATAL) << res.error_message();
  }
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
          {"input", input},
  };
  res = session->Run({inputs}, {"dim"}, {}, &resized_tensors);
  if(!res.ok()){
    LOG(FATAL) << res.error_message();
  }
  const Tensor& image_tensor = resized_tensors[0];
  */


  LOG(ERROR)
      << "image shape:" << image_tensor.shape().DebugString()
      << ",type:"<< image_tensor.dtype();
      //<< ",data:" << image_tensor.flat<tensorflow::uint8>();


  // Interface
  string input_layer = "image_tensor:0";
    //vector<string> output_layer ={ "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0", "detection_masks:0" };
    vector<string> output_layer ={ "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};
  std::vector<Tensor> outputs;

  auto start2 = std::chrono::high_resolution_clock::now();
  Status run_status = session->Run({{input_layer, image_tensor}},
                                   output_layer, {}, &outputs);
  if (!run_status.ok()) {
    LOG(FATAL) << "Running model failed: " << run_status;
  }
  auto end2 = std::chrono::high_resolution_clock::now();
  auto duration2 =
      std::chrono::duration_cast<std::chrono::microseconds>(end2-start2);
  std::cout << " inference : " << duration2.count() / 1.e6 << " s" << std::endl;


  auto detection_boxes_raw = outputs[0];
  auto detection_scores_raw = outputs[1];
  auto detection_classes_raw = outputs[2];
  auto num_detections_raw = outputs[3];
  //auto detection_masks_raw = outputs[4];

  LOG(ERROR) << "boxes" << detection_boxes_raw.DebugString();
  LOG(ERROR) << "scores" << detection_scores_raw.DebugString();
  LOG(ERROR) << "classes" << detection_classes_raw.DebugString();
  LOG(ERROR) << "num_detections" << num_detections_raw.DebugString();
  //LOG(ERROR) << "masks" << detection_masks_raw.DebugString();
  auto boxes = detection_boxes_raw.flat_inner_dims<float,2>();
  tensorflow::TTypes<float>::Flat scores = detection_scores_raw.flat<float>();
  tensorflow::TTypes<float>::Flat classes = detection_classes_raw.flat<float>();
  tensorflow::TTypes<float>::Flat num_detections = num_detections_raw.flat<float>();
  int num_detection = static_cast<int>(num_detections(0));
  LOG(ERROR) << "num_detections:" << num_detection; // << "," << outputs[0].shape().DebugString();
  if(!num_detection){
    return;
  }

   auto start3 = std::chrono::high_resolution_clock::now();

	/*
  // Parse masks
  std::vector<Tensor> output_tensors;
  Status parse_mask_status = ParseMaskTensor(
      detection_boxes_raw,
      detection_masks_raw,
      //num_detections_raw,
      num_detection,
      image_height,
      image_width,
      &output_tensors);

  Tensor& detection_masks_reframe = output_tensors[0];
  LOG(ERROR) << "masks_reframe " << detection_masks_reframe.DebugString();

  auto end3 = std::chrono::high_resolution_clock::now();
  auto duration3 =
      std::chrono::duration_cast<std::chrono::microseconds>(end3-start3);
  std::cout << " parse mask : " << duration3.count() / 1.e6 << " s" << std::endl;
  */

  /*
  auto masks = detection_masks_reframe
  .flat_outer_dims<float,3>();
  std::cout << "masks parse: ";
  for (int j = 0; j < image_height ; ++j) {
    std::cout << std::endl;
    for (int k = 0; k < image_width; ++k) {
      std::cout << masks(0,j,k) << " ";
    }
  }
  std::cout << std::endl;
  */

  auto start4 = std::chrono::high_resolution_clock::now();

  //float * image_p = detection_masks_reframe.flat<float>().data();
  for(size_t i = 0; i < num_detection; ++i) {
    const float score = scores(i);
    const float cls = classes(i);
    const int ymin = boxes(i,0)*image_height;
    const int xmin = boxes(i,1)*image_width;
    const int ymax = boxes(i,2)*image_height;
    const int xmax = boxes(i,3)*image_width;

    //cv::Mat image_mask(image_height, image_width, CV_32FC1, image_p+i*image_height*image_width);

    bool is_valid = false;
    if (score > 0.5) {
      is_valid = true;
    }

    ROI roi;
    roi.score = score;
    roi.cls = cls;
    roi.box = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    //roi.mask = image_mask;
    roi.is_valid = is_valid;
    roi.id = i;

    rois.push_back(roi);

    LOG(ERROR) << i
               << " score:" << score
               << " class:" << cls
               << " ymin:" << ymin
               << " xmin:" << xmin
               << " ymax:" << ymax
               << " xmax:" << xmax;

    cv::rectangle(image, cv::Point(xmin, ymin),
                  cv::Point(xmax, ymax),
                  cv::Scalar(0, 255, 0),
                  2);
    cv::imshow("image_box"+to_string(i), image);
    //cv::imshow("image_mask"+to_string(i), image_mask);
    cv::waitKey(0);

    /*
    float alpha = 0.5;
    float beta = 1 - alpha;
    cv::Scalar color = cv::Scalar(0,255,0);
    cv::Mat background;
    image.copyTo(background);
    cv::Mat mask;
    image_mask.copyTo(mask);

    // Get background
    background.convertTo(background, CV_32FC3);

    // Get the foreground
    mask.convertTo(mask, CV_8UC1); //
    cv::Mat foreground = cv::Mat(image.rows, image.cols, image.type(), color);
    foreground.convertTo(foreground, CV_32FC3);
    cv::Mat fg_masked;
    foreground.copyTo(fg_masked, mask);
    //cv::imshow("fg_masked",fg_masked/255);

    cv::Mat bg_inv_masked;
    background.copyTo(bg_inv_masked, 1-mask);
    //cv::imshow("bg_inv_masked", bg_inv_masked/255);

    // final froreground
    cv::add(bg_inv_masked, fg_masked, fg_masked);
    //cv::imshow("froreground", fg_masked/255);

    cv::Mat image_with_mask;
    cv::addWeighted(fg_masked, alpha, background, beta, -1, image_with_mask);

    cv::imshow("alpha blended image", image_with_mask/255);
    cv::waitKey(0);
    */

  }

  auto end4 = std::chrono::high_resolution_clock::now();
  auto duration4 =
      std::chrono::duration_cast<std::chrono::microseconds>(end4-start4);
  std::cout << " parse result : " << duration4.count() / 1.e6 << " s" << std::endl;

  std::cout << "Detect end.";
}
