
#include <iostream>
#include "detector.h"

using namespace std;
using namespace tensorflow;

#include <dirent.h>
void read_directory(const std::string& name, std::vector<std::string>& v, std::function<bool(std::string)> check)
{
  DIR* dirp = opendir(name.c_str());
  struct dirent * dp;
  while ((dp = readdir(dirp)) != NULL) {
    if(!check(dp->d_name))
    {
      continue;
    }

    v.push_back(dp->d_name);
  }
  closedir(dirp);
}

inline bool startsWith(std::string mainStr, std::string toMatch)
{
  // std::string::find returns 0 if toMatch is found at starting
  return mainStr.find(toMatch) == 0? true : false;
}

inline int endsWith(string s,string sub){
  return s.rfind(sub)==(s.length()-sub.length())? true:false;
}

int main(int argc, char* argv[]) {
  std::cout << argv[0] << " image_path model_path(*.pb)";

  string image_path(argv[1]);
  string model_path(argv[2]);
  string labels = "data/mscoco_label_map.pbtxt";

  const string& graph_path = model_path;
  //const string& graph_path = tensorflow::io::JoinPath(root_dir, graph);
  //const string& image_path = tensorflow::io::JoinPath(root_dir, image_name);
  LOG(ERROR) << "graph_path:" << graph_path;
  LOG(ERROR) << "image_path:" << image_path;

  auto detector = std::make_shared<detector::Detector>(graph_path, "");

  std::vector<std::string> allimages_l;
  read_directory(image_path, allimages_l, [](std::string name){
      return endsWith(name, ".png") || endsWith(name, ".jpg");
  });

  for (auto it_l = allimages_l.begin(); it_l < allimages_l.end(); it_l += 1) {

    bool file_matched = false;
    std::string left_image_name;
    std::string left_image_path;
    left_image_name = (*it_l);
    left_image_path = image_path + "/" + left_image_name;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<detector::ROI> rois;
    detector->Detect(left_image_path, rois);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    std::cout << " total time: " << duration.count() / 1.e6 << " s" << std::endl;

    //for (int i = 0; i < rois.size(); ++i) {
    //  std::cout << rois[i];
    //}
  }

  return 0;
}

