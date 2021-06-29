#include "NvInfer.h"
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <fstream>
#include "opencv2/opencv.hpp"

using namespace std;
#define CHECK(x) (x)

namespace nvinfer1 {
 class int8EntroyCalibrator : public nvinfer1::IInt8EntropyCalibrator {
 public:
    int8EntroyCalibrator(const int &bacthSize,
    const std::string &imgPath,
    const std::string &calibTablePath);

    virtual ~int8EntroyCalibrator();

    int getBatchSize() const override { return batchSize; }

    bool getBatch(void *bindings[], const char *names[], int nbBindings) override;

    const void *readCalibrationCache(std::size_t &length) override;

    void writeCalibrationCache(const void *ptr, std::size_t length) override;

  private:

    bool forwardFace;

    int batchSize;
    size_t inputCount;
    size_t imageIndex;

    std::string calibTablePath;
    std::vector<std::string> imgPaths;

    float *batchData{ nullptr };
    void  *deviceInput{ nullptr };



    bool readCache;
    std::vector<char> calibrationCache;
 };

 int8EntroyCalibrator::int8EntroyCalibrator(const int &bacthSize, const std::string &imgPath,
  const std::string &calibTablePath) :batchSize(bacthSize), calibTablePath(calibTablePath), imageIndex(0), forwardFace(
   false) {
    int inputChannel = 3;
    int inputH = 416;
    int inputW = 416;
    inputCount = bacthSize*inputChannel*inputH*inputW;
    std::fstream f(imgPath);
    if (f.is_open()) {
    std::string temp;
    while (std::getline(f, temp)) imgPaths.push_back(temp);
    }
    int len = imgPaths.size();
    for (int i = 0; i < len; i++) {
    cout << imgPaths[i] << endl;
    }
    batchData = new float[inputCount];
    CHECK(cudaMalloc(&deviceInput, inputCount * sizeof(float)));
 }

 int8EntroyCalibrator::~int8EntroyCalibrator() {
    CHECK(cudaFree(deviceInput));
    if (batchData)
    delete[] batchData;
 }

 bool int8EntroyCalibrator::getBatch(void **bindings, const char **names, int nbBindings) {
      cout << imageIndex << " " << batchSize << endl;
      cout << imgPaths.size() << endl;
      if (imageIndex + batchSize > int(imgPaths.size()))
      return false;
      // load batch
      float* ptr = batchData;
      for (size_t j = imageIndex; j < imageIndex + batchSize; ++j)
      {
      //cout << imgPaths[j] << endl;
      cv::Mat img = cv::imread(imgPaths[j]);
      vector<float>inputData = prepareImage(img);
      //  float* matData = (float*)myMat.data
      cout << inputData.size() << endl;
      cout << inputCount << endl;
      if ((int)(inputData.size()) != inputCount)
      {
        std::cout << "InputSize error. check include/ctdetConfig.h" << std::endl;
        return false;
      }
      assert(inputData.size() == inputCount);
      int len = (int)(inputData.size());
      memcpy(ptr, inputData.data(), len * sizeof(float));

      ptr += inputData.size();
      std::cout << "load image " << imgPaths[j] << "  " << (j + 1)*100. / imgPaths.size() << "%" << std::endl;
      }
      imageIndex += batchSize;
      CHECK(cudaMemcpy(deviceInput, batchData, inputCount * sizeof(float), cudaMemcpyHostToDevice));
      bindings[0] = deviceInput;
      return true;
   }

 const void* int8EntroyCalibrator::readCalibrationCache(std::size_t &length)
 {
    calibrationCache.clear();
    std::ifstream input(calibTablePath, std::ios::binary);
    input >> std::noskipws;
    if (readCache && input.good())
    std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
      std::back_inserter(calibrationCache));

    length = calibrationCache.size();
    return length ? &calibrationCache[0] : nullptr;
 }

 void int8EntroyCalibrator::writeCalibrationCache(const void *cache, std::size_t length)
 {
    std::ofstream output(calibTablePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
 }
}