#include "NvInfer.h"
#include <iostream>
#include <string>
#include <vector>
#include <strstream>

using namespace std;
#define CHECK(x) (x)

// ONNX模型转为TensorRT引擎
bool onnxToTRTModel(const std::string& modelFile, // onnx文件的名字
 const std::string& filename,  // TensorRT引擎的名字 
 IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
 // 创建builder
 IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
 assert(builder != nullptr);
 nvinfer1::INetworkDefinition* network = builder->createNetwork();

 if (!builder->platformHasFastInt8()) return false;

 // 解析ONNX模型
 auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());


 //可选的 - 取消下面的注释可以查看网络中每层的详细信息
 //config->setPrintLayerInfo(true);
 //parser->reportParsingInfo();

 //判断是否成功解析ONNX模型
 if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
 {
  gLogError << "Failure while parsing ONNX file" << std::endl;
  return false;
 }

  
 // 建立推理引擎
 builder->setMaxBatchSize(BATCH_SIZE);
 builder->setMaxWorkspaceSize(1 << 30);

 nvinfer1::int8EntroyCalibrator *calibrator = nullptr;
 if (calibFile.size()>0) calibrator = new nvinfer1::int8EntroyCalibrator(BATCH_SIZE, calibFile, "F:/TensorRT-6.0.1.5/data/v3tiny/calib.table");


 //builder->setFp16Mode(true);
 std::cout << "setInt8Mode" << std::endl;
 if (!builder->platformHasFastInt8())
  std::cout << "Notice: the platform do not has fast for int8" << std::endl;
 builder->setInt8Mode(true);
 builder->setInt8Calibrator(calibrator);
 /*if (gArgs.runInInt8)
 {
  samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
 }*/
 //samplesCommon::setAllTensorScales(network, 1.0f, 1.0f);
 cout << "start building engine" << endl;
 ICudaEngine* engine = builder->buildCudaEngine(*network);
 cout << "build engine done" << endl;
 assert(engine);
 if (calibrator) {
  delete calibrator;
  calibrator = nullptr;
 }
 // 销毁模型解释器
 parser->destroy();

 // 序列化引擎
 trtModelStream = engine->serialize();

 // 保存引擎
 nvinfer1::IHostMemory* data = engine->serialize();
 std::ofstream file;
 file.open(filename, std::ios::binary | std::ios::out);
 cout << "writing engine file..." << endl;
 file.write((const char*)data->data(), data->size());
 cout << "save engine file done" << endl;
 file.close();

 // 销毁所有相关的东西
 engine->destroy();
 network->destroy();
 builder->destroy();

 return true;
}