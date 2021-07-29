import unittest
import os
from model_convertor import ModelConvertor

dir_path = os.path.split(os.path.realpath(__file__))[0]
verbosity = "info"
nloop = 1 # 10000

class TestYolov3(unittest.TestCase):
    def test_yolov3_totrt_fp32(self):
        bsize = 8
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/workspace/jupdate.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_fp32_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='fp32',
            verbosity=verbosity
        )

    def test_yolov3_totrt_int8(self):
        bsize = 8
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/workspace/jupdate.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_int8cali_cocoval_calisize320_n500_preyolov3_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='int8',
            verbosity=verbosity,
            max_calibration_size=500,
            calibration_batch_size=32,
            calibration_data='/home/jliu/data/coco/images/val2017/',
            preprocess_func='preprocess_coco_mmdet_yolov3',
            cali_input_shape=(320, 320),
            save_cache_if_exists=True
        )        


if __name__ == '__main__':
    TestYolov3().test_yolov3_totrt_fp32()
    TestYolov3().test_yolov3_totrt_int8()