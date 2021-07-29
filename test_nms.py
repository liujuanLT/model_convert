import unittest
import os
import torch
from model_convertor import ModelConvertor

dir_path = os.path.split(os.path.realpath(__file__))[0]
verbosity = "info"
nloop = 1 # 10000

class TestNMS(unittest.TestCase):
    def test_onlynms_convert(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model =  "/home/jliu/data/models/cumstom_op_nms.onnx"
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

    def test_onlynms_predict(self):
        bsize = 1
        convertor = ModelConvertor()
        # laod tensorRT model
        trt_model_path = "/home/jliu/data/models/cumstom_op_nms_fp32_bsize1.trt"
        convertor.load_model(trt_model_path, dummy_input=None, onnx_model_path=None, engine_path=None)
        # dummy input
        bsize = 1
        num_class = 80
        num_detections = 3652
        after_top_k = 200
        boxes = torch.rand(bsize, num_detections, 1, 4)
        scores = torch.rand(bsize, num_detections, num_class)
        max_output_boxes_per_class = torch.tensor([after_top_k], dtype=torch.int32)
        iou_threshold = torch.tensor([0.5], dtype=torch.float32)
        score_threshold = torch.tensor([0.02], dtype=torch.float32)      
        dummy_input = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
        # predict
        out = convertor.predict([boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold])
        print(out)


if __name__ == '__main__':
    TestNMS().test_onlynms_convert() # not OK yet
    TestNMS().test_onlynms_predict() # not OK yet

