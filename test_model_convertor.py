import unittest
import os
import numpy as np
import torchvision.models as models
import torch

from model_convertor import ModelConvertor

dir_path = os.path.split(os.path.realpath(__file__))[0]
modelname = 'vgg16'

def load_test_model():
    if modelname=='resnet50':
        model = models.resnext50_32x4d(pretrained=True)
    elif modelname=='vgg16':
        model = models.vgg16(pretrained=True)
    return model


class TestModelConvertor(unittest.TestCase):
    def test_fp32_convert_and_predict(self):
        convertor = ModelConvertor()
        # convert model
        model = load_test_model()
        bsize = 8
        dummy_input=torch.randn(bsize, 3, 224, 224)
        engine_path = convertor.load_model(
            model,
            dummy_input,
            onnx_model_path=os.path.join(dir_path, "data/models/{}_fp32_bsize{}.onnx".format(modelname, bsize)),
            engine_path=os.path.join(dir_path, "data/models/{}_fp32_bsize{}.trt".format(modelname, bsize)),
            explicit_batch=True,
            presicion='fp32'
        )

        # predict
        for i in range(0, 10):
            print("batch {}".format(i))
            batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
            out = convertor.predict(batch_in)

    def test_fp32_predict(self):
        convertor = ModelConvertor()
        # load engine
        bsize = 8
        bsizes = convertor.load_engine(os.path.join(dir_path, "data/models/{}_fp32_bsize{}.trt".format(modelname, bsize)))
        self.assertEqual(bsizes, [bsize])
        # predict
        nbatch = 1000
        for i in range(0, nbatch):
            print("batch {}".format(i))
            batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
            out = convertor.predict(batch_in)

    def test_fp16_convert_and_predict(self):
        convertor = ModelConvertor()
        # convert model
        model = load_test_model()
        bsize = 8
        dummy_input=torch.randn(bsize, 3, 224, 224)
        engine_path = convertor.load_model(
            model,
            dummy_input,
            onnx_model_path=os.path.join(dir_path, "data/models/{}_fp16_bsize{}.onnx".format(modelname, bsize)),
            engine_path=os.path.join(dir_path, "data/models/{}_fp16_bsize{}.trt".format(modelname, bsize)),
            explicit_batch=True,
            presicion='fp16'
        )
   
        # predict
        for i in range(0, 10):
            print("batch {}".format(i))
            batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
            out = convertor.predict(batch_in)

    def test_fp16_predict(self):
        convertor = ModelConvertor()
        # load engine
        bsize = 8
        bsizes = convertor.load_engine(os.path.join(dir_path, "data/models/{}_fp16_bsize{}.trt".format(modelname, bsize)))
        self.assertEqual(bsizes, [bsize])
        # predict
        nbatch = 1000
        for i in range(0, nbatch):
            print("batch {}".format(i))
            batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
            out = convertor.predict(batch_in)            

    def test_int8_convert_and_predict(self):
        convertor = ModelConvertor()
        # convert model
        model = load_test_model()
        bsize = 8
        dummy_input=torch.randn(bsize, 3, 224, 224)
        engine_path = convertor.load_model(
            model,
            dummy_input,
            onnx_model_path=os.path.join(dir_path, "data/models/{}_int8_bsize{}.onnx".format(modelname, bsize)),
            engine_path=os.path.join(dir_path, "data/models/{}_int8_bsize{}.trt".format(modelname, bsize)),
            explicit_batch=True,
            presicion='int8',
            max_calibration_size=300,
            calibration_batch_size=32,
            calibration_data=os.path.join(dir_path, "data/images/imagenet100"),
            preprocess_func='preprocess_imagenet'
        )
   
        # predict
        for i in range(0, 10):
            print("batch {}".format(i))
            batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
            out = convertor.predict(batch_in)

    def test_int8_predict(self):
        convertor = ModelConvertor()
        # load engine
        bsize = 8
        bsizes = convertor.load_engine(os.path.join(dir_path, "data/models/{}_int8_bsize{}.trt".format(modelname, bsize)))
        self.assertEqual(bsizes, [bsize])
        # predict
        nbatch = 1000
        for i in range(0, nbatch):
            print("batch {}".format(i))
            batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
            out = convertor.predict(batch_in)


if __name__ == '__main__':
    unittest.main()
