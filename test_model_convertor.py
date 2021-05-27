import unittest
import os
import numpy as np
import torchvision.models as models
import torch

from model_convertor import ModelConvertor

dir_path = os.path.split(os.path.realpath(__file__))[0]
modelname = 'mobilenet_v2'

def load_test_model():
    model = None
    # to view all available models in torchvision, go to https://pytorch.org/vision/stable/models.html
    # Classification
    if modelname == "resnet18":
        model = models.resnet18(pretrained=True)
    elif modelname == "alexnet":
        model = models.alexnet(pretrained=True)
    elif modelname == "squeezenet1_0":        
        model = models.squeezenet1_0(pretrained=True)
    elif modelname == "vgg16":        
        model = models.vgg16(pretrained=True)
    elif modelname == "densenet161":        
        model = models.densenet161(pretrained=True)
    elif modelname == "inception_v3":
        model = models.inception_v3(pretrained=True)
    elif modelname == "googlenet":
        model = models.googlenet(pretrained=True)
    elif modelname == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=True)
    elif modelname == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    elif modelname == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
    elif modelname == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=True)
    elif modelname == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=True)
    elif modelname == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=True)
    elif modelname == "mnasnet1_0":
        model = models.mnasnet1_0(pretrained=True)
    # Semantic Segmentation
    elif modelname == "fcn_resnet50":
        model = models.segmentation.fcn_resnet50(pretrained=True)
    elif modelname == "fcn_resnet101":
        model = models.segmentation.fcn_resnet101(pretrained=True)
    elif modelname == "deeplabv3_resnet50":
        model == models.segmentation.deeplabv3_resnet50(pretrained=True)
    elif modelname == "deeplabv3_resnet101":
        model == models.segmentation.deeplabv3_resnet101(pretrained=True)
    elif modelname == "deeplabv3_mobilenet_v3_large":
        model == models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    elif modelname == "lraspp_mobilenet_v3_large":
        model == models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)
    # detection
    elif modelname == "fasterrcnn_resnet50_fpn":
        model == models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif modelname == "fasterrcnn_mobilenet_v3_large_fpn":
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    elif modelname == "fasterrcnn_mobilenet_v3_large_320_fpn":
        model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    elif modelname == "retinanet_resnet50_fpn":
        model = models.detection.retinanet_resnet50_fpn(pretrained=True)
    elif modelname == "maskrcnn_resnet50_fpn":
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif modelname == "keypointrcnn_resnet50_fpn":
        model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    # Video classification
    elif modelname == "r3d_18":
        model = models.video.r3d_18(pretrained=True)
    elif modelname == "mc3_18":
        model = models.video.mc3_18(pretrained=True)
    elif modelname == "r2plus1d_18":
        model = models.video.r2plus1d_18(pretrained=True)

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
            precision='fp32'
        )

        # predict
        for i in range(0, 10):
            print("batch {}".format(i))
            batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
            out = convertor.predict(batch_in)

    # def test_fp32_predict(self):
    #     convertor = ModelConvertor()
    #     # load engine
    #     bsize = 8
    #     bsizes = convertor.load_engine(os.path.join(dir_path, "data/models/{}_fp32_bsize{}.trt".format(modelname, bsize)))
    #     self.assertEqual(bsizes, [bsize])
    #     # predict
    #     nbatch = 1000
    #     for i in range(0, nbatch):
    #         print("batch {}".format(i))
    #         batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
    #         out = convertor.predict(batch_in)

    # def test_fp16_convert_and_predict(self):
    #     convertor = ModelConvertor()
    #     # convert model
    #     model = load_test_model()
    #     bsize = 8
    #     dummy_input=torch.randn(bsize, 3, 224, 224)
    #     engine_path = convertor.load_model(
    #         model,
    #         dummy_input,
    #         onnx_model_path=os.path.join(dir_path, "data/models/{}_fp16_bsize{}.onnx".format(modelname, bsize)),
    #         engine_path=os.path.join(dir_path, "data/models/{}_fp16_bsize{}.trt".format(modelname, bsize)),
    #         explicit_batch=True,
    #         precision='fp16'
    #     )
   
    #     # predict
    #     for i in range(0, 10):
    #         print("batch {}".format(i))
    #         batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
    #         out = convertor.predict(batch_in)

    # def test_fp16_predict(self):
    #     convertor = ModelConvertor()
    #     # load engine
    #     bsize = 8
    #     bsizes = convertor.load_engine(os.path.join(dir_path, "data/models/{}_fp16_bsize{}.trt".format(modelname, bsize)))
    #     self.assertEqual(bsizes, [bsize])
    #     # predict
    #     nbatch = 1000
    #     for i in range(0, nbatch):
    #         print("batch {}".format(i))
    #         batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
    #         out = convertor.predict(batch_in)            

    # def test_int8_convert_and_predict(self):
    #     convertor = ModelConvertor()
    #     # convert model
    #     model = load_test_model()
    #     bsize = 8
    #     dummy_input=torch.randn(bsize, 3, 224, 224)
    #     engine_path = convertor.load_model(
    #         model,
    #         dummy_input,
    #         onnx_model_path=os.path.join(dir_path, "data/models/{}_int8_bsize{}.onnx".format(modelname, bsize)),
    #         engine_path=os.path.join(dir_path, "data/models/{}_int8_bsize{}.trt".format(modelname, bsize)),
    #         explicit_batch=True,
    #         precision='int8',
    #         max_calibration_size=300,
    #         calibration_batch_size=32,
    #         calibration_data=os.path.join(dir_path, "data/images/imagenet100"),
    #         preprocess_func='preprocess_imagenet'
    #     )
   
    #     # predict
    #     for i in range(0, 10):
    #         print("batch {}".format(i))
    #         batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
    #         out = convertor.predict(batch_in)

    # def test_int8_predict(self):
    #     convertor = ModelConvertor()
    #     # load engine
    #     bsize = 8
    #     bsizes = convertor.load_engine(os.path.join(dir_path, "data/models/{}_int8_bsize{}.trt".format(modelname, bsize)))
    #     self.assertEqual(bsizes, [bsize])
    #     # predict
    #     nbatch = 1000
    #     for i in range(0, nbatch):
    #         print("batch {}".format(i))
    #         batch_in = np.random.random((bsize, 3, 224, 224)).astype(np.float32)
    #         out = convertor.predict(batch_in)


if __name__ == '__main__':
    unittest.main()
