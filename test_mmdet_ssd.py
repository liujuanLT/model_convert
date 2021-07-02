from posix import listdir
import unittest
import os
import numpy as np
import torch
import time
import img_helper
from model_convertor import ModelConvertor

dir_path = os.path.split(os.path.realpath(__file__))[0]
verbosity = "info"
nloop = 1 # 10000

class TestModelConvertor(unittest.TestCase):
    def test_fp32_mmdet_SSD(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000.onnx"
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

    def test_fp16_mmdet_SSD(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_fp16_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='fp16',
            verbosity=verbosity
        )

    def test_int8_mmdet_SSD(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_int8cali_cocoval_calisize512_n500_precoco_bsize{}.trt".format(prefix_model, bsize))
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
            preprocess_func='preprocess_coco_mmdet_ssd',
            cali_input_shape=(512, 512),
            save_cache_if_exists=True
        )
    
    def test_mmdet_SSD_loadtrt(self):
        bsize = 1
        convertor = ModelConvertor()
        img_readpath = '/home/jliu/data/coco/images/val2017/'
        img_savepath = os.path.join(dir_path, 'data/images/test_mmdet_ssd')
        # laod tensorRT model
        for trt_model_path in [ \
            "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000_fp32_bsize1.trt", \
            "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000_fp16_bsize1.trt", \
            "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000_int8nocalibytrtexec_bsize1.trt", \
            "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000_int8cali_cocoval_calisize512_n500_precoco_bsize1.trt",
        ]:
            convertor.load_model(trt_model_path, dummy_input=None, onnx_model_path=None, engine_path=None)

            # for relpath in os.listdir('/home/jliu/data/coco/val2017'):
            for relimgpath in [
                '000000000139.jpg',
                '000000255917.jpg',
                '000000037777.jpg',
                '000000397133.jpg',
                '000000000632.jpg',
                ]:
                # single_image_path = '/home/jliu/data/images/mmdetection/demo.jpg'
                single_image_path = os.path.join(img_readpath, relimgpath)
                input_config = {
                    'input_shape': (1, 3, 512, 512),
                    'input_path': single_image_path,
                    'normalize_cfg': {
                        'mean': (123.675, 116.28, 103.53),
                        'std': (1, 1, 1)
                        }
                        }
                one_img, one_meta = img_helper.preprocess_example_input(input_config)
                batch_in = one_img.contiguous().detach().numpy()
                stime = time.time()
                for iloop in range(0, nloop):
                    labels, boxes_and_scores = convertor.predict(batch_in)
                etime = time.time()
                print("boxes_and_scores: \n{}".format(boxes_and_scores))
                print("labels: \n{}".format(labels))

                # save image
                boxes_and_scores = np.squeeze(boxes_and_scores, axis=0)
                labels = np.rint(np.squeeze(labels, axis=0)).astype(np.int32)
                for score_thr in [0.02, 0.05, 0.1, 0.2, 0.3]: # modify here
                    prefix_image= os.path.split(single_image_path)[-1].split('.')[0]
                    prefix_model= os.path.split(trt_model_path)[-1].split('.')[0]
                    out_image_path = os.path.join(img_savepath, prefix_model+'_input_'+prefix_image+"_score"+str(score_thr)+'.jpg')
                    img_helper.imshow_det_bboxes(
                        one_meta['show_img'],
                        boxes_and_scores,
                        labels,
                        class_names=img_helper.coco_classes(),
                        score_thr=score_thr,
                        bbox_color='red',
                        text_color='red',
                        thickness=1,
                        font_size=4,
                        win_name="tensorrt",
                        out_file=out_image_path)
                    print('output image to {}'.format(out_image_path))  
                    print('time to infer for {} times={:.2f}s'.format(nloop, etime-stime))       

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


    def test_fp32_mmdet_yolov3(self):
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

    def test_int8_mmdet_yolov3(self):
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
    # TestModelConvertor().test_fp32_mmdet_SSD()
    # TestModelConvertor().test_fp16_mmdet_SSD()
    TestModelConvertor().test_int8_mmdet_SSD()
    # TestModelConvertor().test_mmdet_SSD_loadtrt() 
    # TestModelConvertor().test_onlynms_convert() # not OK yet
    # TestModelConvertor().test_onlynms_predict() # not OK yet
    # TestModelConvertor().test_fp32_mmdet_yolov3()
    # TestModelConvertor().test_int8_mmdet_yolov3()
