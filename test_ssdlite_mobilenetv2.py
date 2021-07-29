import unittest
import os
import numpy as np
import time
import img_helper
from model_convertor import ModelConvertor

dir_path = os.path.split(os.path.realpath(__file__))[0]
verbosity = "info"
nloop = 1 # 10000

class TestSSDlite(unittest.TestCase):
    def test_SSDlite_mobilenetv2_int8withcali(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/home/jliu/data/models/ssdlite_mobilenetv2_coco_shape320x320_trtnmsonnxsimplyreshape_topk1000_dynamicbatch.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_int8cali_cocoval_calisize320_n500_precoco_bsize{}.trt".format(prefix_model, bsize))
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
            preprocess_func='preprocess_imagenet',
            cali_input_shape=(320, 320),
            save_cache_if_exists=True
        )    

    
    def test_SSDlite_mobilenetv2_trtinfer(self):
        bsize = 1
        convertor = ModelConvertor()
        img_readpath = '/home/jliu/data/coco/images/val2017/'
        img_savepath = os.path.join(dir_path, 'data/images_inferd/ssdlite_mobilenetv2_320x320')
        # laod tensorRT model
        for trt_model_path in [ \
            # "/home/jliu/data/models/ssdlite_mobilenetv2_coco_shape320x320_trtnms_topk1000_dynamicbatch_fp32_trtexec_min1_opt16_max48.trt", \
            # "/home/jliu/data/models/ssdlite_mobilenetv2_coco_shape320x320_trtnms_dynamicbatch_int8nocali_trtexec_min1_opt16_max48.trt",
            "/home/jliu/data/models/ssdlite_mobilenetv2_320x320/ssd_mobilenet_v2_batch1_int8.trt",
            "/home/jliu/data/models/ssdlite_mobilenetv2_320x320/ssd_mobilenet_v2_batch1_fp16.trt",
            "/home/jliu/data/models/ssdlite_mobilenetv2_320x320/ssd_mobilenet_v2_batch1_fp32.trt",
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
                single_image_path = '/home/jliu/data/images/mmdetection/demo.jpg'
                # single_image_path = os.path.join(img_readpath, relimgpath)
                input_config = {
                    'input_shape': (1, 3, 320, 320),
                    'input_path': single_image_path,
                    'normalize_cfg': {
                        'mean': (123.675, 116.28, 103.53),
                        'std': (58.395, 57.12, 57.375)
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


if __name__ == '__main__':
    # TestSSDlite().test_SSDlite_mobilenetv2_int8withcali()
    # TestSSDlite().test_SSDlite_mobilenetv2_trtinfer()
    unittest.main()