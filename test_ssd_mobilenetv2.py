import unittest
import os
import numpy as np
import time
import img_helper
import data_helper
from model_convertor import ModelConvertor

dir_path = os.path.split(os.path.realpath(__file__))[0]
verbosity = "info"
nloop = 1 # 10000

class TestSsdMobilenetV2(unittest.TestCase):
 
    def test_SSD_mobilenetv2_int8withcali(self):
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/home/jliu/data/models/ssd_mobilenetv2_512x512/ssd_mobilenet_v2_dynamicbatch.onnx"
        relfn = os.path.split(onnx_model)[-1]
        prefix_model= relfn[:relfn.rfind('.')]
        trt_model_path = os.path.join(dir_path, f"data/models/{prefix_model}_int8cali_calisize512_n500.trt")
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

    
    def test_SSD_mobilenetv2_trtinfer(self):
        convertor = ModelConvertor()
        img_readpath = '/home/jliu/data/coco/images/val2017/'
        img_savepath = os.path.join(dir_path, 'data/images_inferd/ssd_mobilenetv2_512x512/')
        # laod tensorRT model
        for trt_model_path in [ \
            "/home/jliu/data/models/ssd_mobilenetv2_512x512/ssd_mobilenet_v2_batch1_int8.trt",
            "/home/jliu/data/models/ssd_mobilenetv2_512x512/ssd_mobilenet_v2_batch1_fp16.trt",
            "/home/jliu/data/models/ssd_mobilenetv2_512x512/ssd_mobilenet_v2_batch1_fp32.trt",
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
                    num_detections,nmsed_boxes,nmsed_scores,nmsed_classes = convertor.predict(batch_in)
                etime = time.time()
                labels, boxes_and_scores = data_helper.process_nmstrt_result(num_detections,nmsed_boxes,nmsed_scores,nmsed_classes)
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
    # TestSsdMobilenetV2().test_SSD_mobilenetv2_int8withcali()
    # TestSsdMobilenetV2().test_SSD_mobilenetv2_trtinfer()
    unittest.main()