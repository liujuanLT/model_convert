import unittest
import os
import numpy as np
import img_helper
from model_convertor import ModelConvertor

dir_path = os.path.split(os.path.realpath(__file__))[0]
verbosity = "info"

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

    def test_fp32_mmdet_SSD_loadtrt(self):
        bsize = 1
        convertor = ModelConvertor()
        # laod tensorRT model
        trt_model_path = "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000_fp32_bsize1.trt"
        convertor.load_model(trt_model_path, dummy_input=None, onnx_model_path=None, engine_path=None)

        # test using image
        single_image_path = '/home/jliu/data/images/mmdetection/demo.jpg'
        input_config = {
            'input_shape': (1, 3, 512, 512),
            'input_path': single_image_path,
            'normalize_cfg': {
                'mean': (123.675, 116.28, 103.53),
                'std': (58.395, 57.12, 57.375)
                   }
                }
        one_img, one_meta = img_helper.preprocess_example_input(input_config)
        batch_in = one_img.contiguous().detach().numpy()
        labels, boxes_and_scores = convertor.predict(batch_in)
        print("boxes_and_scores: \n{}".format(boxes_and_scores))
        print("labels: \n{}".format(labels))

        # save image
        prefix_image= os.path.split(single_image_path)[-1].split('.')[0]
        prefix_model= os.path.split(trt_model_path)[-1].split('.')[0]
        out_image_path = os.path.join('/home/jliu/data/images/test_mmdet_ssd/', prefix_model+'_input_'+prefix_image+'.jpg')
        boxes_and_scores = np.squeeze(boxes_and_scores, axis=0)
        labels = np.rint(np.squeeze(labels, axis=0)).astype(np.int32)
        img_helper.imshow_det_bboxes(
            one_meta['show_img'],
            boxes_and_scores,
            labels,
            class_names=img_helper.coco_classes(),
            score_thr=0.3,
            bbox_color='red',
            text_color='red',
            thickness=1,
            font_size=4,
            win_name="tensorrt",
            out_file=out_image_path)
        print("done")
        

if __name__ == '__main__':
    unittest.main()
