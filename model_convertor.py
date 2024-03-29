import os
import onnx
import torch # needed to get version and cuda setup
import torchvision.models as models
import torch.onnx
import tensorrt as trt
import pycuda.driver as cuda
from typing import Tuple, List
import numpy as np
import pycuda.autoinit
import ctypes
from onnx_to_tensorrt_api import onnx_to_tensorrt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def numpy_dtype_from_trt(dtype):
    if dtype == trt.bool:
        return bool
    elif dtype == trt.int8:
        return np.int8
    elif dtype == trt.int32:
        return np.int32
    elif dtype == trt.float16:
        return np.float16
    elif dtype == trt.float32:
        return np.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)

def load_engine(filename: str):
    # Load serialized engine file into memory
    with open(filename, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def pytorch_to_onnx(pytorch_model, dummy_input, onnx_model, verbose=True):
    torch.onnx.export(pytorch_model, dummy_input, onnx_model, verbose=verbose, opset_version=11)
    print("finished to export to onnx file {}".format(onnx_model))

def get_binding_idxs(engine: trt.ICudaEngine, prof_idx: int):
    # Calculate start/end binding indices for current context's profile
    n_bind_per_prof = engine.num_bindings // engine.num_optimization_profiles
    start_bind_idx = prof_idx * n_bind_per_prof
    end_bind_idx = start_bind_idx + n_bind_per_prof

    # print("Engine/Binding Metadata")
    # print("\tNumber of optimization profiles: {}".format(engine.num_optimization_profiles))
    # print("\tNumber of bindings per profile: {}".format(n_bind_per_prof))
    # print("\tFirst binding for profile {}: {}".format(prof_idx, start_bind_idx))
    # print("\tLast binding for profile {}: {}".format(prof_idx, end_bind_idx-1))

    # Separate input and output binding indices for convenience
    in_bind_idxs = []
    out_bind_idxs = []
    for bind_idx in range(start_bind_idx, end_bind_idx):
        if engine.binding_is_input(bind_idx):
            in_bind_idxs.append(bind_idx)
        else:
            out_bind_idxs.append(bind_idx)

    return in_bind_idxs, out_bind_idxs

def is_dynamic(shape: Tuple[int]):
    return any(dim is None or dim < 0 for dim in shape)

def setup_binding_shapes(
    engine: trt.ICudaEngine,
    ctx: trt.IExecutionContext,
    h_inputs: List[np.ndarray],
    in_bind_idxs: List[int],
    out_bind_idxs: List[int],
):
    # Explicitly set the dynamic input shapes, so the dynamic output
    # shapes can be computed internally
    for host_input, bind_idx in zip(h_inputs, in_bind_idxs):
        ctx.set_binding_shape(bind_idx, host_input.shape)

    assert ctx.all_binding_shapes_specified

    h_outputs = []
    d_outputs = []
    for bind_idx in out_bind_idxs:
        out_shape = ctx.get_binding_shape(bind_idx)
        # Allocate buffers to hold output results after copying back to host
        dtype=numpy_dtype_from_trt(ctx.get_binding_dtype(bind_idx))
        buf = np.empty(out_shape, dtype=dtype)
        h_outputs.append(buf)
        # Allocate output buffers on device
        d_outputs.append(cuda.mem_alloc(buf.nbytes))

    return h_outputs, d_outputs    
    

class ModelConvertor(object):
    def __init__(self, reuse_last_gpu=True):
        self.reuse_last_gpu = reuse_last_gpu
        self.engine_path = None
        self.engine = None
        self.ctxs = {} # a dict of batchsize-context
        self.iprofs = {}
        self.prev_batchsize = 0
        self.h_inputs = None
        self.h_outputs = None        
        self.d_inputs = None
        self.d_outputs = None

    def get_batchsizes(self):
        return list(self.ctxs.keys())

    def load_engine(self, engine_path):
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        engine = load_engine(engine_path)
        if engine is None:
            print("error when load engine {}".format(engine_path))
            return None
        nprof = engine.num_optimization_profiles
        ctxs = {}
        iprofs = {}
        for iprof in range(nprof):
            ctx = engine.create_execution_context()
            ctx.active_optimization_profile = iprof
            in_shape = ctx.get_binding_shape(0)
            if is_dynamic(in_shape): # TODO, need check
                ctxs[1] = ctx
                iprofs[1] = iprof
                break
            else:
                ctxs[in_shape[0]] = ctx
                iprofs[in_shape[0]] = iprof

        for batchsize, iprof in iprofs.items():
            print("Profile {}: batchsize={}".format(iprof, batchsize))

        self.engine_path = engine_path
        self.engine = engine
        self.ctxs = ctxs
        self.iprofs = iprofs
        print("finished to load TensorRT engine {}".format(engine_path))
        return self.get_batchsizes()

    def load_model(self,
        modelOrPath,  # can be pytorch model, onnx model path, trt model path
        dummy_input:torch.Tensor,
        onnx_model_path:str,
        engine_path:str,
        explicit_batch=True,
        precision='int8',
        max_calibration_size=300,
        calibration_batch_size=32,
        calibration_data=None,
        preprocess_func='preprocess_imagenet', # TODO, select automatic
        use_cache_if_exists=False,
        save_cache_if_exists=False,
        cali_input_shape=(224, 224),
        verbosity="info" # "error" , "info" , "verbose"
        ):
        
        if precision not in ['int8', 'fp32', 'fp16']:
            print("error: precision must be int8, fp32 or fp16")
            return None

         # trt engine path
        if isinstance(modelOrPath, str) and len(modelOrPath.split('.'))>1 and (modelOrPath.split('.')[-1] in ['trt', 'plan']):
            self.engine_path = modelOrPath
        else:
            # onnx_model_path
            if isinstance(modelOrPath, str) and len(modelOrPath.split('.'))>1 and modelOrPath.split('.')[-1] == 'onnx':
                onnx_model_path = modelOrPath
            # pytorch model
            else: 
                if not ( (isinstance(dummy_input, torch.Tensor) and len(dummy_input.shape)==4) or isinstance(dummy_input, List) ):
                    print("error: dummy_input must be torch.Tensor of [bsize, c, w, h] or list of torch.Tensor of [c, w, h]")
                    return None
                if not isinstance(onnx_model_path, str) or len(onnx_model_path.strip()) == 0:
                    print("error: onnx_model_path must be a non-empty string")
                pytorch_to_onnx(modelOrPath, dummy_input, onnx_model_path, verbose=True)
        
            # conver onnx model to tensorrt engine
            if precision == 'int8':
                onnx_to_tensorrt(onnx_model_path, \
                    output=engine_path, \
                    int8=(True if precision=='int8' else False), \
                    fp16=(True if precision=='fp16' else False), \
                    max_calibration_size=max_calibration_size, \
                    calibration_batch_size=calibration_batch_size, \
                    calibration_data=calibration_data, \
                    preprocess_func=preprocess_func, \
                    explicit_batch=explicit_batch, \
                    use_cache_if_exists=use_cache_if_exists,
                    save_cache_if_exists=save_cache_if_exists,
                    cali_input_shape=cali_input_shape,
                    verbosity=None if verbosity=="err" else (1 if verbosity =="info" else 2))
            else:
                onnx_to_tensorrt(onnx_model_path, \
                output=engine_path, \
                int8=(True if precision=='int8' else False), \
                fp16=(True if precision=='fp16' else False), \
                explicit_batch=explicit_batch,
                cali_input_shape=cali_input_shape,
                verbosity=None if verbosity=="err" else (1 if verbosity =="info" else 2))
                        
            self.engine_path = engine_path

        # load engine
        self.load_engine(self.engine_path)

        return engine_path

    def predict(self, feed):
        h_inputs = None
        if isinstance(feed, List):
            batchsize = feed[0].shape[0] # all input nodes has same batchsize
            h_inputs = feed
        else:
            batchsize = feed.shape[0]
            h_inputs = [feed]

        if not batchsize in self.iprofs.keys():
            print("error: batch size must one of {}".format(self.get_batchsizes()))
            return None
        iprof = self.iprofs[batchsize]
        ctx = self.ctxs[batchsize]

        in_bind_idxs, out_bind_idxs = get_binding_idxs(self.engine, iprof)
        d_inputs = [cuda.mem_alloc(h.nbytes) for h in h_inputs]
        for h, d in zip(h_inputs, d_inputs):
            cuda.memcpy_htod(d, h)
        h_outputs, d_outputs = setup_binding_shapes(self.engine, ctx, h_inputs, in_bind_idxs, out_bind_idxs)
        
        print("\tInput shapes: {}".format([i.shape for i in h_inputs]))
        print("\tOutput shapes: {}".format([o.shape for o in h_outputs]))

        # Inference
        bindings = d_inputs + d_outputs
        ctx.execute_v2(bindings)

        for h, d in zip(h_outputs, d_outputs):
            cuda.memcpy_dtoh(h, d)

        self.prev_batchsize = batchsize

        return h_outputs


def main():
        convertor = ModelConvertor()
        # convert model
        model = models.resnext50_32x4d(pretrained=True)
        bsize = 8
        dummy_input=torch.randn(bsize, 3, 224, 224)
        saved_modelname = 'resnet50'
        dir_path = os.path.split(os.path.realpath(__file__))[0]
        engine_path = convertor.load_model(
            model,
            dummy_input,
            onnx_model_path=os.path.join(dir_path, "data/models/{}_int8_bsize{}.onnx".format(saved_modelname, bsize)),
            engine_path=os.path.join(dir_path, "data/models/{}_int8_bsize{}.trt".format(saved_modelname, bsize)),
            explicit_batch=True,
            precision='int8',
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

if __name__ == "__main__":
    main()