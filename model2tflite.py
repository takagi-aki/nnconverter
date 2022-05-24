import subprocess

import tensorflow as tf
TFLiteConverter =  tf.lite.TFLiteConverter

import onnx
from onnx_tf.backend import prepare



def onnx2tf(input: str, output: str):
    subprocess.run(f'python -m onnx-tf convert -i {input} -o {output}')


def onnx2tflite(input: str, output: str, tmp: str = "./tmp"):
    onnx_model = onnx.load(input) 
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tmp)

    converter: TFLiteConverter = TFLiteConverter.from_saved_model(tmp)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    serialized_model = converter.convert()
    with open(output, 'wb') as wf:
        if(isinstance(serialized_model,str)):
            serialized_model = serialized_model.encode('utf-8')
        wf.write(serialized_model)


def saved_model2tflite(input: str, output: str):
    converter: TFLiteConverter = TFLiteConverter.from_saved_model(tmp)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    serialized_model = converter.convert()
    with open(output, 'wb') as wf:
        if(isinstance(serialized_model,str)):
            serialized_model = serialized_model.encode('utf-8')
        wf.write(serialized_model)


def keras2tflite(input: str, output: str):
    converter: TFLiteConverter = TFLiteConverter.from_keras_model(tmp)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    serialized_model = converter.convert()
    with open(output, 'wb') as wf:
        if(isinstance(serialized_model,str)):
            serialized_model = serialized_model.encode('utf-8')
        wf.write(serialized_model)
