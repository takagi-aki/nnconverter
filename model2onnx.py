import subprocess

import onnx
import tf2onnx

import tensorflow as tf

def graphdef2onnx(input:str, output:str, model_inputs, model_outputs):
    subprocess.run(f'python -m tf2onnx.convert --graphdef {input} --output {output} --inputs {model_inputs} --outputs {model_outputs}')

def ckpt2onnx(input:str, output:str, model_inputs, model_outputs):
    subprocess.run(f'python -m tf2onnx.convert --checkpoint {input} --output {output} --inputs {model_inputs} --outputs {model_outputs}')

def keras2onnx(input:str, output:str):
    subprocess.run(f'python -m tf2onnx.convert --keras {input} --output {output}')

def saved_model2onnx(input:str, output:str):
    subprocess.run(f'python -m tf2onnx.convert --saved_model {input} --output {output}')

def tflite2onnx(input:str, output:str):
    subprocess.run(f'python -m tf2onnx.convert --tflite {input} --output {output}')

def tfjs2onnx(input:str, output:str):
    subprocess.run(f'python -m tf2onnx.convert --tfjs {input} --output {output}')
