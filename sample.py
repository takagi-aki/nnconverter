import model2tflite
import model2onnx

model2onnx.keras2onnx('./mobilenet.h5','./mobilenet.onnx')
#model2tflite.onnx2tflite('./yunet.onnx','./yunet.tflite')
#model2tflite.keras2tflite('./mobilenet_7.h5','./mobilenet_7.tflite')
