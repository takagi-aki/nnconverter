# nnconverter

機械学習モデルのフォーマットをいろいろ変えるコード  

## 依存

tensorflow<=2.8  
tf2onnx  
onnx-tf  

## 問題

onnx-tfでonnx->tfするとtransposeレイヤーがやたらと挿入される。  
