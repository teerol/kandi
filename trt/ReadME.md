**Integroitu osaksi testipenkkiä**

`convert_trt.py` optimoi onnx -mallin TensorRT moottoriksi.   
`test_trt_engine.py` tarjoaa TensorRT -mallin suorituskyvy analysointiin tarvittavat työkalut.  
`trt_inference.py` tarjoaa aputyökalut TensorRT -mallin suorituskyvy analysointiin. *(Pycuda, TensorRT Python API)*

Käyttö
----
`python convert_trt <onnnx_model_file> <trt_model_path> <tarkkuus (FP32 tai FP16)>`  
`trt_model_path` -parametri määrittää mihin uusi malli tallennetaan.  
*Jetson Nano ei tue INT8 kvantifiontia.*  
*Muunnosparametrejä voi kontrolloida tiedostosta.*  
*Maksimieräkoko (batch size) onnx-mallin rajotteista johtuen 1*  

Työssä suoritusnopeuden mittaukset suoritettiin `trtexec` komennolla  
Käytetty komento:  
`trtexec --onnx=<model_file>.onnx --workspace=<megabytes> --exportTimes=<file>.json --iterations=1000 (--fp16)`

	