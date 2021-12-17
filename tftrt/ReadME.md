**Integroitu osaksi testipenkkiä**

`optimize_TF_TRT.py` optimoi tensorflow:n saved model -mallin TensorRT:n moottoreiksi.   
`tftrt_test.py` tarjoaa tf-trt mallin suorituskyvy analysointiin tarvittavat työkalut.

Käyttö
----
`python optimize_TF_TRT <model_file> <tf_trt_model_path> <tarkkuus (FP32 tai FP16)>`  
`tf_trt_model_path` -parametri määrittää mihin uusi malli tallennetaan. 
*Jetson Nano ei tue INT8 kvantifiontia.*  
*Muunnosparametrejä voi kontrolloida tiedostosta.*  
*Työn testeissä ei saavutettu parannusta suorituskykyyn `tf-trt`:tä käytettäessä.*

	