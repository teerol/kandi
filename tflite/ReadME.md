EI KÄYTETTY TYÖN MITTAUKSIIN
-----

(tf-lite ei tue Nvidian GPU-arkkitehtuuria)  
Voitaisiin integroida osaksi testipenkkiä

`tflite_converter.py` kääntää mallin tensorflow:n saved model -formaatista tf-lite formaattiin.  
`tflite_test.py` tarjoaa tf-lite mallin suorituskyvy analysointiin tarvittavat työkalut.

Käyttö
----
`python tflite_converter <model_file> <tf-lite_model_path>`  

`tf-lite_model_path` -parametri määrittää mihin uusi malli tallennetaan. 
