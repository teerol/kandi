Code used in my Bsc. thesis about Neural Networks in Embedded Systems.  
This repo contais the testbench for different running environments and analysing tools.  
Thesis can be found from Trepo: **<ADD LINK!>**
The rest of the ReadMe is in finnish like the thesis.  

Testit suoritettiin Nvidia **Jetson Nano** järjestelmällä ja **JetPack 4.6** versiolla  
Käytetty data-aineisto: `tensorflow_datasets imagenet_v2`  
Käytetyt kuvantunnistusneuroverkkomallit: `tensorflow.keras.applications.`   

	MobileNet (alpha= 0.25, 0.5, 0.75, 1.0) 
	MobileNet_v2 (alpha= 0.5, 1.0) 
	ResNet50
	DenseNet121
Käyttö
-----

Vaatimukset:

    python 3.6.9
	tensorflow 2.5.0
	tensorflow_datasets

Vapaavalintaiset (vain osassa testejä tarvittavat):

	onnx, onnx_runtime, pycuda, tensorrt, matplotlib
	
	
Alikansiot sisätävät eri optimointimenetelmiä käyttäviä tiedostoja sekä analysointityökaluja  
Katso lisätiedot kansioiden ReadME-tiedostoista.
---
	
**TESTIPENKIN KÄYTTÖ:**  

	usage: test_bench_2.py [-h] [-e] [-p rounds] [-ds DATA_SPLIT] [-bs BATCH_SIZE] [-m MODEL]

	Test bench for Neural Network models and platforms

	optional arguments:
	  -h, --help            show this help message and exit
	  -e, --eval            Evaluate model with data
	  -p rounds, --perf rounds
							Make performance tests (how many times to test the data)
	  -ds DATA_SPLIT, --data_split DATA_SPLIT
							used data split in tfds format e.x. 0:10 50% 500: etc.
	  -bs BATCH_SIZE, --batch_size BATCH_SIZE
							used batch_size during evaluation and perftests
	  -m MODEL, --model MODEL
							model used in tests can be TensorFlow savedmodel or trt_engine
							
Esimerkki:  
`python test_bench_2 -m models/ResNet50/ -e -p 20 -ds 0:320 - bs 32`  
- Käyttää polkuun `models/ResNet50/` tallennettua mallia tai  
lataa uuden mallin `make_model` -funktiossa määritellystä osoitteesta ja tallentaa annettuun sen polkuun.
- Suorittaa tarkkuuden testin käyttäen 320 ensimmäistä kuvaa datasta ja eräkokoa 32
- Suorittaa päättelynopeuden testin, jossa 320 ensimmäistä kuvaa datasta luokitellaan  
20 kertaa käyttäen eräkokoa 32.
---

**onnx-mallien mittauksia ei ole integroitu testipenkkiin.**  
- `run_onnx.py` kykenee vain tarkkuuden mittaamiseen
- Tiedostosta voi määrittää käytetyn datan määrän
- Käyttö: `python run_onnx <onnx_model_file>`


							