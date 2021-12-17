Scriptit `trtexec` ja `tegrastats` työkalujen tulosten analysointiin.

Käytetyt komennot analysointitiedostojen luomiseen:  
`tegrastats --interval 5000 --logfile <filename>.txt --start`  
`trtexec --onnx=<modelfile>.onnx --workspace=<megabytes> --exportTimes=<filename>.json --iterations=1000 (--fp16)`

Käyttö:
------

	python analyze_RAM <file> 
	-> OUTPUT: RAM-käytön MIN ja MAX arvot sekä kuvaaja (matplotlib png-tiedosto)
	
	python analyze_trtexec_times <file> 
	-> OUTPUT: MIN ja AVG suoritusajat sekä vastaavat FPS lukemat
	