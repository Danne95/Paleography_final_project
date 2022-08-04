
Paleography Project:
Instructions and help:

files: main.py, predict.py, paleo_transfer_learning.ipynb, consts.py, preprocess.py

libraries:
	pandas, keras, numpy, opencv, pillow, os, math, datatime, csv,
	pytz, streamlit, sklearn, matplotlib, seaborn, sys, multiprocessing,
	functools, threading, pickle.
	
	*additional libraries need to run model from BGU:
		fastai, tqdm, skimage, wandb, pytorch

Running:
	preprocess:
		command line: 'preprocess.py [input directory path] [output directory path]'
		it is assumed that the input images are named in a specific format (i.e., ashkenazi_cursive_1.jpg)
		
	transfer learning:
		paleo_transfer_learning.ipynb
		
	web application:
		command line: 'streamlit run main.py'
		
		*streamlit uses port 8501 by default, make sure it is open.
		it is also possible to choose a diffrent port by adding '--server.port=[number]' at the end of the line.
	
		*to keep the application running after closing the interface use command 'nohup' at the beginning of the line. 
		to terminate the application use 'ps -aux' to find the process id,
		then use kill -9 [PID].
		
		*it is possible to run the web application on streamlit's cloud service.
		please refer to streamlit's site for help.
	
	
	
	
	
	
	
	
	




