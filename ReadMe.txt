World Prediction NLP Readme File
 
Author:
	Kerollos Lowandy

Environment
	OS: Windows
	Programming Language: Python 3.8.2

*Project Adapted from DeepLearning.ai TensorFlow Developer Certificate Lab*


Description: 
	This NLP neural network predicts the next word in a sequence of words, given at 
	least one previous seed word.


Instructions

Files Needed in Common Directory:
	- functions.py
	- load_model_and_predict.py
	- word_predictor.py
	- (A .txt file to be used as the corpus. Adjust the "FILE" variable within 	 
          load_model_and_predict.py and word_predictor_with_chars.py accordingly)

Steps:
	1. Run word_predictor.py to generate a new model. Edit the "RAND_SAMPLES" variable 
	   to choose how many data items to train on. (This number will largely depend on 
	   processing resources)
        2. To generate predictions with either the new model generated or the already 
	   generated model, run load_model_and_predict.py. When prompted, enter the seed text
	   to begin generating predictions with.

	IDE: PyCharm
	Packages Used: 
			tensorflow (2.15.0)
			matplotlib (3.8.2)
			numpy (1.26.2)
			random (built into Python 3.8.2)
			