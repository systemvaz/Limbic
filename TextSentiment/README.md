# Positive/Negative Text sentiment detection
## About
Detects the sentiment of user inputted text as either Positive or Negative
![alt text](https://github.com/systemvaz/Limbic/blob/master/TextSentiment/img/demo.PNG)

## Training dataset
The Sentiment140 dataset was utilised containing 1.6 million tweets from Twitter and associated annotated sentiment either being Positive or Negative.
## Architecture
Experimentation with tokenisation text preprocessing, including the utilisation of BERT tokenisation and fine tuning of BERT TensorHub model with an additional dense layer at output.
## Usage
* 1.) Download Sentiment140 dataset and place in folder Limbic/TextSentiment/data
* 2.) Run preprocess_data.py to clean dataset
* 3.) Run bert_preprocess.py to generate h5 file with required input ids + masks and segment ids
* 4.) Optionaly run check_bert_data.py to confirm shape and data generated in step 3
* 5.) Run train_BERT.py to perform model training. ~9.5 hours per epoch on a GTX1080ti (only 1 epoch required for 84.5% accuracy)
* 6.) Run main detection program detect.py
## Training Results
![alt text](https://github.com/systemvaz/Limbic/blob/master/TextSentiment/img/Bert_training_results.png)