# nlp assessment

# Set-up
* git clone https://github.com/rmclelland10/nlp_assessment.git
* change directory to nlp_assessment
* pip install -r requirements.txt

# PART-1
HOW TO RUN:
* ./oa_p1.py -task=1  ---> Runs sorted display on exercise_text.txt
* ./oa_p1.py -file=sample.txt -task=1  ---> Runs sorted display on a different file
* ./oa_p1.py -task=2 -search=patient  ---> Displays count of 'patient' occurences in exercise_text.txt
* ./oa_p1.py -file=sample.txt -task=2 -search=patient  ---> Displays count of 'patient' occurences in exercise_text.txt

# PART-2
WORD-BASED FEATURES:
* Bag of Words (BoW) model
* Number of Medical term words / Total # of words, per sentence
* Number of Medical term words normalized across all sentences (and their labels/class), per sentence
* Neighbor sentence comparison (sentence before / sentence after) having the same classification of the sentence at hand

PARAMETERS THAT NEED SET (See main entry point in oa_p2.py)
* text_filename - file to be read
* labels - sentence labels/classification for each sentence in text file (1 - General, 2 - Pathology Results, 3 - Lab Results, 4 - Imaging Results)
* test_size - fraction of data to be used for testing
* classifier_obj_index - index of classifier object in classifiers list in main entry point
* show_plot - set to true to see confusion matrix plot
* save_model - set to true to save model as .av file

HOW TO RUN:
* Set variables in __main__ (see comments, variable description )
* ./oa_p2.py
