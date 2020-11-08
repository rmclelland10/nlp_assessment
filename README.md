# nlp assessment

# Set-up
* git clone https://github.com/rmclelland10/nlp_assessment.git
* change directory to nlp_assessment
* pip install -r requirements.txt
* Open up a python shell, and type:
  * \>>> import nltk
  * \>>> nltk.download('punkt')

# PART-1
HOW TO RUN:
* python/python3 oa_p1.py -task=1  ---> Runs sorted display on exercise_text.txt
* python/python3 oa_p1.py -file=sample.txt -task=1  ---> Runs sorted display on a different file
* python/python3 oa_p1.py -task=2 -search=patient  ---> Displays count of 'patient' occurences in exercise_text.txt
* python/python3 oa_p1.py -file=sample.txt -task=2 -search=patient  ---> Displays count of 'patient' occurences in exercise_text.txt

# PART-2
LIMITATIONS:
* Part 2 runs on a single text file, in which I manually labeled each sentence to what I believe its category would be. This brings two main limitations: inadequate sample size and inefficient labeling. To overcome these challenges, I would most likely have some sort of document library, where I could loop through each file, extract the necessary sentences, and then run through all the sentences at once for the model. To get around the manual labeling, I'd gather files which are entirely homogenuous in regards to the category its sentences corresponded to (for example, a lab report document or a Imaging report), so that the classfication of each sentence could be automated per document. This would, however, essentially invalidate the neighbor-feature (see below), so there would just have to be some other utility that would enable classifying sentences at a faster pace.

PRE-PROCESSING (For each sentence):
* Remove new lines, punctuation (according to the requirements), convert all words to lowercase
* Run each word in sentence through a stemmer
* Remove stop-words from each sentence

WORD-BASED FEATURES:
* Bag of Words (BoW) model --> vocabulary of unique words in text 
* Number of Medical term words / Total # of words, per sentence
* Number of Medical term words normalized across all sentences (and their labels/class), per sentence
* Neighbor sentence comparison (sentence before / sentence after) having the same classification of the sentence at hand
* Features not implemented but could potentially bring benefit:
  * Overall amount of words per sentence
  * Count/frequency of each type of word (noun, verb, adjective, etc.) per sentence
  * Relationships between words in a sentence to additionally be able to label a negative/positive test result to the test
  * Non-mutally exclusive classifications (a sentence can be a part of multiple categories)
  * Section headers, which could give weight to proceeding sentences and help identify them.

PARAMETERS THAT NEED SET (See main entry point in oa_p2.py)
* text_filename - file to be read
* labels - sentence labels/classification for each sentence in text file (1 - General, 2 - Pathology Results, 3 - Lab Results, 4 - Imaging Results)
* test_size - fraction of data to be used for testing
* classifier_obj_index - index of classifier object in classifiers list in main entry point
* show_plot - set to true to see confusion matrix plot
* save_model - set to true to save model as .av file

HOW TO RUN:
* Set variables in __main__ (see comments, variable description )
* python/python3 oa_p2.py
