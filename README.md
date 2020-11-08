# nlp_assessment

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
* Medical term word comparisons (based upon medical words / sentence, and medical words / label or classification)
* Neighbor sentence (sentence before / sentence after) having the same classification of the sentence at hand
