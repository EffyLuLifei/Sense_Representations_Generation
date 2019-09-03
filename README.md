# Sense_Representations_Generation
Semantic Specialization of Word Representation for Individual Word Senses


# Brief Introduction 


## Note:

1. All the essential data sets can be found from the internet.

2. The first two steps are implemented in Python 3, and the third step is implemented in Python 2.7

3. When run the retrofitting model, please filter all "# If" and manually change the setting following the comments.

## Initial Datasets

1. WordNet https://wordnet.princeton.edu/

2. fastText https://fasttext.cc/docs/en/english-vectors.html

3. word2vec https://code.google.com/archive/p/word2vec/


## Implementation:

Step 1. The Constraints Extraction need the _WordNet_ as in put, and the following are three outputs.

	      synpairs.txt
	      antpairs.txt
	      hyppairs.txt

Step 2. The Distributional Word Vector Preparation need the _fastText_ and _word2vec_ pre-trained word representation
        respectivelyfor two scripts, but both of the them need the following output from Step 1.
	
	      synpairs.txt
	      antpairs.txt
	
The outputs are in their own folders.
	 
	      synonym.txt
	      antonym.txt
	      sense_dictionary.txt (This datset is huge and cannot be uploaded)

Step 3. The retrofitting model should be run separately.
	Please copy the evaluation folder, the _fastText_ or _word2vec_ pre-trained word representation, and the
	following output from Step 2 into the folder 3_Retrofitting_Model/fastText or 3_Retrofitting_Model/word2vec.
	
	      synonym.txt
	      antonym.txt
	      sense_dictionary.txt

The evaluation results will be printed and the following are the output in the result folder.

	      final_vectors.txt



