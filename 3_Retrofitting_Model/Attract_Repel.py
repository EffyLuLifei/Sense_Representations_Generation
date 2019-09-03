import ConfigParser
import numpy
import sys
import time
import random
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy.linalg import norm
from numpy import dot
import codecs
from scipy.stats import spearmanr
import tensorflow as tf
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from gensim import models


class ExperimentRun:
    """
    This class stores all of the data and hyperparameters required for an Attract-Repel run.
    """

    def __init__(self, config_filepath):
        """
        To initialise the class, we need to supply the config file, which contains the location of
        the pretrained (distributional) word vectors, the location of (potentially more than one)
        collections of linguistic constraints (one pair per line), as well as the
        hyperparameters of the Attract-Repel procedure (as detailed in the TACL paper).
        """
        self.config = ConfigParser.RawConfigParser()
        try:
            self.config.read(config_filepath)
        except:
            print "Couldn't read config file from", config_filepath
            return None

        sense_vectors_filepath = self.config.get("data", "sense_vectors")
        original_vectors_filepath = self.config.get("data", "original_vectors")

        try:
            self.output_filepath = self.config.get("data", "output_filepath")
        except:
            self.output_filepath = "results/final_vectors.txt"

        # load initial distributional word vectors.
        sense_vectors = load_word_vectors(sense_vectors_filepath)
        if original_vectors_filepath == "GoogleNews-vectors-negative300.bin":
            self.original_vectors = models.KeyedVectors.load_word2vec_format(original_vectors_filepath, binary=True)
        else:
            self.original_vectors = load_word_vectors(original_vectors_filepath)

        if not sense_vectors:
            return
        if not self.original_vectors:
            return
        #print sense_vectors.get('contribution.n.01.contribution')
        #print "\n---- Word that cannot be found in the dictionary."
        #print "o.k..n.01.O.K.", sense_vectors.get('o.k..n.01.O.K.')
        #print "o.k..n.01.OK", sense_vectors.get('o.k..n.01.OK')
        #print "ph.d..n.01.Ph.D.", sense_vectors.get('ph.d..n.01.Ph.D.')
        #print "ph.d..n.01.PhD", sense_vectors.get('ph.d..n.01.PhD')
        #print "g.i..v.01.G.I.", sense_vectors.get('g.i..v.01.G.I.')
        #print "g.i..v.01.GI", sense_vectors.get('g.i..v.01.GI')

        # finally, load the experiment hyperparameters:
        self.load_experiment_hyperparameters()

        print "\n---------- SimLex score (Spearman's rho coefficient) of initial vectors is: \n"
        simlex_scores(sense_vectors, self.original_vectors, "begin")

        self.vocabulary = set(sense_vectors.keys())

        # this will be used to load constraints
        self.vocab_index = {}
        self.inverted_index = {}

        for idx, word in enumerate(self.vocabulary):
            self.vocab_index[word] = idx
            self.inverted_index[idx] = word

        # load list of filenames for synonyms and antonyms.
        synonym_list = self.config.get("data", "synonyms").replace("[", "").replace("]", "").replace(" ", "").split(",")
        antonym_list = self.config.get("data", "antonyms").replace("[", "").replace("]", "").replace(" ", "").split(",")

        self.synonyms = set()
        self.antonyms = set()

        if synonym_list != "":
            # and we then have all the information to load all linguistic constraints
            for syn_filepath in synonym_list:
                if syn_filepath != "":
                    self.synonyms = self.synonyms | self.load_constraints(syn_filepath)
        else:
            self.synonyms = set()

        if antonym_list != "":
            for ant_filepath in antonym_list:
                if ant_filepath != "":
                    self.antonyms = self.antonyms | self.load_constraints(ant_filepath)
        else:
            self.antonyms = set()

        self.embedding_size = random.choice(sense_vectors.values()).shape[0]
        self.vocabulary_size = len(self.vocabulary)

        # Next, prepare the matrix of initial vectors and initialise the model.

        numpy_embedding = numpy.zeros((self.vocabulary_size, self.embedding_size), dtype="float32")
        for idx in range(0, self.vocabulary_size):
            numpy_embedding[idx, :] = sense_vectors[self.inverted_index[idx]]

        # load the handles so that we can load current state of vectors from the Tensorflow embedding.
        embedding_handles = self.initialise_model(numpy_embedding)

        self.embedding_attract_left = embedding_handles[0]
        self.embedding_attract_right = embedding_handles[1]
        self.embedding_repel_left = embedding_handles[2]
        self.embedding_repel_right = embedding_handles[3]

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)

    def initialise_model(self, numpy_embedding):
        """
        Initialises the TensorFlow Attract-Repel model.
        """
        self.attract_examples = tf.placeholder(tf.int32, [None, 2])  # each element is the position of word vector.
        self.repel_examples = tf.placeholder(tf.int32, [None, 2])  # each element is again the position of word vector.

        self.negative_examples_attract = tf.placeholder(tf.int32, [None, 2])
        self.negative_examples_repel = tf.placeholder(tf.int32, [None, 2])

        self.attract_margin = tf.placeholder("float")
        self.repel_margin = tf.placeholder("float")
        self.regularisation_constant = tf.placeholder("float")

        # Initial (distributional) vectors. Needed for L2 regularisation.
        self.W_init = tf.constant(numpy_embedding, name="W_init")

        # Variable storing the updated word vectors.
        self.W_dynamic = tf.Variable(numpy_embedding, name="W_dynamic")

        # Attract Cost Function:

        # placeholders for example pairs...
        attract_examples_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.attract_examples[:, 0]),
                                                   1)
        attract_examples_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.attract_examples[:, 1]),
                                                    1)

        # and their respective negative examples:
        negative_examples_attract_left = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_attract[:, 0]), 1)
        negative_examples_attract_right = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_attract[:, 1]), 1)

        # dot product between the example pairs.
        attract_similarity_between_examples = tf.reduce_sum(tf.multiply(attract_examples_left, attract_examples_right),
                                                            1)

        # dot product of each word in the example with its negative example.
        attract_similarity_to_negatives_left = tf.reduce_sum(
            tf.multiply(attract_examples_left, negative_examples_attract_left), 1)
        attract_similarity_to_negatives_right = tf.reduce_sum(
            tf.multiply(attract_examples_right, negative_examples_attract_right), 1)

        # and the final Attract Cost Function (sans regularisation):
        self.attract_cost = tf.nn.relu(
            self.attract_margin + attract_similarity_to_negatives_left - attract_similarity_between_examples) + \
                            tf.nn.relu(
                                self.attract_margin + attract_similarity_to_negatives_right - attract_similarity_between_examples)

        # Repel Cost Function:

        # placeholders for example pairs...
        repel_examples_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.repel_examples[:, 0]),
                                                 1)  # becomes batch_size X vector_dimension
        repel_examples_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.repel_examples[:, 1]), 1)

        # and their respective negative examples:
        negative_examples_repel_left = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_repel[:, 0]), 1)
        negative_examples_repel_right = tf.nn.l2_normalize(
            tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_repel[:, 1]), 1)

        # dot product between the example pairs.
        repel_similarity_between_examples = tf.reduce_sum(tf.multiply(repel_examples_left, repel_examples_right),
                                                          1)  # becomes batch_size again, might need tf.squeeze

        # dot product of each word in the example with its negative example.
        repel_similarity_to_negatives_left = tf.reduce_sum(
            tf.multiply(repel_examples_left, negative_examples_repel_left), 1)
        repel_similarity_to_negatives_right = tf.reduce_sum(
            tf.multiply(repel_examples_right, negative_examples_repel_right), 1)

        # and the final Repel Cost Function (sans regularisation):
        self.repel_cost = tf.nn.relu(
            self.repel_margin - repel_similarity_to_negatives_left + repel_similarity_between_examples) + \
                          tf.nn.relu(
                              self.repel_margin - repel_similarity_to_negatives_right + repel_similarity_between_examples)

        # The Regularisation Cost (separate for the two terms, depending on which one is called):

        # load the original distributional vectors for the example pairs:
        original_attract_examples_left = tf.nn.embedding_lookup(self.W_init, self.attract_examples[:, 0])
        original_attract_examples_right = tf.nn.embedding_lookup(self.W_init, self.attract_examples[:, 1])

        original_repel_examples_left = tf.nn.embedding_lookup(self.W_init, self.repel_examples[:, 0])
        original_repel_examples_right = tf.nn.embedding_lookup(self.W_init, self.repel_examples[:, 1])

        # and then define the respective regularisation costs:
        regularisation_cost_attract = self.regularisation_constant * (
                    tf.nn.l2_loss(original_attract_examples_left - attract_examples_left) + tf.nn.l2_loss(
                original_attract_examples_right - attract_examples_right))
        self.attract_cost += regularisation_cost_attract

        regularisation_cost_repel = self.regularisation_constant * (
                    tf.nn.l2_loss(original_repel_examples_left - repel_examples_left) + tf.nn.l2_loss(
                original_repel_examples_right - repel_examples_right))
        self.repel_cost += regularisation_cost_repel

        # Finally, we define the training step functions for both steps.

        tvars = tf.trainable_variables()
        attract_grads = [tf.clip_by_value(grad, -2., 2.) for grad in tf.gradients(self.attract_cost, tvars)]
        repel_grads = [tf.clip_by_value(grad, -2., 2.) for grad in tf.gradients(self.repel_cost, tvars)]

        attract_optimiser = tf.train.AdagradOptimizer(0.05)
        repel_optimiser = tf.train.AdagradOptimizer(0.05)

        self.attract_cost_step = attract_optimiser.apply_gradients(zip(attract_grads, tvars))
        self.repel_cost_step = repel_optimiser.apply_gradients(zip(repel_grads, tvars))

        # return the handles for loading vectors from the TensorFlow embeddings:
        return attract_examples_left, attract_examples_right, repel_examples_left, repel_examples_right

    def load_constraints(self, constraints_filepath):
        """
        This methods reads a collection of constraints from the specified file, and returns a set with
        all constraints for which both of their constituent words are in the specified vocabulary.
        """
        constraints_filepath.strip()
        constraints = set()

        with codecs.open(constraints_filepath, "r", "utf-8") as f:
            #print "\n---- Pairs that cannot be fetch."
            for line in f:
                word_pair = line.split()
                word_i = word_pair[0][2:-2]
                word_j = word_pair[1][1:-2]
                if word_i in self.vocabulary and word_j in self.vocabulary and word_i != word_j:
                    constraints |= {(self.vocab_index[word_i], self.vocab_index[word_j])}
                #else:
                    #print word_i, word_j

        return constraints

    def load_experiment_hyperparameters(self):
        """
        This method loads/sets the hyperparameters of the procedure as specified in the paper.
        """
        self.attract_margin_value = self.config.getfloat("hyperparameters", "attract_margin")
        self.repel_margin_value = self.config.getfloat("hyperparameters", "repel_margin")
        self.batch_size = int(self.config.getfloat("hyperparameters", "batch_size"))
        self.regularisation_constant_value = self.config.getfloat("hyperparameters", "l2_reg_constant")
        self.max_iter = self.config.getfloat("hyperparameters", "max_iter")
        self.log_scores_over_time = self.config.get("experiment", "log_scores_over_time")
        self.print_simlex = self.config.get("experiment", "print_simlex")

        if self.log_scores_over_time in ["True", "true"]:
            self.log_scores_over_time = True
        else:
            self.log_scores_over_time = False

        if self.print_simlex in ["True", "true"]:
            self.print_simlex = True
        else:
            self.print_simlex = False

        print "Experiment hyperparameters (attract_margin, repel_margin, batch_size, l2_reg_constant, max_iter):", \
            self.attract_margin_value, self.repel_margin_value, self.batch_size, self.regularisation_constant_value, self.max_iter, "\n"

    def extract_negative_examples(self, list_minibatch, attract_batch=True):
        """
        For each example in the minibatch, this method returns the closest vector which is not
        in each words example pair.
        """
        list_of_representations = []
        list_of_indices = []

        representations = self.sess.run([self.embedding_attract_left, self.embedding_attract_right],
                                        feed_dict={self.attract_examples: list_minibatch})

        for idx, (example_left, example_right) in enumerate(list_minibatch):
            list_of_representations.append(representations[0][idx])
            list_of_representations.append(representations[1][idx])

            list_of_indices.append(example_left)
            list_of_indices.append(example_right)

        condensed_distance_list = pdist(list_of_representations, 'cosine')
        square_distance_list = squareform(condensed_distance_list)

        if attract_batch:
            default_value = 2.0  # value to set for given attract/repel pair, so that it can not be found as closest or furthest away.
        else:
            default_value = 0.0  # for antonyms, we want the opposite value from the synonym one. Cosine Distance is [0,2].

        for i in range(len(square_distance_list)):

            square_distance_list[i, i] = default_value

            if i % 2 == 0:
                square_distance_list[i, i + 1] = default_value
            else:
                square_distance_list[i, i - 1] = default_value

        if attract_batch:
            negative_example_indices = numpy.argmin(square_distance_list,
                                                    axis=1)  # for each of the 100 elements, finds the index which has the minimal cosine distance (i.e. most similar).
        else:
            negative_example_indices = numpy.argmax(square_distance_list,
                                                    axis=1)  # for antonyms, find the least similar one.

        negative_examples = []

        for idx in range(len(list_minibatch)):
            negative_example_left = list_of_indices[negative_example_indices[2 * idx]]
            negative_example_right = list_of_indices[negative_example_indices[2 * idx + 1]]

            negative_examples.append((negative_example_left, negative_example_right))

        negative_examples = mix_sampling(list_minibatch, negative_examples)

        return negative_examples

    def attract_repel(self):
        """
        This method repeatedly applies optimisation steps to fit the word vectors to the provided linguistic constraints.
        """
        current_iteration = 0

        # Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
        for antonym_pair in self.antonyms:
            if antonym_pair in self.synonyms:
                self.synonyms.remove(antonym_pair)

        self.synonyms = list(self.synonyms)
        self.antonyms = list(self.antonyms)

        self.syn_count = len(self.synonyms)
        self.ant_count = len(self.antonyms)

        print "\nAntonym pairs:", len(self.antonyms), "Synonym pairs:", len(self.synonyms)

        list_of_simlex = []
        list_of_wordsim = []

        syn_batches = int(self.syn_count / self.batch_size)
        ant_batches = int(self.ant_count / self.batch_size)

        batches_per_epoch = syn_batches + ant_batches

        print "\nRunning the optimisation procedure for", self.max_iter, "iterations..."

        last_time = time.time()

        if self.log_scores_over_time:
            fwrite_simlex = open("results/simlex_scores.txt", "w")
            fwrite_wordsim = open("results/wordsim_scores.txt", "w")

        while current_iteration < self.max_iter:

            # how many attract/repel batches we've done in this epoch so far.
            antonym_counter = 0
            synonym_counter = 0

            order_of_synonyms = range(0, self.syn_count)
            order_of_antonyms = range(0, self.ant_count)

            random.shuffle(order_of_synonyms)
            random.shuffle(order_of_antonyms)

            # list of 0 where we run synonym batch, 1 where we run antonym batch
            list_of_batch_types = [0] * batches_per_epoch
            list_of_batch_types[syn_batches:] = [1] * ant_batches  # all antonym batches to 1
            random.shuffle(list_of_batch_types)

            if current_iteration == 0:
                print "\nStarting epoch:", current_iteration + 1, "\n"
            else:
                print "\nStarting epoch:", current_iteration + 1, "Last epoch took:", round(time.time() - last_time,
                                                                                            1), "seconds. \n"
                last_time = time.time()

            for batch_index in range(0, batches_per_epoch):

                # we can Log SimLex / WordSim scores
                if self.log_scores_over_time and (batch_index % (batches_per_epoch / 20) == 0):
                    simlex_score = self.create_vector_dictionary()
                    list_of_simlex.append(simlex_score)

                    print >> fwrite_simlex, len(list_of_simlex) + 1, simlex_score

                syn_or_ant_batch = list_of_batch_types[batch_index]

                if syn_or_ant_batch == 0:
                    # do one synonymy batch:

                    synonymy_examples = [self.synonyms[order_of_synonyms[x]] for x in
                                         range(synonym_counter * self.batch_size,
                                               (synonym_counter + 1) * self.batch_size)]
                    current_negatives = self.extract_negative_examples(synonymy_examples, attract_batch=True)

                    self.sess.run([self.attract_cost_step], feed_dict={self.attract_examples: synonymy_examples,
                                                                       self.negative_examples_attract: current_negatives, \
                                                                       self.attract_margin: self.attract_margin_value,
                                                                       self.regularisation_constant: self.regularisation_constant_value})
                    synonym_counter += 1

                else:

                    antonymy_examples = [self.antonyms[order_of_antonyms[x]] for x in
                                         range(antonym_counter * self.batch_size,
                                               (antonym_counter + 1) * self.batch_size)]
                    current_negatives = self.extract_negative_examples(antonymy_examples, attract_batch=False)

                    self.sess.run([self.repel_cost_step], feed_dict={self.repel_examples: antonymy_examples,
                                                                     self.negative_examples_repel: current_negatives, \
                                                                     self.repel_margin: self.repel_margin_value,
                                                                     self.regularisation_constant: self.regularisation_constant_value})

                    antonym_counter += 1

            current_iteration += 1
            self.create_vector_dictionary()  # whether to print SimLex score at the end of each epoch

    def create_vector_dictionary(self):
        """
        Extracts the current word vectors from TensorFlow embeddings and (if print_simlex=True) prints their SimLex scores.
        """
        log_time = time.time()

        [current_vectors] = self.sess.run([self.W_dynamic])
        self.word_vectors = {}
        for idx in range(0, self.vocabulary_size):
            self.word_vectors[self.inverted_index[idx]] = normalise_vector(current_vectors[idx, :])

        if self.log_scores_over_time or self.print_simlex:
            simlex_scores(self.word_vectors, self.original_vectors, "internal", self.print_simlex)
            #return score_simlex

        #return 1.0


def random_different_from(top_range, number_to_not_repeat):
    result = random.randint(0, top_range - 1)
    while result == number_to_not_repeat:
        result = random.randint(0, top_range - 1)

    return result


def mix_sampling(list_of_examples, negative_examples):
    """
    Converts half of the negative examples to random words from the batch (that are not in the given example pair).
    """
    mixed_negative_examples = []
    batch_size = len(list_of_examples)

    for idx, (left_idx, right_idx) in enumerate(negative_examples):

        new_left = left_idx
        new_right = right_idx

        if random.random() >= 0.5:
            new_left = list_of_examples[random_different_from(batch_size, idx)][random.randint(0, 1)]

        if random.random() >= 0.5:
            new_right = list_of_examples[random_different_from(batch_size, idx)][random.randint(0, 1)]

        mixed_negative_examples.append((new_left, new_right))

    return mixed_negative_examples


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word] ** 2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors


def load_word_vectors(file_destination):
    """
    This method loads the word vectors from the supplied file destination.
    It loads the dictionary of word vectors and prints its size and the vector dimensionality.
    """
    print "Loading pretrained word vectors from", file_destination
    word_dictionary = {}
    try:
        #print "---- Print the exceptions."
        f = codecs.open(file_destination, 'r', 'utf-8')
        for line in f:
            line = line.split(", ", 1)
            #key = unicode(line[0].lower())
            #word_dictionary[key] = numpy.fromstring(line[1], dtype="float32", sep=" ")
            #key = unicode((line[0].lower().split("'"))[1].split("'")[0])
            #value = (line[1].split("'"))[1].split("'")[0]
            key = unicode(line[0][2:-1])
            #key = unicode(line[0].lower()[2:-1])
            """
            if str(key) == 'o.k..n.01.O.K.' or str(key) == 'o.k..n.01.OK' or str(key)== 'ph.d..n.01.Ph.D.' \
                    or str(key) == 'ph.d..n.01.PhD' or str(key) == 'g.i..v.01.G.I.' or str(key) == 'g.i..v.01.GI':
                print key, line[1][1:-2]
            """
            word_dictionary[key] = numpy.fromstring(line[1][1:-2], dtype="float32", sep=" ")
    except:
        print "Word vectors could not be loaded from:", file_destination
        return {}
    print len(word_dictionary), "vectors loaded from", file_destination, "\n"
    return word_dictionary


def print_word_vectors(word_vectors, write_path):
    """
    This function prints the collection of word vectors to file, in a plain textual format.
    """
    f_write = codecs.open(write_path, 'w', 'utf-8')

    for key in word_vectors:
        print >> f_write, key, " ".join(map(unicode, numpy.round(word_vectors[key], decimals=6)))

    print "\nPrinted", len(word_vectors), "word vectors to:", write_path


def simlex_analysis(word_vectors, original_vectors, begin_end, language="english", source="simlex", add_prefixes=False):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors.
    """
    pair_list = []
    if source == "simlex-old":
        fread_simlex = codecs.open("evaluation/simlex-english-old.txt", 'r', 'utf-8')
    elif source == "simverb":
        fread_simlex = codecs.open("evaluation/simverb.txt", 'r', 'utf-8')
    elif source == "wordsim":
        fread_simlex = codecs.open("evaluation/wordsim353-" + language + ".txt", 'r', 'utf-8')  # specify english, english-rel, etc.
    elif source == "YP-130":
        fread_simlex = codecs.open("evaluation/YP-130.txt", 'r', 'utf-8')
    elif source == "RG-65":
        fread_simlex = codecs.open("evaluation/RG-65.txt", 'r', 'utf-8')
    elif source == "MEN-3K":
        fread_simlex = codecs.open("evaluation/MEN-3K.txt", 'r', 'utf-8')
    elif source == "scws_ratings":
        fread_simlex = codecs.open("evaluation/scws_ratings.txt", 'r', 'utf-8')

    # needed for prefixes if we are adding these.
    lp_map = {}
    lp_map["english"] = u"en_"
    #lp_map["german"] = u"de_"
    #lp_map["italian"] = u"it_"
    #lp_map["russian"] = u"ru_"
    #lp_map["croatian"] = u"sh_"
    #lp_map["hebrew"] = u"he_"

    if source == "scws_ratings":
        for line in fread_simlex:
            tokens = line.split('\t')
            # word_i = tokens[1].lower()
            # word_j = tokens[3].lower()
            word_i = tokens[1]
            word_j = tokens[3]
            score = float(tokens[7])
            if add_prefixes:
                word_i = lp_map[language] + word_i
                word_j = lp_map[language] + word_j
            if word_i in original_vectors and word_j in original_vectors:
                pair_list.append(((word_i, word_j), score))
            else:
                pass

    else:
        line_number = 0
        for line in fread_simlex:
            if line_number > 0:
                tokens = line.split()
                #word_i = tokens[0].lower()
                #word_j = tokens[1].lower()
                word_i = tokens[0]
                word_j = tokens[1]
                score = float(tokens[2])
                if add_prefixes:
                    word_i = lp_map[language] + word_i
                    word_j = lp_map[language] + word_j
                if word_i in original_vectors and word_j in original_vectors:
                    pair_list.append(((word_i, word_j), score))
                else:
                    pass
            line_number += 1

    if not pair_list:
        return (0.0, 0)
    else:
        if begin_end == "begin":
            print "-- There are ", len(pair_list), " pairs from ", source, " can be fetched in the vocabulary."

    pair_list.sort(key=lambda x: - x[1])

    coverage = len(pair_list)

    word_vectors_list = []
    word_vectors_vocabulary = []
    for key in word_vectors:
        if '.n.' in key:
            word = key.split('.n.')[1][3:]
        if '.v.' in key:
            word = key.split('.v.')[1][3:]
        if '.a.' in key:
            word = key.split('.a.')[1][3:]
        if '.s.' in key:
            word = key.split('.s.')[1][3:]
        if word not in word_vectors_vocabulary:
            word_vectors_vocabulary.append(word)
        word_vectors_list.append([word, key, word_vectors[key]])

    """
    if average_max == "average":
        if begin_end == "begin":
            if source == "simlex":
                write_simlex = codecs.open("results/average/simlex_begin_scores.txt", "w", 'utf-8')
            elif source == "simlex-old":
                write_simlex = codecs.open("results/average/simlex_old_begin_scores.txt", "w", 'utf-8')
            elif source == "simverb":
                write_simlex = codecs.open("results/average/simverb_begin_scores.txt", "w", 'utf-8')
            elif source == "wordsim":
                write_simlex = codecs.open("results/average/wordsim_begin_scores.txt", "w", 'utf-8')
            elif source == "YP-130":
                write_simlex = codecs.open("results/average/YP_130_begin_scores.txt", "w", 'utf-8')
            elif source == "RG-65":
                write_simlex = codecs.open("results/average/RG_65_begin_scores.txt", "w", 'utf-8')
            elif source == "MEN-3K":
                write_simlex = codecs.open("results/average/MEN_3K_begin_scores.txt", "w", 'utf-8')
            print >> write_simlex, "word 1	word 2	begin_distance_score"
        elif begin_end == "end":
            if source == "simlex":
                write_simlex = codecs.open("results/average/simlex_end_scores.txt", "w", 'utf-8')
            elif source == "simlex-old":
                write_simlex = codecs.open("results/average/simlex_old_end_scores.txt", "w", 'utf-8')
            elif source == "simverb":
                write_simlex = codecs.open("results/average/simverb_end_scores.txt", "w", 'utf-8')
            elif source == "wordsim":
                write_simlex = codecs.open("results/average/wordsim_end_scores.txt", "w", 'utf-8')
            elif source == "YP-130":
                write_simlex = codecs.open("results/average/YP_130_end_scores.txt", "w", 'utf-8')
            elif source == "RG-65":
                write_simlex = codecs.open("results/average/RG_65_end_scores.txt", "w", 'utf-8')
            elif source == "MEN-3K":
                write_simlex = codecs.open("results/average/MEN_3K_end_scores.txt", "w", 'utf-8')
            print >> write_simlex, "word 1	word 2	end_distance_score"
    elif average_max == "max":
        if begin_end == "begin":
            if source == "simlex":
                write_simlex = codecs.open("results/max/simlex_begin_scores.txt", "w", 'utf-8')
            elif source == "simlex-old":
                write_simlex = codecs.open("results/max/simlex_old_begin_scores.txt", "w", 'utf-8')
            elif source == "simverb":
                write_simlex = codecs.open("results/max/simverb_begin_scores.txt", "w", 'utf-8')
            elif source == "wordsim":
                write_simlex = codecs.open("results/max/wordsim_begin_scores.txt", "w", 'utf-8')
            elif source == "YP-130":
                write_simlex = codecs.open("results/max/YP_130_begin_scores.txt", "w", 'utf-8')
            elif source == "RG-65":
                write_simlex = codecs.open("results/max/RG_65_begin_scores.txt", "w", 'utf-8')
            elif source == "MEN-3K":
                write_simlex = codecs.open("results/max/MEN_3K_begin_scores.txt", "w", 'utf-8')
            print >> write_simlex, "word 1	word 2	begin_distance_score"
        elif begin_end == "end":
            if source == "simlex":
                write_simlex = codecs.open("results/max/simlex_end_scores.txt", "w", 'utf-8')
            elif source == "simlex-old":
                write_simlex = codecs.open("results/max/simlex_old_end_scores.txt", "w", 'utf-8')
            elif source == "simverb":
                write_simlex = codecs.open("results/max/simverb_end_scores.txt", "w", 'utf-8')
            elif source == "wordsim":
                write_simlex = codecs.open("results/max/wordsim_end_scores.txt", "w", 'utf-8')
            elif source == "YP-130":
                write_simlex = codecs.open("results/max/YP_130_end_scores.txt", "w", 'utf-8')
            elif source == "RG-65":
                write_simlex = codecs.open("results/max/RG_65_end_scores.txt", "w", 'utf-8')
            elif source == "MEN-3K":
                write_simlex = codecs.open("results/max/MEN_3K_end_scores.txt", "w", 'utf-8')
            print >> write_simlex, "word 1	word 2	end_distance_score"
    """

    extracted_list_average = []
    extracted_scores_average = {}
    extracted_list_max = []
    extracted_scores_max = {}

    new_vector_count = 0
    old_vector_count = 0

    for (x, y) in pair_list:
        (word_i, word_j) = x
        set_i = []
        set_j = []
        if word_i in word_vectors_vocabulary:
            new_vector_count += 1
            for sense in word_vectors_list:
                if sense[0] == word_i:
                    set_i.append(sense[2])
        else:
            old_vector_count += 1
            # If it is fastText, use the following code.
            set_i.append(original_vectors.get(word_i))
            # If it is word2vec, use the following code.
            #set_i.append(original_vectors.get_vector(word_i))
        if word_j in word_vectors_vocabulary:
            new_vector_count += 1
            for sense in word_vectors_list:
                if sense[0] == word_j:
                    set_j.append(sense[2])
        else:
            old_vector_count += 1
            # If it is fastText, use the following code.
            set_j.append(original_vectors.get(word_j))
            # If it is word2vec, use the following code.
            #set_j.append(original_vectors.get_vector(word_j))

        average_dis = 0
        #average_sim = 0
        for i in set_i:
            for j in set_j:
                average_dis += distance(i, j)
                #average_sim += cos_sim(i, j)
        current_distance_average = average_dis/(len(set_i) * len(set_j))
        #current_similarity = average_sim/(len(set_i) * len(set_j))

        #max_sim = -100
        c = 0
        for i in set_i:
            for j in set_j:
                d = distance(i, j)
                #s = cos_sim(i, j)
                if c == 0:
                    current_distance_max = d
                if d > current_distance_max:
                    current_distance_max = d
                c = c + 1
                #if s > max_sim:
                    #current_similarity = s

        #current_distance = distance(word_vectors[word_i], word_vectors[word_j])
        extracted_scores_average[(word_i, word_j)] = current_distance_average
        extracted_list_average.append(((word_i, word_j), current_distance_average))
        extracted_scores_max[(word_i, word_j)] = current_distance_max
        extracted_list_max.append(((word_i, word_j), current_distance_max))

        """
        if begin_end == "begin" or begin_end == "end":
            print >> write_simlex, word_i, word_j, current_similarity
        """

    if begin_end == "begin":
        print "---- The amount of words in sense vector dictionary is : ", new_vector_count
        print "---- The amount of words in original word vector dictionary is : ", old_vector_count

    extracted_list_average.sort(key=lambda x: x[1])
    extracted_list_max.sort(key=lambda x: x[1])

    spearman_original_list_average = []
    spearman_target_list_average = []
    spearman_original_list_max = []
    spearman_target_list_max = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores_average[word_pair]
        position_2 = extracted_list_average.index((word_pair, score_2))
        spearman_original_list_average.append(position_1)
        spearman_target_list_average.append(position_2)

    spearman_rho_average = spearmanr(spearman_original_list_average, spearman_target_list_average)

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores_max[word_pair]
        position_2 = extracted_list_max.index((word_pair, score_2))
        spearman_original_list_max.append(position_1)
        spearman_target_list_max.append(position_2)

    spearman_rho_max = spearmanr(spearman_original_list_max, spearman_target_list_max)

    return round(spearman_rho_average[0], 3), round(spearman_rho_max[0], 3), coverage


def simlex_analysis_clss(word_vectors, original_vectors, begin_end, language="english", source="sscws_ratings", add_prefixes=False):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors.
    """
    pair_list = []
    fread_simlex = codecs.open("evaluation/word2sense_test.txt", 'r', 'utf-8')

    # needed for prefixes if we are adding these.
    lp_map = {}
    lp_map["english"] = u"en_"
    #lp_map["german"] = u"de_"
    #lp_map["italian"] = u"it_"
    #lp_map["russian"] = u"ru_"
    #lp_map["croatian"] = u"sh_"
    #lp_map["hebrew"] = u"he_"

    unseen_word = []

    for line in fread_simlex:
        tokens = line.split('\t')
        #word_i = tokens[1].lower()
        #word_j = tokens[3].lower()
        synset = tokens[1].split("%")[0]
        token_i = tokens[2]
        individual_sense = token_i.split("#")[0]
        pos_sense = token_i.split("#")[1]
        if len(token_i.split("#")[2]) == 1:
            dex = '0' + token_i.split("#")[2]
        else:
            dex = token_i.split("#")[2]
        word_i = synset + '.' + pos_sense + '.' + dex + '.' + individual_sense
        word_j = tokens[0].split("#")[0]
        if individual_sense not in original_vectors or word_j not in original_vectors:
            unseen_word.append([individual_sense, word_j])
        score = int(tokens[3].split("-")[1])
        if add_prefixes:
            word_i = lp_map[language] + word_i
            word_j = lp_map[language] + word_j
        if individual_sense in original_vectors and word_j in original_vectors:
            pair_list.append(((word_i, word_j), score))
        else:
            pass

    if not pair_list:
        return (0.0, 0)
    else:
        if begin_end == "begin":
            print "-- There are ", len(pair_list), " pairs from ", source, " can be fetched in the vocabulary."

    pair_list.sort(key=lambda x: - x[1])

    coverage = len(pair_list)

    word_vectors_list = []
    word_vectors_vocabulary = []
    sense_vectors_vocabulary = []
    for key in word_vectors:
        if '.n.' in key:
            #word_sense = 'n.' + key.split('.n.')[1]
            word = key.split('.n.')[1][3:]
        if '.v.' in key:
            word = key.split('.v.')[1][3:]
            #word_sense = 'v.' + key.split('.v.')[1]
        if '.a.' in key:
            word = key.split('.a.')[1][3:]
            #word_sense = 'a.' + key.split('.a.')[1]
        if '.s.' in key:
            word = key.split('.s.')[1][3:]
            #word_sense = 's.' + key.split('.s.')[1]
        if word not in word_vectors_vocabulary:
            word_vectors_vocabulary.append(word)
            sense_vectors_vocabulary.append(key)
        word_vectors_list.append([word, key, word_vectors[key]])

    """
    if average_max == "average":
        if begin_end == "begin":
            if source == "simlex":
                write_simlex = codecs.open("results/average/simlex_begin_scores.txt", "w", 'utf-8')
            elif source == "simlex-old":
                write_simlex = codecs.open("results/average/simlex_old_begin_scores.txt", "w", 'utf-8')
            elif source == "simverb":
                write_simlex = codecs.open("results/average/simverb_begin_scores.txt", "w", 'utf-8')
            elif source == "wordsim":
                write_simlex = codecs.open("results/average/wordsim_begin_scores.txt", "w", 'utf-8')
            elif source == "YP-130":
                write_simlex = codecs.open("results/average/YP_130_begin_scores.txt", "w", 'utf-8')
            elif source == "RG-65":
                write_simlex = codecs.open("results/average/RG_65_begin_scores.txt", "w", 'utf-8')
            elif source == "MEN-3K":
                write_simlex = codecs.open("results/average/MEN_3K_begin_scores.txt", "w", 'utf-8')
            print >> write_simlex, "word 1	word 2	begin_distance_score"
        elif begin_end == "end":
            if source == "simlex":
                write_simlex = codecs.open("results/average/simlex_end_scores.txt", "w", 'utf-8')
            elif source == "simlex-old":
                write_simlex = codecs.open("results/average/simlex_old_end_scores.txt", "w", 'utf-8')
            elif source == "simverb":
                write_simlex = codecs.open("results/average/simverb_end_scores.txt", "w", 'utf-8')
            elif source == "wordsim":
                write_simlex = codecs.open("results/average/wordsim_end_scores.txt", "w", 'utf-8')
            elif source == "YP-130":
                write_simlex = codecs.open("results/average/YP_130_end_scores.txt", "w", 'utf-8')
            elif source == "RG-65":
                write_simlex = codecs.open("results/average/RG_65_end_scores.txt", "w", 'utf-8')
            elif source == "MEN-3K":
                write_simlex = codecs.open("results/average/MEN_3K_end_scores.txt", "w", 'utf-8')
            print >> write_simlex, "word 1	word 2	end_distance_score"
    elif average_max == "max":
        if begin_end == "begin":
            if source == "simlex":
                write_simlex = codecs.open("results/max/simlex_begin_scores.txt", "w", 'utf-8')
            elif source == "simlex-old":
                write_simlex = codecs.open("results/max/simlex_old_begin_scores.txt", "w", 'utf-8')
            elif source == "simverb":
                write_simlex = codecs.open("results/max/simverb_begin_scores.txt", "w", 'utf-8')
            elif source == "wordsim":
                write_simlex = codecs.open("results/max/wordsim_begin_scores.txt", "w", 'utf-8')
            elif source == "YP-130":
                write_simlex = codecs.open("results/max/YP_130_begin_scores.txt", "w", 'utf-8')
            elif source == "RG-65":
                write_simlex = codecs.open("results/max/RG_65_begin_scores.txt", "w", 'utf-8')
            elif source == "MEN-3K":
                write_simlex = codecs.open("results/max/MEN_3K_begin_scores.txt", "w", 'utf-8')
            print >> write_simlex, "word 1	word 2	begin_distance_score"
        elif begin_end == "end":
            if source == "simlex":
                write_simlex = codecs.open("results/max/simlex_end_scores.txt", "w", 'utf-8')
            elif source == "simlex-old":
                write_simlex = codecs.open("results/max/simlex_old_end_scores.txt", "w", 'utf-8')
            elif source == "simverb":
                write_simlex = codecs.open("results/max/simverb_end_scores.txt", "w", 'utf-8')
            elif source == "wordsim":
                write_simlex = codecs.open("results/max/wordsim_end_scores.txt", "w", 'utf-8')
            elif source == "YP-130":
                write_simlex = codecs.open("results/max/YP_130_end_scores.txt", "w", 'utf-8')
            elif source == "RG-65":
                write_simlex = codecs.open("results/max/RG_65_end_scores.txt", "w", 'utf-8')
            elif source == "MEN-3K":
                write_simlex = codecs.open("results/max/MEN_3K_end_scores.txt", "w", 'utf-8')
            print >> write_simlex, "word 1	word 2	end_distance_score"
    """

    extracted_list_average = []
    extracted_scores_average = {}
    extracted_list_max = []
    extracted_scores_max = {}
    extracted_list_s2w = []
    extracted_scores_s2w = {}
    extracted_list_s2a = []
    extracted_scores_s2a = {}

    new_vector_count = 0
    old_vector_count = 0

    for (x, y) in pair_list:
        (word_i, word_j) = x
        set_j = []
        if word_i in sense_vectors_vocabulary:
            new_vector_count += 1
            for sense in word_vectors_list:
                if sense[1] == word_i:
                    set_i = sense[2]
        else:
            old_vector_count += 1
            # If it is fastText, use the following code.
            set_i = original_vectors.get(word_i.split('.', 4)[3])
            # If it is word2vec, use the following code.
            #set_i = original_vectors.get_vector(word_i.split('.', 4)[3])
        if word_j in word_vectors_vocabulary:
            new_vector_count += 1
            for sense in word_vectors_list:
                #if sense[0] == word_j and sense[1][0] == word_j:
                if sense[0] == word_j:
                    set_j.append(sense[2])
        else:
            old_vector_count += 1
            # If it is fastText, use the following code.
            set_j.append(original_vectors.get(word_j))
            # If it is word2vec, use the following code.
            #set_j.append(original_vectors.get_vector(word_j))
        if word_j in original_vectors:
            # If it is fastText, use the following code.
            set_k = original_vectors.get(word_j)
            # If it is word2vec, use the following code.
            #set_k = original_vectors.get_vector(word_j)

        average_dis = 0
        #average_sim = 0
        for j in set_j:
            average_dis = distance(set_i, j) + average_dis
            #average_sim += cos_sim(set_i, j)
        try:
            current_distance_average = average_dis/len(set_j)
        except:
            print word_j
        #current_similarity = average_sim/(len(set_i) * len(set_j))

        #max_sim = -100
        c = -100
        for j in set_j:
            d = distance(set_i, j)
            #s = cos_sim(set_i, j)
            if c == -100:
                current_distance_max = d
            if d > current_distance_max:
                current_distance_max = d
            #c = c + 1
            #if s > max_sim:
                #current_similarity = s

        current_distance_s2w = distance(set_i, set_k)

        '''
        for i in range(0, 300):
            element_sum = 0
            for j in set_j:
                element_sum = j[i] + element_sum
            set_l = element_sum/len(set_j)
        '''
        set_j_array = numpy.asarray(set_j)
        set_l = set_j_array.mean(axis=0)
        current_distance_s2a = distance(set_i, set_l)

        #current_distance = distance(word_vectors[word_i], word_vectors[word_j])
        extracted_scores_average[(word_i, word_j)] = current_distance_average
        extracted_list_average.append(((word_i, word_j), current_distance_average))
        extracted_scores_max[(word_i, word_j)] = current_distance_max
        extracted_list_max.append(((word_i, word_j), current_distance_max))
        extracted_scores_s2w[(word_i, word_j)] = current_distance_s2w
        extracted_list_s2w.append(((word_i, word_j), current_distance_s2w))
        extracted_scores_s2a[(word_i, word_j)] = current_distance_s2a
        extracted_list_s2a.append(((word_i, word_j), current_distance_s2a))

        """
        if begin_end == "begin" or begin_end == "end":
            print >> write_simlex, word_i, word_j, current_similarity
        """

    if begin_end == "begin":
        print "---- The amount of words in sense vector dictionary is : ", new_vector_count
        print "---- The amount of words in original word vector dictionary is : ", old_vector_count

    extracted_list_average.sort(key=lambda x: x[1])
    extracted_list_max.sort(key=lambda x: x[1])
    extracted_list_s2w.sort(key=lambda x: x[1])
    extracted_list_s2a.sort(key=lambda x: x[1])

    spearman_original_list_average = []
    spearman_target_list_average = []
    spearman_original_list_max = []
    spearman_target_list_max = []
    spearman_original_list_s2w = []
    spearman_target_list_s2w = []
    spearman_original_list_s2a = []
    spearman_target_list_s2a = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores_average[word_pair]
        position_2 = extracted_list_average.index((word_pair, score_2))
        spearman_original_list_average.append(position_1)
        spearman_target_list_average.append(position_2)

    spearman_rho_average = spearmanr(spearman_original_list_average, spearman_target_list_average)

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores_max[word_pair]
        position_2 = extracted_list_max.index((word_pair, score_2))
        spearman_original_list_max.append(position_1)
        spearman_target_list_max.append(position_2)

    spearman_rho_max = spearmanr(spearman_original_list_max, spearman_target_list_max)

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores_s2w[word_pair]
        position_2 = extracted_list_s2w.index((word_pair, score_2))
        spearman_original_list_s2w.append(position_1)
        spearman_target_list_s2w.append(position_2)

    spearman_rho_s2w = spearmanr(spearman_original_list_max, spearman_target_list_max)

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores_s2a[word_pair]
        position_2 = extracted_list_s2a.index((word_pair, score_2))
        spearman_original_list_s2a.append(position_1)
        spearman_target_list_s2a.append(position_2)

    spearman_rho_s2a = spearmanr(spearman_original_list_max, spearman_target_list_max)

    return round(spearman_rho_average[0], 3), round(spearman_rho_max[0], 3), round(spearman_rho_s2w[0], 3), round(spearman_rho_s2a[0], 3), coverage


def normalise_vector(v1):
    return v1 / norm(v1)


def distance(v1, v2, normalised_vectors=False):
    """
    Returns the cosine distance between two vectors.
    If the vectors are normalised, there is no need for the denominator, which is always one.
    """
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / (norm(v1) * norm(v2))

def cos_sim(v1, v2):
    """
    Returns the cosine similarity between two vectors.
    """
    return dot(v1, v2)/(norm(v1) * norm(v2))


def simlex_scores(word_vectors, original_vectors, begin_end, print_simlex=True):
    #for language in ["english", "german", "italian", "russian", "croatian", "hebrew"]:
    for language in ["english"]:

        """
        simlex_score, simlex_coverage = simlex_analysis(word_vectors, language)

        if language not in ["hebrew", "croatian"]:
            ws_score, ws_coverage = simlex_analysis(word_vectors, language, source="wordsim")
        else:
            ws_score = 0.0
            ws_coverage = 0

        if language == "english":
            simverb_score, simverb_coverage = simlex_analysis(word_vectors, language, source="simverb")

        if simlex_coverage > 0:

            if print_simlex:

                if language == "english":
                    simlex_old, cov_old = simlex_analysis(word_vectors, language, source="simlex-old")
                    print "SimLex score for", language, "is:", simlex_score, "Original SimLex score is:", simlex_old, "coverage:", simlex_coverage, "/ 999"
                    print "SimVerb score for", language, "is:", simverb_score, "coverage:", simverb_coverage, "/ 3500"
                    print "WordSim score for", language, "is:", ws_score, "coverage:", ws_coverage, "/ 353\n"
                elif language in ["italian", "german", "russian"]:
                    print "SimLex score for", language, "is:", simlex_score, "coverage:", simlex_coverage, "/ 999"
                    print "WordSim score for", language, "is:", ws_score, "coverage:", ws_coverage, "/ 353\n"
                elif language in ["hebrew", "croatian"]:
                    print "SimLex score for", language, "is:", simlex_score, "coverage:", simlex_coverage, "/ 999\n"

        if language == "english":
            simlex_score_en = simlex_score
            ws_score_en = ws_score
        """
        if begin_end == "begin":
            print "\nCount the words that can be fetched in sense word vectors or in the original vectors separately."

        simlex_average, simlex_max, cov_old = simlex_analysis(word_vectors, original_vectors, begin_end, language, source="simlex-old")
        simverb_score_average, simverb_score_max, simverb_coverage = simlex_analysis(word_vectors, original_vectors, begin_end, language, source="simverb")
        ws_score_average, ws_score_max, ws_coverage = simlex_analysis(word_vectors, original_vectors, begin_end, language, source="wordsim")
        YP_score_average, YP_score_max, YP_coverage = simlex_analysis(word_vectors, original_vectors, begin_end, language, source="YP-130")
        RG_score_average, RG_score_max, RG_coverage = simlex_analysis(word_vectors, original_vectors, begin_end, language, source="RG-65")
        MEN_score_average, MEN_score_max, MEN_coverage = simlex_analysis(word_vectors, original_vectors, begin_end,  language, source="MEN-3K")
        scws_score_average, scws_score_max, scws_coverage = simlex_analysis(word_vectors, original_vectors, begin_end, language, source="scws_ratings")
        clss_score_average, clss_score_max, clss_score_s2w, clss_score_s2a, clss_coverage = simlex_analysis_clss(word_vectors,
                                                                                                                 original_vectors,
                                                                                                                 begin_end, language,
                                                                                                                 source="word2sense_test")

        if print_simlex:
            print "\nSimLex AVG Spearman score for is:", simlex_average, ". And MAX Spearman score is:", simlex_max, " #coverage:", cov_old, "/ 999"
            print "SimVerb AVG Spearman score for is:", simverb_score_average, ". And MAX Spearman score is:", simverb_score_max, " #coverage:", simverb_coverage, "/ 3500"
            print "WordSim AVG Spearman score for is:", ws_score_average, ". And MAX Spearman score is:", ws_score_max, " #coverage:", ws_coverage, "/ 353"
            print "YP-130 AVG Spearman score for is:", YP_score_average, ". And MAX Spearman score is:", YP_score_max, " #coverage:", YP_coverage, "/ 130"
            print "RG-65 AVG Spearman score for is:", RG_score_average, ". And MAX Spearman score is:", RG_score_max, " #coverage:", RG_coverage, "/ 65"
            print "MEN-3K AVG Spearman score for is:", MEN_score_average, ". And MAX Spearman score is:", MEN_score_max, " #coverage:", MEN_coverage, "/ 3000"
            print "SCWS AVG Spearman score for is:", scws_score_average, ". And MAX Spearman score is:", scws_score_max, " #coverage:", scws_coverage, "/ 2003"
            print "CLSS AVG Spearman score is:", clss_score_average, ". And MAX Spearman score is:", clss_score_max, ". And S2W Spearman score is:", clss_score_s2w, ". And S2A Spearman score is:", clss_score_s2a, " #coverage:", clss_coverage, "/ 500\n"

    #return simlex_old


def run_experiment(config_filepath):
    """
    This method runs the counterfitting experiment, printing the SimLex-999 score of the initial
    vectors, then counter-fitting them using the supplied linguistic constraints.
    We then print the SimLex-999 score of the final vectors, and save them to a .txt file in the
    results directory.
    """
    current_experiment = ExperimentRun(config_filepath)

    current_experiment.attract_repel()

    print "\n---------- SimLex score (Spearman's rho coefficient) of the final vectors is: \n", \
        simlex_scores(current_experiment.word_vectors, current_experiment.original_vectors, "end"), "\n"

    os.system("mkdir -p results")

    print_word_vectors(current_experiment.word_vectors, current_experiment.output_filepath)


def main():
    """
    The user can provide the location of the config file as an argument.
    If no location is specified, the default config file (experiment_parameters.cfg) is used.
    """
    try:
        config_filepath = sys.argv[1]
    except:
        print "\nUsing the default config file: experiment_parameters.cfg\n"
        config_filepath = "experiment_parameters.cfg"

    run_experiment(config_filepath)


if __name__ == '__main__':
    main()
