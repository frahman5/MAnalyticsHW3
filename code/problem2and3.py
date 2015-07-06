## Movie Structure in the dataset:
## 'year': int[1930,1940,...,2010]
## 'episode' : str
## 'summary' : str|tupleOfStrings
## 'identifer': str
## 'title' : str
def format(string, norepeats=False):
    """
    str -> tuple

    Given a string, returns a tuple of all the words in the summary with some
    normalization (all words lowercase, no punctuation, optionally no repeats)
    """
    import re

    # Format
    formatted = [re.sub("[^a-z]", "", thing.lower()) for 
                                     thing in string.split()]

    # remove repeats, if called for, and return
    if norepeats:
        return tuple(set(formatted))
    return tuple(formatted)

def formatData(dataset):
    """
    listOfDictionaries -> listOfDictionaries

    dataset: each dictionary in the dataset has keys
        title
        year
        episode
        summary
        ... (etc.)

    For each movie in dataset, replaces movie['summary'] with a list of
    all the words in the summary with some normalization (all words lowercase, 
        no punctuation)
    """
    import re

    for movie in dataset:
        summary = movie['summary']
        movie['summary'] = format(summary)

    return dataset

def balanceDataset(dataset):
    """
    listOfDictionaries -> listOfDictionaries

    dataset: each dictionary in the dataset has keys
        title
        year
        episode
        summary
        ... (etc.)

    Uniformly samples 6000 movie dictionaries for each decade between 1930
    and 2010
    """
    import random
    # shortener = 10 # for training
    # print "TAKE SHORTENER OUT FOR PRODUCTION PHASE IN balanceDataset"
    num_samples = 6000
    balanced_dataset = []

    ## Create generators that contain the movies for each relevant decade
    movies_by_year = (
        [movie_dict for movie_dict in dataset if movie_dict['year'] == year] for 
            year in xrange(1930, 2020, 10))

    ## Randomly sample movies for each year
    for movie_generator in movies_by_year:
        balanced_dataset.extend(random.sample(movie_generator, num_samples))

    ## Sanity check and return the dataset
    assert len(balanced_dataset) == 54000 # 9 decades, 6000 movies each
    return balanced_dataset

def plotPMF(dataset, plotMetadata, condition = lambda x: True):
    """
    listOfDictionaries (tuple->boolean) str -> None

    dataset: each dictionary in the dataset has keys
        title
        year
        episode
        summary
        ... (etc.)
    condition: a function that given a tuple, returns true or false
        depending on some condition the functio defines (used for conditional
        plots)
    plotMetadata: dict with key,value pairs:
        title: title_of_plot (str)
        save: filepath_to_save_plot_to (str)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    ## Count the number of movies for each year
    year_counter = {}
    for movie_summary in dataset:
        # Condition on the data, if need be
        if not condition(movie_summary['summary']):
            continue

        ## Update the counter
        year = movie_summary['year']
        if year not in year_counter.keys():
            year_counter[year] = 1
            continue
        year_counter[year] += 1

    ## Plot the relevant plot
        # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

        # Format the data
    width = 1
    buckets = np.array(year_counter.keys())
    year_counts = [year_counter[year] for year in buckets]
    rects = ax.bar(buckets, year_counts, width, color='red')

        # axes and labels
    ax.set_xlim(min(buckets)-width, max(buckets)+width  )
    ax.set_ylim(0, max(year_counts))
    ax.set_ylabel('Num Movies from decade')
    ax.set_title(plotMetadata['title'])
    xTickMarks = ['{}'.format(year) for year in buckets]
    ax.set_xticks(buckets+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)

        # save the plot to file
    plt.savefig(plotMetadata['save'])

def plotHistogramForMovie(prob_dict, movie_title, summary, true_year):
    """
    dictOfDictionaries string tuple int -> None

    Given a training dictionary (see output of train function), a movie_title, 
    the movie summary, and the movie's decade, 
    produces a histogram of log(P(Y=y|movie=movie_title)) 
    vs. y for y in (1930, 1940, 1950, ... , 2010)

    Saves it to the approproiate filepath as defined in CONFIG
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from CONFIG import TWOJROOT                         # root of saved filepath

    # Store the probability for each year
    probs = []
    for year in xrange(1930, 2020, 10):
        probs.append(calcProbDecadeGivenWords(prob_dict, year, summary))
    predicted_year = predictYear(prob_dict, summary)

    ## Plot it as a histogram
        # Create the figure
    fig = plt.figure()
    fig.suptitle('Which decade is {} from'.format(movie_title))
    ax = fig.add_subplot(111)

        # Format the data
    width = 1
    xaxis = np.array(range(1930, 2020, 10))
    yaxis = probs
    rects = ax.bar(xaxis, yaxis, width, color='blue')

        # axes and labels
    ax.set_xlim(min(xaxis)-width, max(xaxis)+width)
    ax.set_ylim(min(yaxis), max(max(yaxis), 0))
    ax.set_ylabel('log(P(Y=y|movie={})'.format(movie_title))
    ax.set_title('True, Predicted: {}. {}'.format(true_year, predicted_year))
    xTickMarks = ['{}'.format(year) for year in xaxis]
    ax.set_xticks(xaxis+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)

        # save the plot to file
    plt.savefig('{}/2J{}.png'.format(TWOJROOT, movie_title.replace(' ', '')))

def train(dataset):
    """
    listOfDictionaries -> dictionaryOfDictionaries

    dataset: each dictionary in the dataset has keys
        title
        year
        episode
        summary
        ... (etc.)

    Outputs a dictionary with key, value pairs:
        year: {word1:P(word1|year), ..., wordn: P(wordn|year)}

    P(wordi|year) is num_occurences(wordi)/total_words(year)
    """
    probability_dict = {year: {} for year in xrange(1930, 2020, 10)}
    total_words_in_year = {year: 0.0 for year in xrange(1930, 2020, 10)}

    ## Store Count(word|year) for all the words, years. If word doesn't show up
    ## in probability_dict[year], that means word didnt show up in year
    len_dataset = len(dataset)
    for index, movie in enumerate(dataset):
        if index % 100 == 0:
            print "finished {} of {} movies in training phase".format(index, len_dataset)
        year = movie['year']
        for word in movie['summary']:
            if word not in probability_dict[year].keys():
                probability_dict[year][word] = 1.0
            else:
                probability_dict[year][word] += 1.0
            total_words_in_year[year] += 1.0

    ## Divide all those counts by the total words in each decade to turn them
    ## into probabilities
    for year in total_words_in_year.keys():
        prob_dict = probability_dict[year]
        total_words = total_words_in_year[year]
        probability_dict[year] = {key: value/total_words for key, value in prob_dict.iteritems()}

    return probability_dict

def calcProbDecadeGivenWords(prob_dict, year, words):
    """
    dictOfDictionaries int tuple -> float

    Given a training dictionary (see output of train function), a decade, and a tuple
    of words, outputs log(P(decade|word1>0, word2>0, ..., wordn>0)). Does so
    using the naive bayes assumption that P(wordi>0|year) is independent of 
    P(wordj>0|year) for i != j
    """
    import math
    log = lambda num: math.log(num, 10) # define log base 10

    # P(year|word1, ..., wordn) ~ P(year) * P(word1|year) * P(word2|year) ... P(wordn|year)
    # Since P(year) is the same for all years (we balanced the dataset), 
    # we don't include it
    log_probability = 0.0
    small_prob = 0.0001 # dirichlet prior
    word_sets = {year: set(prob_dict[year].keys()) for year in xrange(1930, 2020, 10)}
    for word in words:
        if word in word_sets[year]:
            log_prob = log(max(prob_dict[year][word], small_prob))
            log_probability += log_prob
        else:
            log_probability += log(small_prob)

    return log_probability

def predictYear(prob_dict, words, year_range=range(1930,2020,10)):
    """
    dictOfDictionaries tupleOfStrings list -> int

    Given a probability dictionary (see output of train), a tuple of 
    words to condition upon, and a range of years to test for, returns
    the most likely decade from which a movie with the given words is
    """
    f = lambda year: calcProbDecadeGivenWords(prob_dict, year, words)
    predicted = max(year_range, key=f) # argmax

    return predicted

def calculateAccuracy(prob_dict, test_data):
    """
    dictOfDictionaries listOfDictionaries -> float

    Given a probability dict (see output of train function) and a list of
    movie dictionaries, calculates the accuracy of a Naive Bayes Classifier
    on that test_data
    """
    num_correct = 0.0
    total = float(len(test_data))
    for index, movie in enumerate(test_data):
        if index % 100 == 0:
            print "on movie {} of {} in calculateAccuracy".format(index, total)
        words = tuple(set(movie['summary'])) # extract summary,remove duplicate words
        predicted = predictYear(prob_dict, words)
        true = movie['year']
        if predicted == true:
            num_correct += 1.0

    return num_correct/total
def plotHistForCMC(xaxis, yaxis, ylabel, xticks, title, saveLoc):
    """
    tuple tuple string string listOfStrings string -> None

    xaxis: a tuple of length n indicating the values along the x axis
    yaxis: a tuple of length n indicating the values along the y axis
    ylabel: the label for the y axis
    xticks: labels for the x axis
    title: title for the plot
    saveLoc: filepath to which to save the plot 

    Helper function for CMC. Allows us to test the plotting part seperately
    """
    import numpy as np
    import matplotlib.pyplot as plt

    ## Plot the CMC as a histogram
        # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

        # Format the data
    width = 1
    xaxis = np.array(xaxis)
    yaxis = yaxis
    rects = ax.bar(xaxis, yaxis, width, color='green')

        # axes and labels
    ax.set_xlim(min(xaxis)-width, max(xaxis)+2*width)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    xTickMarks = xticks
    ax.set_xticks(xaxis + 0.5 *width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=0, fontsize=10, horizontalalignment='center')

        # save the plot to file
    plt.savefig(saveLoc)

def plotCMC(krange, prob_dict, test_data):
    """
    tuple dictOfDictionaries listOfDictionaries -> None

    Given a range of k values (expected is (1, 2, 3, 4, 5, 6, 7, 8, 9) since
        we have 9 decades we are considering), a probability dict 
        (see output of train function), and a list of movie dictionaries, 
        plots a cumulative match curve, where f(x) is the percentage of movies 
        that were in our classifier's top x guesses. Saves plot to file
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from CONFIG import TWOLPLOT

    # Construct a dictionary with key, value pairs
    # movie: sorted_guesses
    # e.g: 'Finding Nemo': (2010, 2000, 1980, 1970, 1990, 1960, 1950, 1940, 1930)
    movie_dict = {} # holds sorted guesses for each movie
    num_iters = len(test_data)
    for index, movie in enumerate(test_data):
        if index % 100 == 0:
            print "movie {} of {} in plotCMC".format(index, num_iters)
        year_dict = {} # holds probabilities for each year
        words = tuple(set(movie['summary']))
        for year in xrange(1930, 2020, 10):
            year_dict[year] = calcProbDecadeGivenWords(prob_dict, year, words)
        sorted_guesses = sorted(year_dict.keys(), key=year_dict.__getitem__, 
                                reverse=True)
        movie_dict[movie['title']] = {'guesses': sorted_guesses, 'year': movie['year']}

    ## Plot the CMC as a histogram
    xaxis = krange                                                    # set axis
    total_movies = float(len(movie_dict))                             # set yaxis
    yaxis = []
    for x in xaxis:
        total_correct = float(len([title for title in movie_dict.iterkeys() if 
                 movie_dict[title]['year'] in movie_dict[title]['guesses'][0:x]]))
        freq_correct = total_correct / total_movies
        print "x, total correct, freq_correct: {}, {}. {}".format(x, total_correct, freq_correct)
        yaxis.append(freq_correct)
    ylabel = 'Percent of Movies with correct decade in top x guesses' # set ylabel
    xticks = ['{}'.format(k) for k in xaxis]                           # set xticks
    title = 'Cumulative Match Curve for k = {} to {}'.format(krange[0], krange[-1]) # set title
    saveLoc = TWOLPLOT                                                # set savelocation
    plotHistForCMC(xaxis, yaxis, ylabel, xticks, title, saveLoc)      # plot that biatch

def plotMatrixForCM(matrix, title, saveLoc):
    """
    numpy.matrix string string -> None

    matrix: an nxn numpy.matrix to plot
    title: the plot's title
    saveLoc: filePath to which to save the matrix

    Helper function for plotConfusionMatrix,to allow for independent testing
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_title(title)

    xTickMarks = range(1930, 2020, 10)
    ax.set_xticks(np.array(range(len(matrix[0:]))))
    ax.set_xticklabels(xTickMarks)

    yTickMarks = range(1930, 2020, 10)
    ax.set_yticks(np.array(range(len(matrix))))
    ax.set_yticklabels(yTickMarks)

    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.savefig(saveLoc)

def plotConfusionMatrix(prob_dict, test_data):
    """
    dictOfDictionaries listOfDictionaries -> None

    Given a probability dict (see output of train function) and a list of
    movie dictionaries, constructs and plots a confusion matrix. 

    Cell i, j of the confusion matrix is the number of times movies with decade
    D_i were classified as decade D_j
    """
    from CONFIG import TWOMPLOT
    import numpy as np
    import matplotlib.pylab as plt

    ## Make the matrix
    matrix = np.matrix([[0 for j in range(9)] for i in range(9)])
    years = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    num_iters = len(test_data)
    for index, movie in enumerate(test_data):
        if index % 100 == 0:
            print "finished {} of {} in plotConfusionMatrix".format(index, num_iters)
        words = tuple(set(movie['summary']))
        predicted_year = predictYear(prob_dict, words)
        true_year = movie['year']
        i, j = years.index(true_year), years.index(predicted_year)
        matrix[i, j] += 1

    ## Plot the matrix
    title = 'Confusion Matrix'
    plotMatrixForCM(matrix, title, TWOMPLOT)

def probWordInDecade(probability_dict, word, decade):
    """
    dictOfDictionaries string int -> float

    Given a probability_dict (see output of train function in problem2.py), a
    word and a decade, find the probability that the count of word is nonzero
    in that decade
    """
    import math

    small_prob = 0.0001
    log = lambda p: math.log(p, 10)
    if word in probability_dict[decade].keys():
        return float(log(probability_dict[decade][word]))
    return float(log(small_prob))

def findXMostInformativeWords(probability_dict, decade, x=10):
    """
    dictOfDictionaries int int -> tupleOfStrings

    Given a decade, a probability dict (see output of train function in 
        problem2.py) and an int x, finds the x most informative words for that decade, 
        as defined by 

        argmax_(w in words){f(w, decade)} where

        f(w, decade) = P(Xw > 0|Y=decade)
                       ------------------
                       min_y(P(X_w>0|Y=y))
    """
    words_in_decade = probability_dict[decade].keys()
    word_informativeness = {}

    for word in words_in_decade:
        pDecadeShort = lambda decade: probWordInDecade(probability_dict, word, decade)
        numerator = probWordInDecade(probability_dict, word, decade)
        denominator = min(range(1930, 2020, 10), key=pDecadeShort)
        word_informativeness[word] = numerator/denominator

    top_x = sorted(word_informativeness.keys(), key=word_informativeness.__getitem__, 
        reverse=True)[0:x]

    return tuple(top_x)

if __name__ == '__main__':
    from CONFIG import MOVIES, TWOAPLOT, TWOBPLOT, TWOCPLOT, TWODPLOT, \
        TWOEPLOT, TWOFPLOT, TWOGPLOT
    from parse import load_all_movies

    # Parsing movies list
    print "reading in movie data"
    movies = formatData(list(load_all_movies(MOVIES)))
    for movie in movies:
        if "nem" in movie['title']:
            print movie['title']

    # PLOT PMFS (unbalanced) 2a - 2d
    # Plot PMF of P(Y) (2a)
    meta = {'title': 'PMF for P(Y) across entire dataset',
            'save': TWOAPLOT }
    print "plotting {}".format(meta['title'])
    plotPMF(movies, meta)

    # Plot PMF of P(Y|"radio" > 0) (2b)
    meta = {'title': 'PMF for P(Y|"radio" > 0) across entire dataset', 
            'save': TWOBPLOT }
    print "plotting {}".format(meta['title'])
    plotPMF(movies, meta, condition=lambda summary: 'radio' in summary) 

    # Plot PMF of P(Y|"beaver" > 0) (2c)
    meta = {'title': 'PMF for P(Y|"beaver" > 0) across entire dataset', 
            'save': TWOCPLOT }
    print "plotting {}".format(meta['title'])
    plotPMF(movies, meta, condition=lambda summary: 'beaver' in summary) 

    # Plot PMF of P(Y|"the" > 0) (2d)
    meta = {'title': 'PMF for P(Y|"the" > 0) across entire dataset', 
            'save': TWODPLOT }
    print "plotting {}".format(meta['title'])
    plotPMF(movies, meta, condition=lambda summary: 'the' in summary) 

    ## PLOT PMFS (unbalanced) 2e - h
    ## Balance the dataset and replot
    print "balancing movie data"
    balanced_movies = balanceDataset(movies)

    # Plot PMF of P(Y|"radio" > 0) (2e)
    meta = {'title': 'PMF for P(Y|"radio" > 0) across balanced dataset', 
            'save': TWOEPLOT }
    print "plotting {}".format(meta['title'])
    plotPMF(balanced_movies, meta, condition=lambda summary: 'radio' in summary) 

    # Plot PMF of P(Y|"beaver" > 0) (2f)
    meta = {'title': 'PMF for P(Y|"beaver" > 0) across balanced dataset', 
            'save': TWOFPLOT }
    print "plotting {}".format(meta['title'])
    plotPMF(balanced_movies, meta, condition=lambda summary: 'beaver' in summary) 

    # Plot PMF of P(Y|"the" > 0) (2g)
    meta = {'title': 'PMF for P(Y|"the" > 0) across balanced dataset', 
            'save': TWOGPLOT }
    print "plotting {}".format(meta['title'])
    plotPMF(balanced_movies, meta, condition=lambda summary: 'the' in summary) 

    # Seperate the dataset into training and testing data
    # We can be sure that the distribution of years in training and test sets
    # are the same because balanced_movies has 6000 1930 movies, then 6000 1940 movies, etc.
    train_movies = [balanced_movies[i] for i in range(0, len(balanced_movies), 2)]
    test_movies = [balanced_movies[i] for i in range(1, len(balanced_movies), 2)]
    print "len train movies: {}".format(len(train_movies))
    print "len test movies: {}".format(len(test_movies))

    ## Train
    print "training"
    probability_dict = train(train_movies)

    # Make figures for  movies
    finding_nemo_summary = format('After his son is captured in the Great Barrier Reef' +\
        'and taken to Sydney, a timid clownfish sets out on a journey to bring him home.', 
        norepeats=True)
    the_matrix_summary = format('A computer hacker learns from mysterious' +\
     'rebels about the true nature of his reality and his role in the war' +\
      'against its controllers.', norepeats=True)
    gone_with_the_wind_summary = format('A manipulative Southern belle carries' +\
     'on a turbulent affair with a blockade runner during the American Civil War.', 
        norepeats=True)
    harry_potter_summary = format('Harry finds himself mysteriously selected' +\
    ' as an under-aged competitor in a dangerous tournament between three' +\
    ' schools of magic.', norepeats=True)
    avatar_summary = format('A paraplegic Marine dispatched to the moon' +\
     'Pandora on a unique mission becomes torn between following his orders' +\
     ' and protecting the world he feels is his home.', norepeats=True)
    summaries = (finding_nemo_summary, the_matrix_summary, 
        gone_with_the_wind_summary, harry_potter_summary, avatar_summary)
    titles = ('Finding Nemo', 'The Matrix', 'Gone with the Wind', 
        'Harry Potter and the Goblet of Fire', 'Avatar')
    decades = (2000, 1990, 1930, 2000, 2000)
    print "finding nemo summary: {}".format(finding_nemo_summary)
    for title, summary , decade in zip(titles, summaries, decades):
        print "plotting histogram for {}".format(title)
        print "summary for that title: {}".format(summary)
        print "decade for that title: {}".format(decade)
        plotHistogramForMovie(probability_dict, title, summary, decade)

    # ## Calculate accuracy of Naive Bayes Classifer
    print "calculating accuracy"
    from CONFIG import TWOOUTPUT
    accuracy = calculateAccuracy(probability_dict, test_movies)
    with open(TWOOUTPUT, "a+") as outputFile:
        outputFile.write('\n')
        outputFile.write("Accuracy of Naive Bayes Classifier: {}%".format(accuracy))

    # ## Plot the CMC for k = 1 to 9
    plotCMC((1, 2, 3, 4, 5, 6, 7, 8, 9), probability_dict, test_movies)

    ## Plot a conufusion matrix for the classifier
    plotConfusionMatrix(probability_dict, test_movies)

    #### PROBLEM 3
    print "calculating top ten words"
    from CONFIG import THREEOUTPUT

    # 3a: find ten most informative words from each decade
    for decade in xrange(1930, 2020, 10):
        print "calculating top ten words for {}".format(decade)
        x = 10
        top_ten_words = findXMostInformativeWords(probability_dict, decade, x=x)
        with open(THREEOUTPUT, "a+") as outputFile:
            outputFile.write('\n')
            outputFile.write("Top {} most informative words in {}: {}".format(x, decade, top_ten_words))

    ## 3b: Strip top 100 most informative words from each decade and calculate accuracy
    for decade in xrange(1930, 2020, 10):
        print "stripping top 100 most informative words from {}".format(decade)
        top_100_words = findXMostInformativeWords(probability_dict, decade, x=100)
        for word in top_100_words:
            del probability_dict[decade][word]
    accuracy = calculateAccuracy(probability_dict, test_movies)
    with open(THREEOUTPUT, "a+") as outputFile:
        outputFile.write('\n')
        outputFile.write("Accuracy of Naive Bayes Classifier with top 100 most" +\
                         " informativewords from each decade removed: {}%".format(accuracy))
