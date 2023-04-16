# Training Custom Named Entity Recognition Models

## Identifying Entities of interest in Movie Trivia Questions
## By: Barker French
<br>
<br>

## 1. Introduction

"Over 2.5 quintillion bytes of data are generated every day.  80% of it is unstructured" (Vasilis, 2022), and we, humans, can't 
consume and classify all of this data on our own.  However, with the help of computers, humans can hope to make sense of this data.
One method that can help to structure the data is named entity recognition (NER).  NER — sometimes referred to as entity 
chunking, extraction, or identification — is the task of identifying and categorizing key information (entities) in text (Marshall, 2019).  
These tags can then be used to order and classify data so that it can be retreived when needed. 

While many off the shelf NER models exist, these models classify text into general categories that can be relevant across many types of data. For example, spaCy's standard NER model includes the general categories of "person", "organization", "date", etc. While this text classification capability is powerful, it leaves gaps for important, specific business use cases in the law, medicine, customer relations, etc. These specific business use cases require a more specialized NER tool capable
of identifying entities of interest in each of the respective fields. 

This project centers on the the creation of a specialized NER model--based on conditional random fields (CRF)--that can identify entities of interest in movie trivia questions.  Though the focus of the project's model is movie trivia, the same techniques can be used to train NER models for application in law, medicine, customer relations, etc.  The  custom model's performance will then be compared against spaCy's pre-trained, mainstream NER model that is further trained on movie trivia question data. The custom NER tool I build will be heavily based on a custom NER discussed in Text Analytics with Python by Dipanjan Sarkar.

## 2. Research Questions

To focus the research, training, and building of the custom NER model, the following questions were posed.  At what performance level can a custom NER model built on a CRF be trained to work on specific data sets?  How well can spaCy's professionally built generic NER model be trained on the movie triviadata set, and how does spaCy's model perform compared to the custom CRF-based named entity recognition model?  Since the performance of both of these models depends in part on how they are trained, what kinds of techniques can be used to improve their performance?  Finally, the project includes a shallow dive into the workings of both a CRF NER model and spaCy's model.

## 3. Related Work

There is a large body of existing work investigating the application and the improvement of CRF models to the problem of NER.  Tran et al. (2017) developed "a method that combines active learning and self-learning to reduce the labeling effort for the named entity recognition task from tweet streams by using both machine-labeled and manually-labeled data...[showing] it can significantly imporve the performance of the [CRF] systems (Tran et al., 2017, p. 1)."  VeeraSekharReddy et al. (2022) found that when a "CRF classifier is integrated into an active learning-trained hybrid model...it can shorten the time it takes for the model to converge and cut down on the labor intensive cost of traditional approaches (VerraSekharReddy, 2022)."  Liu et al "proposed a novel NER system for tweets, which combines a KNN classifier with a CRF labeler under a semi-supervised learning framework...[showing] the effectiveness of [their] method (Lui 2011)."

## 4. Dataset

<p>The data used to train both the existing spaCy and the custom CRF NER model is a combination of two datasets--eng.bio and trivia.bio--that are a part of the Spoken Language Systems (SLS) project in MIT's Computer Science and Artificial Intelligence Laboratory.  Each dataset includes a list of movie trivia questions and an entity tag.  In training the models, a supervised learning task, the dependent target variable is the NER tag while the words in the trivia questions form the basis for the independent variable features.  Read on for more information on the dataset</p>

![](../images/Movie_trivia_data_exemplar2.JPG)

**Eng Dataset**

eng.bio is a 1.9mb dataset that consists of 12,218 simple movie questions over 124,177 rows, one word and one ner label per row. Each row in the dataset consists of an IOB2 tag (more on IOB2 below) and a word from a specific movie trivia question.  The boundary of each move trivia question in the file is identified by a blank return line appearing between the previous and subsequent questions.  The following target, dependent-variable, movie entities are identified in the eng.bio data: 'OTHER', 'ACTOR', 'YEAR', 'TITLE', 'GENRE', 'DIRECTOR', 'SONG', 'PLOT', 'REVIEW', 'CHARACTER', 'RATING', 'RATINGS_AVERAGE', 'TRAILER'.
<br>

**Trivia Dataset**

The second dataset, called trivia.bio, is 3.0mb, and consists of 9,769 complex movie questions over 197,858 rows. Each row in the dataset consists of an IOB2 tag (more on IOB2 below) and a word from a specific movie trivia question.  The boundary of each movie trivia question is identified by a blank return line appearing between the previous and subsequent questions.  The following target, dependent-variable, movie entities are identified in the trivia.bio data: 'Actor', 'Outside', 'Plot', 'Opinion', 'Award', 'Year', 'Genre', 'Origin', 'Director', 'Soundtrack', 'Relationship','Character_Name', 'Quote'.
<br>

**IOB2 Format**

IOB2 (Inside, Outside, Beginning) format "is a common tagging format for tagging tokens in a chunking task in computational linguistics... The I- prefix before a tag indicates that the tag is inside a chunk.  An O tag indicates that the token belongs to no chunk...The B-tag is used in the beginning of every chunk (Wikipedia 2022).  Each dataset has two columns: an IOB2 tag and the word that is being tagged.</p>
<br>


<p>The below table describes the specific target variables and shows the relationship between entities in the two datasets.</p>

|eng variable            |trivia variable  |description                      |
| ---------------------- | -----------| ---                                  |
|outside                 | outside    | untracked entity                     |
|actor                   |  actor     | person's name acting in the movie    |
|year                    |year      | two and four digit years            |      
| genre                  | genre      | movie's subject matter               |
| director               | director   | person's name in charge of movie     |
| song                   | soundtrack | song's in a movie                    |
| plot                   | plot       | movie's story line                   |
| review, ratings_average| opinion    | words describing the movie           |
| character              | character_name| person's name in movie story line |
| rating                 | na         | classification of movie appropriateness for different age groups
| title                  | na         | movie name
| trailer                | na         | identfies a word as preview or trailer|
| na                     | award      | recognition for acheivement in a given category
| na                     | origin     | source of inspiration for movie script |
| na                     | relationship| how movie is related to other movies in the same story line|
|

<br>

The data can be found here:

https://groups.csail.mit.edu/sls/downloads/movie/


## 5. Exploratory Data Analysis

<img align="right" src=../images/combined_dataset_entity_count2.JPG>

<p>Consisting of words and IOB2 labels, the data set appeared relatively straight-forward.  However, the owners of the data did not describe the datasets.  To better understand the datasets, the data were merged into a single set, additional columns were added to track question number, dataset source, and the IOB2 label was divided into it's bio label and it's entity label.  Exploratory data analysis was conducted.</p>

<img align="right" src=../images/movie_entity_count.JPG>
<p>In the "LOB2_Label_count" table, the data were aggregated by iob2_label to determine how many unique entities existed in the data.  The resulting table, "IOB2_Label Count" highlighted that a particular IOB2_label had multiple components.  For example, the ACTOR tag appears twice, once as B-ACTOR--the first descriptor of the actor name--and again as I-ACTOR--any additional name descriptor associated with the actor.  To quantify the number of instances of each entity in the data, multi-word entities were combined into single strings, and then displayed in a bar graph called the "Move Entity Count - Combined Data Set.</p>

<p>To confirm that the resulting, merged entities were meaninful, a sample of the merged entities was then displayed to confirm the meaningfulness of the combined entry.</p>

## 6. Pre-Processing

<p>Neither the CRF nor spaCy NER models used the same data format as was provided by the movie trivia data source.  Preprocessing was required to match the model's required data format and, for the CRF model, feature engineering was then conducted on the training data.  SpaCy's input data neither required nor allowed enrichmetn of the training data.</p>


### CRF Pre-Processing

<p>To prepare the data for the CRF model, the dataframe was converted into a list of sentences where each sentence was itself a list of tuples containing one word from the sentence, that word's POS tag, and that word's IOB2 tag.  </p>
<img align="left" src=../images/CRF_initial_dataset.JPG>

<br><br><br><br><br><br><br>
<p>Once in the list of sentences format, the data underwent feature engineering.  The following features were created for each word. </p>
 
| Feature          | Feature Description                                                |
| -----------------|----------------------------------------------                      |
| bias             | the bias for the model                                             |
| word.lower       | the lower case form of the word                                    |
| word[-3:]        | the last 3 letters fo the word                                     |
| word[-2:]        | the last 2 letters fo the word                                     |
| word.isupper     | boolean value - true if word is upper case else false              |
| word.istitle     | boolean value - true if word is title else false                   |
| word.isdigit     | boolean value - true if word is digit else false                   |
| postag           | part of speech tag                                                 |
| postag[:2]       | first two characters in the part of speech tag                     |
| BOS              | boolean value - tests if a word is the beginning of the sentence   |
| -1: word.lower   | prior word - lower case form                                       |
| -1: word.istitle | prior word - boolean value - true if word is title else false      |
| -1: word.isupper | prior word - boolean value - true if word is upper case else false |
| -1: postag       | prior word - part of speech tag                                    |
| -1: postag[:2]   | prior word - first two characters in the part of speech tag        |
| +1: word.lower   | next word - lower case form                                        |
| +1: word.istitle | next word - boolean value - true if word is title else false       |
| +1: word.isupper | next word - boolean value - true if word is upper case else false  |
| +1: postag       | next word - part of speech tag                                     |
| +1: postag[:2]   | next word - first two characters in the part of speech tag         |

<p>As a part of the feature creation the list of sentences was converted from its list of sentences that were structured as lists of tuples to one list of dictionaries, where each dictionary contained the feature enrichment of one word.  In this new format, there was no distinction between the ending of one sentence and the beginning of the next sentence.  Finally the data were split into a training and a test set in prepration for training the CRF model. 

  
### SpaCy Pre-Processing

<p>SpaCy's NER model requires a completely different format for model training.  The spaCy model requires that the movie trivia questions be configured as a list of 2 element tuples, where each tuple contains the movie trivia question as a string and a dictionary with one key called entities and value that is a list of 3 element tuples, where the first two elements are integers representing the location of an entity in the sentence string and the last element is a the IOB2_label.
<br>
<br>
  
<img align="left" src=../images/spaCy_sentence_data_format.JPG>

<br><br><br><br><br><br><br><br><br><br><br>
  
## 7. Models

  
### CRF Model
  The first NER model implemented was the conditional random field model identified by Dipanjan Sarkar in his book "Text Analytics with Python".  As Sarkar explains "the key point to remember ... is that NER is a sequence modeling problem at its core.  It is more related to the classification suite of problems, wherein we need a labeled dataset to train a classifier (Sarkar, 2019).
  
  
<br><br><br><br><br><br>
## Sources

Liu, X., Zhang S., Wei F., Zhou M. (2011). Recognizing Named Entities in Tweets. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies - Volume 1June 2011 Pages 359–367.

Marshall, Christopher. (2019 December 18).  What is named entity recognition (NER) and how can I use it?.
Medium. https://medium.com/mysuperai/what-is-named-entity-recognition-ner-and-how-can-i-use-it-2b68cf6f545d

Sienčnik, Scharolta Katharina. 2015. Adapting word2vec to Named Entity Recognition. In Proceedings of the 20th Nordic Conference of Computational Linguistics (NODALIDA 2015), pages 239–243, Vilnius, Lithuania. Linköping University Electronic Press, Sweden.

Tran, V.C., Nguyen N.T., Fujita H., Hoang, D.T. (2017). A combination of active learning and self-learn for named entity recognition
on Twitter using conditional random fields. Elsevier, 132, 179-187.

Vasilis, Theo. (2022 February 9). When Data Gets Too Big Why You Need Structured Data. apify.
https://blog.apify.com/when-data-gets-too-big-why-you-need-structured-data/

VeeraSekharReddy B., Rao, K. S., and Koppula, N., "Named Entity Recognition using CRF with Active Learning Algorithm in English Texts," 2022 6th International Conference on Electronics, Communication and Aerospace Technology, Coimbatore, India, 2022, pp. 1041-1044, doi: 10.1109/ICECA55336.2022.10009592.

Wikipedia. (2022, December 13). Inside-outside-beginning(tagging). https://owl.purdue.edu/owl/research_and_citation/apa_style/apa_formatting_and_style_guide/reference_list_electronic_sources.html 

