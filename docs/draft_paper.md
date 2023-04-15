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
|year                   |year      | two and four digit years            |      
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

<img align="right" src=../images/eng_train.bio_entity_count2.JPG>
To better understand the datasets, exploratory data analysis was conducted.  The data in the eng.train.bio dataset was counted by each entity tag (see below).  The resulting table highlighted that a given entity could contain several words.  For example, the actor Harrison Ford would be reprsented as b-actor Harrison and i-actor Ford.  While the actor Madonna would be reprsented as i-actor Madonna.  











<br>
<br>
<br>
<br>
<br>
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

