# PHASE I: PROJECT PROPOSAL & PLANNING
### Data 606 Capstone: Tuesday
### By: Barker French

---
<br>

**Topic**
<p>My project will evaluate the effectiveness of two Named Entity Recognition (NER) models in identifying entities of interest in a dataset of movie trivia questions. NER is an important area of study because it helps to bring order to and make use of the large volumes of unstructured data produced daily.  Help is needed in classifying unstructured data because unstructured data represents 80% of the data generated on a daily basis, and humans don't have the capacity to consume and classify all of this data without the help of computers.</p>
<br>

**Motivation/Techiniques**
<p>While many off the shelf NER models exist, these models classify text into general categories that are relevant across many types of data.  For example, spaCy's standard NER model includes the general categories of "person", "organization", "date", etc.  While this text classification capability is powerful, it leaves gaps for important, specific business use cases in areas like law, medicine, customer relations, etc.  These specific business use cases require a more specialized NER tool.  Keeping in mind the need to fulfill specific use case requirements like these, my project will center on building an NER tool capable of classifying movie trivia questions. I will then compare this custom model's performance against spaCy's mainstream NER model that I will train on movie trivia question data.  The custom NER tool I build will be heavily based on a custom NER discussed in Text Analytics with Python by Dipanjan Sarkar.</p>
<br>

**Research Questions/Metrics**
<p>In building and training these models.  The following questions will be answered:

 - How do the spaCy and Sarkar models work?
 - At what performance level can a spaCy's NER tool be trained to work on specific data sets?
 - At what performance level can a Sarkar's custom NER tool be trained to work on specific data sets?
 - How does the performance of spaCy's trained standard NER compare to the performance of the standard spaCy tool when it is used on standard documents?
 - What kinds of techniques can be used to improve a given model's performance?

Because at its core NER is a classification problem, the answers to the above  performance based questions will be based on several quantitative measures used in classification problems, including Precision, Recall, F-Score, and Accuracy.
</p>
<br>

**Dataset**
<p>The data that will be used to train both the existing and custom NER models comes from two datasets that are a part of MIT's Spoken Language Systems (SLS) project in MIT's Computer Science and Artificial Intelligence Laboratory.  The first dataset is called eng.bio.  The second dataset is called trivia.bio.  This Capstone project will combine these two labeled movie datasets, which are in IOB2 (Inside, Outside, Beginning) format. The IOB2 format "is a common tagging format for tagging tokens in a chunking task in computational linguistics... The I- prefix before a tag indicates that the tag is inside a chunk.  An O tag indicates that token belongs to no chunk...The B-tag is used in the beginning of every chunk (Wikipedia 2022).  Each dataset has two columns: an IOB2 tag and the word that is being tagged.</p>
<br>

***Eng Dataset***
<p>The first data set, eng.bio, is 1.9mb, and consists of 12,218 simple movie questions over 124,177 rows.  The following movie entities are identified in the data: 'OTHER', 'ACTOR', 'YEAR', 'TITLE', 'GENRE', 'DIRECTOR', 'SONG', 'PLOT', 'REVIEW', 'CHARACTER', 'RATING', 'RATINGS_AVERAGE', 'TRAILER'.</p>
<br>

***Trivia Dataset***
<p>The second dataset,called trivia.bio, is 3.0mb, and consists of 9,769 complex movie questions over 197,858 rows.  The following movie entities are identified in the data: 'Actor', 'Outside', 'Plot', 'Opinion', 'Award', 'Year', 'Genre', 'Origin', 'Director', 'Soundtrack', 'Relationship','Character_Name', 'Quote'.</p>
<br>

<p>The below table describes the variables and shows the relationship between entities in the two datasets.</p>

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

<p></p>

<br>

**Models**
<p>One of the goals of this project is to understand the workings of the spaCy NER model and a custom NER model described in Dipanjan Sarkar's Text Analytics with Python.  With that goal in mind, the below description seeks to provide a high-level, basic understanding of the two models.  The project itself will attempt to describe these models in more detail and to highlight areas of significance.
<br>
<br>

***SpaCy NER model***
<p>The SpaCy NER model is "a transition-based named entity recognition" tool.  "The enitity recognizer identifies non-overlapping labelled spans of tokens" (spaCy.io 2023).  According to Matthew Honnibal, one of spacCy's founders, "spaCy v2.0’s Named Entity Recognition system features a sophisticated word embedding strategy using subword features and “Bloom” embeddings, a deep convolutional neural network with residual connections, and a novel transition-based approach to named entity parsing. The system is designed to give a good balance of efficiency, accuracy and adaptability."

***Sarkar's Model***
<p>In describing his model, Sarkar notes that "NER is a sequence modeling problem at its core. It is... related to the classification suite of problems" (Sarkar pg. 545). Sarkar "develops [his] own NER [model] based on CRF's [Conditional Random Fields]...CRFs [are] an undirected graphical model whose nodes can be divided into exactly two disjoint sets X and Y, the observed and output variables, respectively; the conditional distribution p(Y|X) is then modeled" (Sakar pg.547).
<br>
<br>

**Outcome**
<p>At the project's conclusion, I will better understand NER models and their abilities.  In particular, I should understand how to build a custom NER model based on a conditional random field.  I should be able to quantify the ability of both existing and custom NER models to correctly classify data from a custom data set.  I should be able to identify techniques that can be used to improve an NER's performance and offer both quantitative and qualitative assessments of their efficacy.</p>
<br>
<br>
<br>
<br>

**References**

Wikipedia. (2022, December 13). Inside-outside-beginning(tagging). https://owl.purdue.edu/owl/research_and_citation/apa_style/apa_formatting_and_style_guide/reference_list_electronic_sources.html 

spacy.io. (2023, February 12). EntityRecognizer.
https://spacy.io/api/entityrecognizer 


Honnibal Matthew. (2023, February 12). spaCy's NER model: incremental parsing with Bloom embeddings and residual CNNs. spaCy. https://spacy.io/universe/project/video-spacys-ner-model


Sarkar Dipanjan. (2019). Text Analytics with Python. Apress
