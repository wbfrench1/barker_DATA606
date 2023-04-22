


def word2features(l_sent: list, i) -> dict:

    '''   Description:  Takes a list of sentence tuples and an integer 
                        identifying the word the function will work on and
                        returns a dictionary containing the words in
                        each sentence and corresponding features.
                        
                        l_sents: is formatted as a list of tuples where each 
                                 tuple is a:
                                   word in the sentence, 
                                   a part of speech tag,
                                   and an NER tag

                                   (word:str, POS:str, NER:str)
        
          Returns:      a dictionary, where each dictionary contains
                        features for one word in one sentence.
    
    '''
    word = l_sent[i][0]
    postag = l_sent[i][1]

    features = {'bias': 1.0,
                'word.lower()': word.lower(),
                'word[-3:]': word[-3:],
                'word[-2:]': word[-2:],
                'word.isupper': word.isupper(),
                'word.istitle()': word.istitle(),
                'word.isdigit()': word.isdigit(),
                'postag': postag,
                'postag[:2]': postag[:2]
                }
    
    if i > 0:
        word1 = l_sent[i - 1][0]
        postag1 = l_sent[i - 1][1]
        features.update( {'-1:word.lower()': word1.lower(),
                          '-1:word.istitle()': word1.istitle(),
                          '-1:word.isupper': word1.isupper(),
                          '-1:postag': postag1,
                          '-1:postag[:2]': postag1[:2]
                          }
                        )
    
    else:
        features['BOS'] = True

    if i < len(l_sent) - 1:
        word1 = l_sent[i + 1][0]
        postag1 = l_sent[i + 1][1]
        features.update({'+1:word.lower()': word1.lower(),
                         '+1:word.istitle()': word1.istitle(),
                         '+1:word.isupper': word1.isupper(),
                         '+1:postag': postag1,
                         '+1:postag[:2]': postag1[:2]
                         }
                        )
    else:
        features['EOS'] = True
    
    return features




def sent2features(l_sent:list ) -> list:
    '''   Description:  Takes a sentence as a list of word tuples and returns a
                        list of dictionaries. The list contains tuples of
                        sentences, where each tuple is broken up into a word in
                        the sentence, a part of speech tag, and an NER tag.  
                        
                        l_sents: is formatted as a list of tuples where each 
                                 tuple contains a word in the sentence, the 
                                 word's part of speech tag, and an NER tag:  
                                 (word:str, POS:str, NER:str)
        
          Returns:      list of dictionaries of word features
    
    '''

    return [ word2features(l_sent, i) for i in range(len(l_sent))]




def sent2labels(l_sent: list) -> list:
    '''   Description:  Takes a list of sentence word tuples and returns the ner 
                        label from each word tuple. 
                        
                        l_sents is formatted as a list of tuples where each 
                                 tuple contains a word in the sentence, the 
                                 word's part of speech tag, and an NER tag:  
                                 (word:str, POS:str, NER:str)

          Returns:      list of dictionaries of word features
    
    '''

    return [label for token, postag, label in l_sent]




def word2features_w2v(l_sent: list, i, word2vec_model) -> dict:

    '''   Description:  Takes a sentence represented as a list of word tuples 
                        and an integer identifying the word tuple the function
                        will work on and returns a dictionary containing the 
                        words in each sentence and the corresponding features.

                        this function includes and additional feature: the 
                        word2vec vector for each word
                        
                        l_sent: is formatted as a list of tuples where each 
                                 tuple is a:
                                   word in the sentence, 
                                   a part of speech tag,
                                   and an NER tag

                                   (word:str, POS:str, NER:str)
        
          Returns:      a dictionary, where each dictionary contains
                        features for one word in one sentence.
    
    '''
    word = l_sent[i][0]
    postag = l_sent[i][1]

    features = {'bias': 1.0,
                'word.lower()': word.lower(),
                'word[-3:]': word[-3:],
                'word[-2:]': word[-2:],
                'word.isupper': word.isupper(),
                'word.istitle()': word.istitle(),
                'word.isdigit()': word.isdigit(),
                'postag': postag,
                'postag[:2]': postag[:2],
               }
    
    features.update( create_w2v_feature_dict(word2vec_model, word) )

    if i > 0:
        word1 = l_sent[i - 1][0]
        postag1 = l_sent[i - 1][1]
        features.update( {'-1:word.lower()': word1.lower(),
                          '-1:word.istitle()': word1.istitle(),
                          '-1:word.isupper': word1.isupper(),
                          '-1:postag': postag1,
                          '-1:postag[:2]': postag1[:2]
                          }
                        )
    
    else:
        features['BOS'] = True

    if i < len(l_sent) - 1:
        word1 = l_sent[i + 1][0]
        postag1 = l_sent[i + 1][1]
        features.update({'+1:word.lower()': word1.lower(),
                         '+1:word.istitle()': word1.istitle(),
                         '+1:word.isupper': word1.isupper(),
                         '+1:postag': postag1,
                         '+1:postag[:2]': postag1[:2]
                         }
                        )
    else:
        features['EOS'] = True
    
    return features




def sent2features_w2v(l_sent:list, word2vec_model ) -> list:
    '''   Description:  Takes a sentence as a list of word tuples and returns a
                        list of dictionaries. The list contains tuples of
                        sentences, where each tuple is broken up into a word in
                        the sentence, a part of speech tag, and an NER tag.  
                        
                        this function is updated to take the word2vec vector

                        l_sents: is formatted as a list of tuples where each 
                                 tuple contains a word in the sentence, the 
                                 word's part of speech tag, and an NER tag:  
                                 (word:str, POS:str, NER:str)
        
          Returns:      list of dictionaries of word features
    
    '''

    return [ word2features_w2v(l_sent, i, word2vec_model) for i in range(len(l_sent))]




def sent2labels(l_sent: list) -> list:
    '''   Description:  Takes a list of sentence word tuples and returns the ner 
                        label from each word tuple. 
                        
                        l_sents is formatted as a list of tuples where each 
                                 tuple contains a word in the sentence, the 
                                 word's part of speech tag, and an NER tag:  
                                 (word:str, POS:str, NER:str)

          Returns:      list of dictionaries of word features
    
    '''

    return [label for token, postag, label in l_sent]



def create_w2v_feature_dict (word2vec_model, word):
    '''   Description:  Takes a word2vec model and a word that is in the
                        model and returns a dictionary representings the 
                        word2vec vector, which can be added to the word's 
                        feature vector in the training data
                        

          Returns:      dictionary of elements in the word2vec vector
    
    '''

    dict_w2v = {}
    for sub_feat_num in range(len(word2vec_model.wv[word])):
        dict_w2v.update({'w2v_' + str(sub_feat_num): word2vec_model.wv[word][sub_feat_num]})
    return dict_w2v
