
#Name:Pawan Khadka
#Date:05/06/2021
#Program:Question Answering



#import libraries
import nltk
#nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from lemminflect import getInflection
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree as Tree
import spacy
import sys
nlp1 = spacy.load("en_core_web_sm",disable=["tagger","parser","lemmatizer"])
nlp = StanfordCoreNLP('stanford-corenlp-4.2.1',lang='en')

#this function generate tokenized sentences for the query and the article

def dataset_creation(text):
  #senetences containing words like e.g., i.e. were being treated as end of sentences
  #so,the following words were added to the abbreviation type 
  abbreviations = ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e','e.g']
  tokenize_sentence = nltk.data.load('tokenizers/punkt/english.pickle')
  tokenize_sentence._params.abbrev_types.update(abbreviations)
  tokenized_sentence=tokenize_sentence.tokenize(text)
  return tokenized_sentence

#this fucntion will calculate td-idf and return a pandas dataframe containing all the values 

def tf_idf_dataframe(dataset1,dataset2):

  vectorizer = TfidfVectorizer(input=dataset1, analyzer='word', ngram_range=(1,1),
                     min_df = 0, stop_words=None,lowercase=True)
  #fit_transform for the documents
  tfidf_matrix_1 =  vectorizer.fit_transform(dataset1)
  #transform for the query
  #print("1st matrix")
  #print(tfidf_matrix_1)
  tfidf_matrix_2 = vectorizer.transform([dataset2])
  #print("2nd matrix")
  #print(tfidf_matrix_2)

   # Format the TF-IDF table into the pd.DataFrame format.
  vocab = vectorizer.get_feature_names()

  documents_tfidf_lol_1 = [{word:tfidf_value for word, tfidf_value in zip(vocab, sent)} 
                       for sent in tfidf_matrix_1.toarray()]
  documents_tfidf_1 = pd.DataFrame(documents_tfidf_lol_1)
  documents_tfidf_1.fillna(0, inplace=True)

  documents_tfidf_lol_2 = [{word:tfidf_value for word, tfidf_value in zip(vocab, sent)} 
                       for sent in tfidf_matrix_2.toarray()]
  documents_tfidf_2 = pd.DataFrame(documents_tfidf_lol_2)
  documents_tfidf_2.fillna(0, inplace=True)
  documents_tfidf_2.loc[:, (documents_tfidf_2 != 0).any(axis=0)]


  return documents_tfidf_1,documents_tfidf_2

#given two dataframe(1:document tfidf,2:query tfidf) and this function calculate similarity between query and each doc and return
#the result in the form of a list

def score_per_doc(df_doc,df_query):

  #result will hold the scores for each document(sentence in this case)
  result=[0]*len(df_doc)

  #the dataframe for doc, and query filled with tfidf values are used to compute similarity
  for row in range(len(df_doc)) :
    for column in df_query:
      first_term=df_query[column][0]
      second_term=0
      try:
        second_term= df_doc.loc[row,column]
        result[row]+=first_term*second_term
      except:
        result[row]+=0
  return result

#this function  calls necessary functions and returns output as Matching sentence(if any) and the score
def query(dataset1,dataset2):

  #getting the tfidf for documents and query
  dataframe_doc,dataframe_query=tf_idf_dataframe(dataset1,dataset2)

  #print(dataframe_query)
  
  #getting the result,selecting the max score and the document with max score
  result=score_per_doc(dataframe_doc,dataframe_query)
  max_score=max(result)
  #print("max_score")
  #print(max_score)
  max_index=result.index(max_score)
  
  #if clasue is executed if no match is found(i.e max score=0), otherwise the else clasue is executed
  if max_score==0:
    matching_string="No match"
    return matching_string,max_score
  else:
    matching_string=dataset1[max_index]
    return matching_string,max_score

#returns the verb,label from the question

def get_verb(question):
  #print(question)
  if "%" in question:
    question= question.replace("%","percentage")
  try:
    tree_str=nlp.parse(question)
    tree=Tree.fromstring(str(tree_str))
  except:
    return "not found","not found"
  vb=["VB","VBP","VBD"]
  for subtree in tree[0]:
    if subtree.label()=="VP":
      for t in subtree:
        if t.label() in vb:
          return t.leaves(),t.label()
        else:
          if t.label()=="VP":
            for t_f in t:
              if t_f.label() in vb:
                return t_f.leaves(),t_f.label()
  return "not found","not found"




#this function checks for the negative word in the sentence;returns positive,negative
def check_for_negative(sentence):
  if "%" in sentence:
    sentence=sentence.replace("%","percentage")
  try:
    tree_str=nlp.parse(sentence)
    tree=Tree.fromstring(str(tree_str))
  except:
    return "could not parse"
  for subtree in tree[0]:
    if subtree.label()=="VP":
      for t in subtree:
        if t.label()=="RB":
          return "negative"
  return "positive"

#this returns lists of entity with tag and list of terms

def check_for_entities(sentence):
  entity_tag_list=[]
  entities=[]
  sen= nlp1(sentence)
  for entity in sen.ents:
    entity_tag_list.append(entity.label_)
    entities.append(str(entity))
  return entity_tag_list,entities

#this function returns the adjective from the sentence
#this function returns a list of adjectives from a given sentence/question
def check_for_adjective(sentence):
  if "%" in sentence:
    sentence=sentence.replace("%","percentage")
  adjective_pos=['JJ','JJS','JJR']
  adjective=[]
  try:
    tree_str=nlp.parse(sentence)
    tree=Tree.fromstring(str(tree_str))
  except:
    return "could not parse"
  for subtree in tree[0]:
    if subtree.label()=="VP":
      for t in subtree:
        if t.label()=="ADJP":
          for t_final in t:
            if t_final.label()=="JJ":
              adjective.append(t_final.leaves())
          return adjective
        if t.label()=="NP":
          for t_f in t:
            if t_f.label()=="ADJP":
              for t_f_1 in t_f:
                if t_f_1.label() in adjective_pos:
                  adjective.append(t_f_1.leaves())
              return adjective
            if t_f.label() in adjective_pos:
              adjective.append(t_f.leaves())
          return adjective
    if subtree.label()=="NP":
      for t in subtree:
        #print(t)
        if t.label() in adjective_pos:
          adjective.append(t.leaves())
  return adjective

#returns True/False

def binary_questions(question,matching_sentence):

  #check for presence of negatives,"not"-RB
  question_RB=check_for_negative(question)
  matching_sentence_RB=check_for_negative(matching_sentence)
  if question_RB=="could not parse" or matching_sentence_RB=="could not parse":
    return "Not found"
  if question_RB=="negative" and question_RB=="positive":
    return "False"
  elif question_RB=="positive" and question_RB=="negative":
    return "False"

  #check for match in entities between matching sentence and the query
  question_entity_tag,question_entities=check_for_entities(question)
  matching_sentence_entity_tag,matching_sentence_entities=check_for_entities(matching_sentence)
  index_question=0
  for tag in question_entity_tag:
    try:
      index_sentence= matching_sentence_entity_tag.index(tag)
    except:
      return "False"
    
    if question_entities[index_question] == matching_sentence_entities[index_sentence]:
      index_question=index_question+1
    else:
      return "False"

  #need to check for the similarity of the adjectives:???
  #dependent/independent
  adjective_from_question=check_for_adjective(question)
  adjective_from_matching_sentence=check_for_adjective(matching_sentence)

  if adjective_from_matching_sentence=="could not parse" or adjective_from_question=="could not parse":
    return "Not found"
  
  #the following loop checks for the presence of adjcetives 
  for adjective_word in adjective_from_matching_sentence:
    if adjective_word not in adjective_from_question:
      return "False"
  return "TRUE"



#given entity type,matching sentence from the article and the query, this function returns the expected answer

def answer_phrase(entity_type,sentence,question):
  doc=nlp1(sentence.lower())
  #print(entity_type)
  #print(sentence)
  #print(question)
  #looping over the Named entity in the sentence
  for t in (doc.ents):
    #if expected ner tag is present in the sentence,the if clasue is executed
    if (t.label_ in entity_type):
      #this if clasue makes sure we do not return the entity from the question
      #eg: sentence:John married Sally.question:Who did John marry?, ner tagger finds John first, which is not the expected answer
      if str(t) not in question.lower():
        return (str(t).capitalize())
  return ("Not Found")

#def find_answer(filename,question)
def find_answer(article_dataset,question_dataset):

  question=question_dataset.lower()
  #print(question)

#convert question to sentence: Did Pawan play football? ,Pawan played football.
  if question.split()[0].lower()=="did":
    verb,label=get_verb(question)
    if verb!="not found":
      changed_verb=getInflection(verb[0],tag='VBD')[0]
      question=question.replace("did","")
      question=question.replace("?",".")
      question=question.replace(verb[0],changed_verb)

#convert question to sentence: Does Pawan play football? ,Pawan plays football.
  if question.split()[0].lower()=="does":
    verb,label=get_verb(question)
    if verb!="not found":
      changed_verb=getInflection(verb[0],tag='VBZ')[0]
      question=question.replace("does","")
      question=question.replace("?",".")
      question=question.replace(verb[0],changed_verb)


  #get the sentence with highest similarity to the question using cosine similarity
  matching_sentence,max_score=query(article_dataset,question)

  #print(matching_sentence)

  #if matching sentence is not found, the following if clause returns "False "
  
  if(matching_sentence=="No match"):
    return ("False")
  
  #Different WH question expect different answer type,[WHERE expects location entity
  #WHEN expects DATE/TIME entity,WHO excepts PER/ORG entity]
  if question.split()[0].lower()=="where":
    returned_answer=answer_phrase(["LOC","GPE"],matching_sentence,question)
    return (returned_answer)
  elif question.split()[0].lower()=="when":
    returned_answer=answer_phrase(["DATE","TIME"],matching_sentence,question)
    return (returned_answer)
  elif question.split()[0].lower()=="who" or question.split()[0].lower()=="whom"or question.split()[0].lower()=="whose":
    returned_answer=answer_phrase(["PERSON","ORG"],matching_sentence,question)
    return (returned_answer)
  elif question.split()[0].lower()=="how" and question.split()[1].lower()=="many":
    returned_answer=answer_phrase(["CARDINAL"],matching_sentence,question)
    return (returned_answer)
  elif question.split()[0].lower()=="how" and question.split()[1].lower()=="much":
    returned_answer=answer_phrase(["MONEY","QUANTITY"],matching_sentence,question)
    return (returned_answer)
  elif question.split()[0].lower()=="what":
    if "\n" in matching_sentence:
      remove_index=matching_sentence.count("\n")
      return matching_sentence.split("\n")[remove_index]
    else:
      return (matching_sentence)
  else:
    #print(max_score)
    #print(matching_sentence)
    if max_score>=1:
      return ("True")
    else:
      returned_answer=binary_questions(question,matching_sentence)
      return (returned_answer)

def main():

  #reading the arguments from the command line
  article_filename=sys.argv[1]
  question_filename=sys.argv[2]
  #article_filename="/content/a8.txt"
  #question_filename="/content/a8-question.txt"


  #reading the article.txt file and creating tokenized sentences__ which is a collection of tokenized sentences
  article_file=open(article_filename)
  sentences=article_file.read()
  #creating dataset for article_sentences
  dataset1=dataset_creation(sentences)



  #reading the question.txt file and creating tokenized sentences
  question_file=open(question_filename)
  questions=question_file.read()
  #creating dataset for article_sentences_ which is a collection of tokenized sentences
  dataset2=dataset_creation(questions)


  #this list containes all the answers produced
  list_of_answers=[]

  #getting answer for each question 
  for question in dataset2:
    list_of_answers.append(find_answer(dataset1,question))

  #printing the result/produced answers to the console
  for answer in list_of_answers:
    print(answer)


if __name__ == "__main__":
  main()
