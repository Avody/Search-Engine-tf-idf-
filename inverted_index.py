import mysql.connector
import json
import nltk
import os 
import numpy as np
import time 
import math
import xml.etree.ElementTree as ET
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from os.path import exists as file_exists




# a function that makes more readable the xml file
def prettify(element,indent= ''):
    queue = [(0,element)]
    while queue:
        level,element = queue.pop(0)
        children = [(level + 1, child) for child in list(element)]
        if children:
            element.text = '\n' + indent * (level+1)
        if queue:
            element.tail = '\n' + indent * queue[0][0]
        else:
            element.tail = '\n' + indent * (level-1)
        queue[0:0] = children


###### FETCH ARTICLES FROM DATABASE #######

#connect to database
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database = "crawled_news_database"

)

mycursor = mydb.cursor()

#retrieve data from database
mycursor.execute("SELECT Article FROM articles")
articles = mycursor.fetchall()

mycursor.execute("SELECT Title FROM articles")
titles = mycursor.fetchall()

mycursor.execute("SELECT Id FROM articles")
article_id = mycursor.fetchall()


mycursor.execute("Select Id,Title FROM articles")
art_id=mycursor.fetchall()



# list of closed class categories
closed_class_categories = ['CD','CC','DT','EX','IN','LS','MD','PDT','POS','PRP','PRP$','RP','TO','UH','WDT','WP','WP$','WRB']

number_of_articles = len(articles)

##### POS TAG WORDS FOR EVERY ARTICLE, SAVE POS TAGGED ARTICLES AND CREATION OF INVERTED FILE 

Pos_Taggs={}
inverted_file = {}
num_of_words_in_art={}
start=time.time()
count_1=0
print('ARTICLES TOKKENIZED LEMMATIZED AND POST TAGGED')
for i in range(len(articles)):
    count_1 +=1
    print('{} out of {}'.format(count_1,len(articles)))
    title = ''.join(titles[i])
    article = ''.join(articles[i])
    
    #tokenize each artile
    sentences=nltk.sent_tokenize(article)
    
    #pos tag each token in the article
    pos_tagged_words = []
    for sent in sentences:
        pos_tagged_words += nltk.pos_tag(nltk.word_tokenize(sent))
    
    #save pos tagged articles in a dictionary
    Pos_Taggs[title] = pos_tagged_words
    
        
    #remove stopwords
    no_stop_words_list = []
    
    num_of_words = 0
    for w in pos_tagged_words:
        if w[1] in closed_class_categories:
            continue
        else:
            num_of_words+=1
            no_stop_words_list.append(w[0])
    num_of_words_in_art[title]=num_of_words #save the number of words, after excluding stopwords, for each article
    
    ### remove punctuation
    symbols = "!\"#$%&()\'*+-, ./:;<=>?@[\]^_`{|}~\n"
    for symbol in symbols:
        no_stop_words_list = np.char.replace(no_stop_words_list,symbol,'')
    '''   
    #stemmimng words
    porter = PorterStemmer()
    stemmed_words = []
    for jjword in no_stop_words_list:
        jjword = jjword.lower()
        stem = porter.stem(jjword)
        stemmed_words.append(stem)
    '''
    
    #lemmatize words
    lemmatizer=WordNetLemmatizer()
    lemmatized_words = []
    for jjword in no_stop_words_list:
        jjword = jjword.lower()
        lemma = lemmatizer.lemmatize(jjword)
        lemmatized_words.append(lemma)
    
    #inverted file creation

    for word in lemmatized_words:
        if word not in inverted_file:
            inverted_file[word] = tuple([[article_id[i][0],1]])
        else:
            exists = 0
            Tuple_to_List = list(inverted_file[word])
            for j in range(len(Tuple_to_List)):
                if Tuple_to_List[j][0] == article_id[i][0]:
                    Tuple_to_List[j][1] +=1
                    exists = 1 
            if exists != 1:
                Tuple_to_List.append([article_id[i][0],1])

            inverted_file[word] = tuple(Tuple_to_List)
            
#save pos tagged article
if file_exists('Pos_Taggs.json')==False or os.path.getsize('Pos_Taggs.json') == 0:
    with open('Pos_Taggs.json','a+') as file:
        file.write(json.dumps(Pos_Taggs))
        file.close()
else:
    with open('Pos_Taggs.json','r') as file:
        file_data = json.load(file)
        file_data.update(Pos_Taggs)
    with open('Pos_Taggs.json','w') as file:
        json.dump(file_data,file)
        file.close()


#sort inverted file alphabetically
sorted_list = []
for i in inverted_file:    
    sorted_list.append(i)
sorted_list = sorted(sorted_list)

###### CREATION OF XML FILE WITH THE LEMMAS AND TF-IDF WEIGHTS ######

count= 0
start = time.time()
xml_doc = ET.Element('inverted_index')

    
for word in sorted_list:
    count+=1
    print('{} OUT OF {}'.format(count,len(sorted_list)))
    
    lemmas = ET.SubElement(xml_doc,'lemma',name=word)
    total_words_in_article = 0
    for cur_id in article_id:
        
        #how many times does the word appear in this article
        appears_in_the_doc = 0
        
        for term in range(len(inverted_file[word])):
            
            if inverted_file[word][term][0]==cur_id[0]:
                appears_in_the_doc = inverted_file[word][term][1]

        #total words in the document
        
        total_words_in_article = num_of_words_in_art[titles[cur_id[0]-1][0]]
                    
        
        tf = appears_in_the_doc / total_words_in_article
        
        
        #idf = log(total number of documents/number of documents where the term appears)
        idf = math.log(len(article_id)/len(inverted_file[word]))
        
        
        tf_idf = tf*idf
        if(tf_idf) != 0:
            ET.SubElement(lemmas,'document id = "{}" TF-IDF = "{}"'.format(cur_id[0],tf_idf))
   

prettify(xml_doc) 
tree = ET.ElementTree(xml_doc)
tree.write('Inverted_Index.xml',encoding='utf-8',xml_declaration=True)
end = time.time() 

print('Total time runned for xml file creation: {}'.format(end-start))

###### EVALUATION OF INVERTED INDEX WITH A "SEARCH ENGINE" #######

mytree=ET.parse('Inverted_Index.xml')
myroot = mytree.getroot()

query_tokens =[]
lemmatized_query_tokens=[]
article_weight = {}

query = input('Set your query: ')
query_tokens = nltk.word_tokenize(query)


for tok in query_tokens:
    tok = tok.lower()
    lem_tok = lemmatizer.lemmatize(tok) 
    lemmatized_query_tokens.append(lem_tok)
    

for quer_tok in lemmatized_query_tokens: 
    for lemma in myroot.findall('lemma'):
        if(lemma.attrib['name']==quer_tok):
            for doc in lemma:
                title_id = int(doc.attrib['id']) - 1
                article_title = art_id[title_id][1]
                if article_title in article_weight:
                    article_weight[article_title] += float(doc.attrib['TF-IDF'])
                else:
                    article_weight[article_title] = float(doc.attrib['TF-IDF'])
            

sorted_weights = sorted(article_weight.items(),key = lambda x:x[1],reverse=True)
for i in sorted_weights:
    print('\n'+i[0])