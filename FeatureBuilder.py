import os
import nltk
import numpy as np
import spacy
import pickle
from names_dataset import NameDataset
from gensim.models import KeyedVectors
FEATURES=[
	"pos",
	"bio",
	"is_title",
	"head",
	"norm",
	"prefix",
	"shape",
	"suffix",
	"isFirstName",
	"isLastName"
	]
nlp=spacy.load('en')
m=NameDataset()
word2vec = KeyedVectors.load_word2vec_format("/media/shivesh/Acer/Users/shive/Downloads/glove.6B/word2vec.6B.100d.txt", binary=False)
def setup():
	file=open("CONLL_train_dev.pos-chunk-name","r+")
	sentences=[]
	pos_list=[]
	bio_list=[]
	name_list=[]
	pos=[]
	bio=[]
	name=[]
	sentence=[]
	words=[]
	strings=[]
	s=""
	for line in file:
		if(line!='\n'):
			line=line.strip().split('\t')
			sentence.append(line[0])
			s=s+line[0]+" "
			pos.append(line[1])
			bio.append(line[2])
			name.append(line[3])
			words.append(line[0])
		elif(line=='\n'):
			s=s[:-1]
			strings.append(s)
			s=""
			sentences.append(sentence)
			pos_list.append(pos)
			bio_list.append(bio)
			name_list.append(name)
			pos=[]
			bio=[]
			name=[]
			sentence=[]
			words.append(line[0])
	file.close()
	pickle.dump(strings,open('fullset.pickle','wb'))	
	return sentences,pos_list,bio_list,name_list,words

def dev_setup():
	file=open("CONLL_test.pos-chunk","r+")
	strings=[]
	sentences=[]
	pos_list=[]
	bio_list=[]
	name_list=[]
	words=[]
	pos=[]
	bio=[]
	sentence=[]
	s=""
	for line in file:
		if(line!='\n'):
			line=line.strip().split('\t')
			s=s+line[0]+" "
			words.append(line[0])
			sentence.append(line[0])
			pos.append(line[1])
			bio.append(line[2])
		elif(line=='\n'):
			s=s[:-1]
			strings.append(s)
			s=""
			words.append(line[0])
			sentences.append(sentence)
			pos_list.append(pos)
			bio_list.append(bio)
			pos=[]
			bio=[]
			sentence=[]
	pickle.dump(strings,open('test_strings.pickle','wb'))
	file.close()
	return sentences,pos_list,bio_list,words


def is_country(word,sentence,pos_list,b,s,d,j):
	countries = get_countries()
	for code,country in countries:
		if word.lower() == country.lower():
			return "True"
	return "False"
def create_feature_dictionary():
	sentences,pos_list,bio_list,name_list,words_train = setup()
	dev_sentences,dev_pos_list,dev_bio_list,words=dev_setup()
	dictionary={}
	dictionary_dev={}
	strings = pickle.load(open("fullset.pickle","rb"))
	dev_strings=pickle.load(open("test_strings.pickle","rb"))
	doc_train=create_doc_list(strings)
	doc_dev=create_doc_list(dev_strings)
	word_count=0
	for i in range(len(sentences)):
		sentence=sentences[i]
		for j in range(len(sentence)):
			token = sentence[j]
			pos=pos_list[i][j]
			bio=bio_list[i][j]
			name=name_list[i][j] 
			dictionary[(token,word_count)]=(populate(token,pos,bio,name,sentence,pos_list[i],bio_list[i],strings[i],doc_train[i],j),j)
			word_count+=1
	word_count=0
	for i in range(len(dev_sentences)):
		sentence=dev_sentences[i]
		for j in range(len(sentence)):
			token = sentence[j]
			pos=dev_pos_list[i][j]
			bio=dev_bio_list[i][j]
			dictionary_dev[(token,word_count)]=(populate_dev(token,pos,bio,sentence,dev_pos_list[i],dev_bio_list[i],dev_strings[i],doc_dev[i],j),j)
			word_count+=1
	return dictionary,dictionary_dev,words,words_train,sentences,dev_sentences

def populate(word,pos,bio,name,sentence,pos_list,bio_list,strings,doc,j):
	features=[]
	features.append(pos)
	features.append(bio)
	global train_flag
	for f in FUNCTIONS:
		train_flag=True
		features.append(f(word,sentence,pos_list,bio_list,strings,doc,j))
	features.append(name)
	return features
def create_doc_list(sentences):
	docs=[]
	for sentence in sentences:
		docs.append(nlp(sentence))
	return docs

def populate_dev(word,pos,bio,sentence,pos_list,bio_list,strings,doc,j):
	features=[]
	features.append(pos)
	features.append(bio)
	for f in FUNCTIONS:
		train_flag=False
		features.append(f(word,sentence,pos_list,bio_list,strings,doc,j))
	return features

def make_feature_string(token,dictionary,is_test,sentence):
	features = dictionary[token][0]
	line=token[0]+"\t"+"curToken="+token[0]+"\t"
	excluded_features=["lemma","lower","leftedge","rightedge","suffix","orth","ispunct","idx","likeNumber","prev_name"]
	for i in range(len(FEATURES)):
		feature=FEATURES[i]
		line+=feature+"="+features[i]+"\t"
	index=dictionary[token][1]
	prev_token=""
	if index==0:
		prev_token="-B-"
		prev_features = ["-B-" for i in range(len(FEATURES))]
		name_tag="-B-"
		name_dev_tag="-B-"
	else:
		prev_token=sentence[index-1]
		prev_features = dictionary[(prev_token,token[1]-1)][0]
		name_tag=prev_features[len(prev_features)-1]
		name_dev_tag="@@"

	line+="prev_token="+prev_token+"\t"
		
	for i in range(len(FEATURES)):
		feature=FEATURES[i]
		if feature in excluded_features:
			continue
		line+="prev"+feature+"="+prev_features[i]+"\t"

	index=dictionary[token][1]
	grand_prev_token=""
	if index<2:
		grand_prev_token="-B-"
		grand_prev_features = ["-B-" for i in range(len(FEATURES))]
		name_prev_tag="-B-"
		name_dev_prev_tag="-B-"
	else:
		grand_prev_token=sentence[index-2]
		grand_prev_features = dictionary[(grand_prev_token,token[1]-2)][0]
		name_prev_tag=grand_prev_features[len(grand_prev_features)-1]
		name_dev_prev_tag="$$"
	line+="grandprev_token="+grand_prev_token+"\t"
	
	for i in range(len(FEATURES)):
		feature=FEATURES[i]
		if feature in excluded_features:
			continue
		line+="grandprev"+feature+"="+grand_prev_features[i]+"\t"
	
	index=dictionary[token][1]
	grand_grand_prev_token=""


	if index<3:
		grand_grand_prev_token="-B-"
		grand_grand_prev_features = ["-B-" for i in range(len(FEATURES))]
		name_dev_prev_prev_tag="-B-"
		name_prev_prev_tag="-B-"
	else:
		grand_grand_prev_token=sentence[index-3]
		grand_grand_prev_features = dictionary[(grand_grand_prev_token,token[1]-3)][0]
		name_dev_prev_prev_tag="~~"
		name_prev_prev_tag=grand_grand_prev_features[len(grand_grand_prev_features)-1]
	line+="grandgrandprev_token="+grand_grand_prev_token+"\t"
	
	for i in range(len(FEATURES)):
		feature=FEATURES[i]
		if feature in excluded_features:
			continue
		line+="grandgrandprev"+feature+"="+grand_grand_prev_features[i]+"\t"



	next_token=""
	index=dictionary[token][1]
	if index==len(sentence)-1:
		next_token="-E-"
		next_features = ["-E-" for i in range(len(FEATURES))]
	else:
		next_token=sentence[index+1]
		next_features = dictionary[(next_token,token[1]+1)][0]
	line+="next_token="+next_token+"\t"
	
	for i in range(len(FEATURES)):
		feature=FEATURES[i]
		if feature=="prev_name" or feature in excluded_features:
			continue
		line+="next"+feature+"="+next_features[i]+"\t"

	grand_next_token=""
	index=dictionary[token][1]
	if index>len(sentence)-3:
		grand_next_token="-E-"
		grand_next_features = ["-E-" for i in range(len(FEATURES))]
	else:
		grand_next_token=sentence[index+2]
		grand_next_features = dictionary[(grand_next_token,token[1]+2)][0]
	line+="grand_next_token="+grand_next_token+"\t"
	
	for i in range(len(FEATURES)):
		feature=FEATURES[i]
		if feature=="prev_name" or feature in excluded_features:
			continue
		line+="grandnext"+feature+"="+grand_next_features[i]+"\t"

	line=line+word_vectors(token[0].lower())+"\t"
	if(is_test):
		line+="prev_name="+name_tag+"\t"
		line+="grand_prev_name="+name_prev_tag+"\t"
		line+="grand_grand_prev_name="+name_prev_prev_tag+"\t"
		line+=features[len(features)-1]+"\t"
	else:
		line+="prev_name="+name_dev_tag+"\t"
		line+="grand_prev_name="+name_dev_prev_tag+"\t"
		line+="grand_grand_prev_name="+name_dev_prev_prev_tag+"\t"
	line=line[:-1]
	line+="\n"
	return line

def write_to_file():
	if('test.dat' in os.listdir()):
		os.remove('test.dat')
	if('dev.dat' in os.listdir()):
		os.remove('dev.dat')
	if('model' in os.listdir()):
		os.remove('model')
	if('response.name' in os.listdir()):
		os.remove('response.name')
	file=open("train.dat","a")
	dictionary,dev_dictionary,dev_words,words_train,sentences,dev_sentences=create_feature_dictionary()
	index=0
	words=0
	for token in words_train:
		if(token!='\n'):
			file_line = make_feature_string((token,words),dictionary,True,sentences[index])
			file.write(file_line)
			words+=1
		else:
			file.write('\n')
			index+=1
	file.close()
	file=open("test.dat","a")
	index=0
	words=0
	for token in dev_words:
		if(token!='\n'):
			file_line = make_feature_string((token,words),dev_dictionary,False,dev_sentences[index])
			file.write(file_line)
			words+=1
		else:
			file.write('\n')
			index+=1
	file.close()
def is_upper(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(word.isupper())
def like_upper(word,sentence,pos_list,bio_list,strings,doc,j):
	if  word[0].isupper():
			return "True"
	return "False"
def is_lower(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(word.islower())
def is_number(word,sentence,pos_list,bio_list,strings,doc,j):
	for c in word:
		if not c.isdigit():
			return "False"
	return "True"
def like_number(word,sentence,pos_list,bio_list,strings,doc,j):
	for c in word:
		if c.isdigit():
			return "True"
	return "False"
def idx(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(j)
def prefix(word,sentence,pos_list,bio_list,strings,doc,j):
	return word[0]
def suffix(word,sentence,pos_list,bio_list,strings,doc,j):
	return word[len(word)-3:]
def is_alpha(word,sentence,pos_list,bio_list,strings,doc,j):
	doc=nlp(strings)
	return str(doc[j].is_alpha)
def is_punct(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(doc[j].is_punct)
def is_titled(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(doc[j].is_title)
def head(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(doc[j].head)
def lemma(word,sentence,pos_list,bio_list,strings,doc,j):
	return doc[j].lemma_
def lower(word,sentence,pos_list,bio_list,strings,doc,j):
	return doc[j].lower_
def norm(word,sentence,pos_list,bio_list,strings,doc,j):
	return doc[j].norm_
def orth(word,sentence,pos_list,bio_list,strings,doc,j):
	return doc[j].orth_
def shape(word,sentence,pos_list,bio_list,strings,doc,j):
	return doc[j].shape_
def left_edges(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(doc[j].left_edge)
def right_edges(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(doc[j].right_edge)
def first_name(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(m.search_first_name(word))
def last_name(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(m.search_last_name(word))
def dep(word,sentence,pos_list,bio_list,strings,doc,j):
	return str(doc[j].dep_)
def word_vectors(word):
	feat=""
	if word in word2vec:
		vec = word2vec[word]
		i=0
		for v in vec:
			if i ==25:
				break
			feat = feat + str(i)+"d="+str(v)+"\t"
			i+=1
	else:
		for j in range(25):
			feat=feat+str(j)+"d="+"0"+"\t"
	feat=feat[:-1]
	return feat


FUNCTIONS=[
		is_titled,
		head,
		norm,
		prefix,
		shape,
		suffix,
		first_name,
		last_name
		]	
write_to_file()