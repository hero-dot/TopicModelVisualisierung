
# coding: utf-8

# # The LDA Topic Model

# This is the code from the Tutorial on the English Wikipedia from the gensim Homepage http://radimrehurek.com/gensim/wiki.html

# In[ ]:

import logging, gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load id-> word mapping (the dictionary)
id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt.bz2')
#load corpus iterator
mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')

print(mm)


# same as in the tutorial

# In[ ]:

#extract 100 LDA topics, using 1 pass
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100,update_every=1, chunksize=10000, passes=1)

#show ten topics
lda.print_topics(10)

#save the LDA_model
lda.save('wiki_en_LDA')


# In[ ]:



