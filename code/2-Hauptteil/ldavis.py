from gensim import corpora, models
import pyLDAvis.gensim
import pyLDAvis

dic = corpora.Dictionary.load('data/model/newsgroups.dict')
corp = corpora.MmCorpus('data/model/newsgroups.mm')
lda = models.ldamodel.LdaModel.load('data/model/newsgroups_50.model')

# Prepare the data for the visualization
newsgroup_data = pyLDAvis.gensim.prepare(lda, corp, dic)

# Create the visualization
pyLDAvis.display(newsgroup_data)

# Save the visualization as a html file 
pyLDAvis.save_html(newsgroup_data, 'data/model/newsgroup_ldavis.html')