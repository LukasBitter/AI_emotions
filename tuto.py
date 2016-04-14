import sklearn
import sklearn.datasets

# tutorial: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

'''chargement de données d'entrainement'''
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
#twenty_train=sklearn.datasets.load_files("C:\\enseignement\\tutorial\\text_analytics\\data\\twenty_newsgroups\\20news-bydate-train", description=None, categories=categories,load_content=True,shuffle=True, encoding='latin-1', decode_error='strict', random_state=42)
twenty_train=sklearn.datasets.load_files("C:\\Users\\Lukas\\Google Drive\\He-Arc\\III\\intelligenceArtificielle\\TP\\classification\\20news-bydate.tar\\20news-bydate\\20news-bydate-train", description=None, categories=categories,load_content=True,shuffle=True, encoding='latin-1', decode_error='strict', random_state=42)

print('twenty_train.target_names: ', twenty_train.target_names)

print('len(twenty_train.data: )', len(twenty_train.data))
print('len(twenty_train.filenames: )', len(twenty_train.filenames))

print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])
print(twenty_train.target[:10])