from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(stop_words=None)
content = open('./a9a/a9a').read()
vectorizer.fit_transform(content)

import pprint
pprint.pprint(content)
print(vectorizer.get_feature_names())