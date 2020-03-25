from sklearn.feature_extraction.text import CountVectorizer
from praw.models import Comment
import pickle, spacy, re, os, pandas
from yellowbrick.text import FreqDistVisualizer


num_threads = 3
random_state=0
thread_id = 'fo7s7f'


print(f"Generating Frequency Counts for {thread_id}.")


def clean_comment(comment):
    comment = re.sub('\((.+)\)\[.+\]', '\1', comment) #remove markdown urls but keep description text
    comment = re.sub(r'https?:\/\/.*[\r\n]*', ' ', comment, flags=re.MULTILINE) #remove all urls
    return comment

comments = pickle.load(open(f"{thread_id}.pk", 'rb'))
documents = []
authors = []
#Get top level comments
for comment in comments:
    if isinstance(comment, Comment):
        documents.append(clean_comment(comment.body))
        authors.append(comment.author)

print(f"Gathered {len(documents)} comments.")
lemmatized_documents = []
if os.path.exists(f"cache/{thread_id}_lemmatized_comments.pk"):
    lemmatized_documents = pickle.load(open(f"cache/{thread_id}_lemmatized_comments.pk", 'rb'))
else:
    language = spacy.load("en_core_web_sm")
    idx=0
    for doc in language.pipe(documents, batch_size=100, n_process=num_threads):
        lemmatized_documents.append(" ".join([token.lemma_ for token in doc if not token.is_stop]).replace("PRON", " "))
    pickle.dump(lemmatized_documents, open(f"cache/{thread_id}_lemmatized_comments.pk", 'wb'))

vectorizer = CountVectorizer()
docs = vectorizer.fit_transform(lemmatized_documents)
features = vectorizer.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(docs)
visualizer.show()

user_to_comments = {}
for user, comment in zip(authors, documents):
    user = str(user)
    if user not in user_to_comments:
        user_to_comments[user] = []
    user_to_comments[user].append(comment)

user_to_comments = {k: v for k, v in sorted(user_to_comments.items(), key=lambda item: -len(item[1]))}
df = pandas.DataFrame(data = {'username':[k for k in user_to_comments for doc in user_to_comments[k]]})

df['username'].value_counts().sort_values()[-25:].plot(kind = 'barh', title='User counts')
import matplotlib.pyplot as plt
plt.show()




