from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer
from praw.models import Comment
import pickle, spacy, re, os, csv
from sklearn.cluster import KMeans
from reddit_visualizer.scrape import scrape_thread
"""
Gathers top levels comments from reddit. Threads with large amounts of comments take a large amount of time.
A default cached thread is included for instant visualization. You will need to make a reddit api account to
cache your own. See https://github.com/reddit-archive/reddit/wiki/oauth2.

1) Processes comments
2) Generates tfidf representations and assigns them to clusters for coloring
3) Trains a t-SNE network to embed the tfidf representations and generates scatterplot.

All intermediary files are saved in /cache.
"""
thread_url = "https://www.reddit.com/r/politics/comments/fod02m/discussion_thread_white_house_coronavirus_task/"
num_threads = 3
random_state=0
thread_id = thread_url.split('/')[6]

n_clusters=10
tsne_svd = 30 #change between 30-100 until clusters look right.
tsne_iterations = 2000

print(f"Generating TSNE plot for {thread_id}.")

if os.path.exists(f'cache/{thread_id}.pk'):
    comments = pickle.load(open(f"cache/{thread_id}.pk", 'rb'))
else:
    import praw
    #make a reddit api app account here: tutorial https://github.com/reddit-archive/reddit/wiki/oauth2
    username = 'naturalanguage'
    userAgent = "naturalanguage/0.1 by " + username
    clientId = ''
    clientSecret = ""
    reddit = praw.Reddit(user_agent=userAgent, client_id=clientId, client_secret=clientSecret)
    comments = scrape_thread(reddit, thread_id)
    pickle.dump(comments, open(f"cache/{thread_id}.pk", "wb"))

if not os.path.exists(f'cache/{thread_id}'):
    os.mkdir(f'cache/{thread_id}')

def clean_comment(comment):
    comment = re.sub('\((.+)\)\[.+\]', '\1', comment) #remove markdown urls but keep description text
    comment = re.sub(r'https?:\/\/.*[\r\n]*', ' ', comment, flags=re.MULTILINE) #remove all urls
    return comment


documents = []
authors = []
#Get top level comments
for comment in comments:
    if isinstance(comment, Comment):
        documents.append(clean_comment(comment.body))
        authors.append(comment.author)

print(f"Gathered {len(documents)} comments.")

lemmatized_documents = []
if os.path.exists(f"cache/{thread_id}/{thread_id}_lemmatized_comments.pk"):
    print(f"Loading pre-trained lemmatized comments.")
    lemmatized_documents = pickle.load(open(f"cache/{thread_id}/{thread_id}_lemmatized_comments.pk", 'rb'))
else:
    language = spacy.load("en_core_web_sm")
    idx=0
    for doc in language.pipe(documents, batch_size=100, n_process=num_threads):
        lemmatized_documents.append(" ".join([token.text for token in doc if not token.is_stop]).replace("PRON", " "))
    pickle.dump(lemmatized_documents, open(f"cache/{thread_id}/{thread_id}_lemmatized_comments.pk", 'wb'))


tfidf = TfidfVectorizer()

X = tfidf.fit_transform(lemmatized_documents)
print(f"Generated tfidf vectors for lemmatized documents")

if os.path.exists(f"cache/{thread_id}/{thread_id}_fitted_kmeans.pk"):
    print(f"Loading pre-trained k-means.")
    kmeans = pickle.load(open(f"cache/{thread_id}/{thread_id}_fitted_kmeans.pk", 'rb'))
else:
    print(f"Fitting k-means")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, verbose=1, n_jobs=num_threads)
    kmeans = kmeans.fit(X)
    pickle.dump(kmeans, open(f"cache/{thread_id}/{thread_id}_fitted_kmeans.pk", 'wb'))

y = kmeans.labels_



if os.path.exists(f"cache/{thread_id}/{thread_id}_tsne_fitted.pk"):
    print(f"Loading pre-trained tsne comments.")
    tsne = pickle.load(open(f"cache/{thread_id}/{thread_id}_tsne_fitted.pk", 'rb'))
else:
    tsne = TSNEVisualizer(decompose_by=tsne_svd, n_iter = tsne_iterations, verbose = 2)
    tsne = tsne.fit(X, y)
    pickle.dump(tsne, open(f"cache/{thread_id}/{thread_id}_tsne_fitted.pk", 'wb'))


tsne.colors = """#33000e, #660029, #bf0080, #660080, #0088ff, #00708c, #008066, #4cbf00, #735c00, #ff8800, #995200, #402200, #ff4400, #590000, #ff4073, #ff40f2, #7736d9, #101040, #233f8c, #36ced9, #36d98d, #538020, #b6bf30, #b22d2d, #733960, #8959b3, #1a2033, #46628c, #73bfe6, #1a2e33, #204020, #ffd580, #f29979, #8c5946, #cc99c2, #bfbfff, #698c8a, #eaffbf, #8c8569, #4d4439, #bfa38f, #e6acac""".split(', ')
tsne.draw(tsne.vecs, y, point_annotations=[f"{document}\n{author}" for document, author in zip(documents, authors)])


with open(f'cache/{thread_id}/{thread_id}_clustered.tsv', 'w') as file:
    csv_writer = csv.writer(file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow("x	y	topic	text	author".split('\t'))
    for vec, label,text,author in zip(tsne.vecs,y,documents,authors):
        csv_writer.writerow(list(vec)+[tsne.colors[label],text,author])

tsne.show()