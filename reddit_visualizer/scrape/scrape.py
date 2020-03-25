import praw, sys
sys.setrecursionlimit(10000)


#taken from https://stackoverflow.com/a/36377995
def getSubComments(comment, allComments, verbose=True):
  allComments.append(comment)
  if not hasattr(comment, "replies"):
    replies = comment.comments()
    if verbose: print("fetching (" + str(len(allComments)) + " comments fetched total)")
  else:
    replies = comment.replies
  for child in replies:
    getSubComments(child, allComments, verbose=verbose)

def getAll(r, submissionId, verbose=True):
  submission = r.submission(submissionId)
  comments = submission.comments
  commentsList = []

  for comment in comments:
      #print(comment.body)
      getSubComments(comment, commentsList, verbose=verbose)
  return commentsList

def scrape_thread(reddit: praw.Reddit,  thread_id: str):
    """

    :param praw: A praw API instance. Set up an account here: https://praw.readthedocs.io/en/latest/getting_started/authentication.html
    :param thread_id:
    :return:
    """
    return getAll(reddit, thread_id)