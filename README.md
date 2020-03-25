### Visual Reddit Threads with some NLP

Converts a reddit thread ID to:
- interactive t-SNE plots of comments.
- Plots of various comment statistics (user counts, token statistics, etc).

See [examples](/examples).

### Installation
Install the package with:
```python
pip install git+https://github.com/AndriyMulyar/reddit_visualizer
```
and install a custom version of yellowbrick with:
```python
pip install git+https://github.com/AndriyMulyar/yellowbrick@develop
```
![TSNE Example](docs/tsne_example.png)

### Use
Clone the repository, pip install it, then directly run any script in the examples folder.

### Implementation Details
Leverages praw for reddit api, yellowbrick as a wrapper for sklearn t-SNE, spacy for preprocessing.