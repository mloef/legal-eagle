import re
from sentence_transformers import SentenceTransformer, util

MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def index_flow(text):
    chunks = split(text)
    embeddings = embed(chunks)
    return embeddings


def preprocess(text):
    return text.replace("\\n", " ").replace("\\'", "'").replace('\\"', '"') #TODO: ugh


def split(text): #TODO: implement a sliding window split that respects sentence bounds, then dedupe overlapping windows at topk time
    chunks = []
    start = 0
    count = 0
    for match in re.finditer(" ",  text):
        count += 1
        if count == 100:
            end = match.end()
            chunks.append((text[start:end], (start, end)))
            start = end
            count = 0

    chunks.append((text[start:len(text)-1], (start, len(text)-1)))
    #TODO: further splitting, cleaning
    return chunks


def split_to_sentences(text):
    sentences = []
    start = 0
    for match in re.finditer("\. ",  text):
        sentences.append((text[start:match.end()], (start, match.end())))
        start = match.end()
    sentences.append((text[start:len(text)-1], (start, len(text)-1)))
    #TODO: further splitting, cleaning
    return sentences


def embed(chunks):
    embeddings = [(MODEL.encode(chunk[0], convert_to_tensor=True), chunk[1]) for chunk in chunks]
    return embeddings


def run_query(query, index, k=10): #TODO: use more builtins by modifying index to avoid for loops and not sort list
    query_embedding = MODEL.encode(query, convert_to_tensor=True)

    cos_scores = []
    for filename, embeddings in index.items():
        for embedding, location in embeddings:
            cos_score = util.cos_sim(query_embedding, embedding)[0]
            cos_scores.append((cos_score, location, filename))

    cos_scores.sort(key=lambda x: x[0], reverse=True)

    if k > len(cos_scores) or k == -1:
        return cos_scores

    return cos_scores[0:k]


def highlight(query, passage):
    pieces = split_to_sentences(passage)
    subindex = {"dummy": embed(pieces)}
    return run_query(query, subindex, k=1)
