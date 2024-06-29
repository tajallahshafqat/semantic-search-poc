import sys
import os
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import psycopg2
from pgvector.psycopg2 import register_vector
import json
import string
from dotenv import load_dotenv

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

embedder = SentenceTransformer("all-MiniLM-L6-v2")

stopwords_nltk = stopwords.words('english')

def preprocess_job_description(job):
    text = job['job_position_description'].lower()
    # Tokenize keeping special terms intact
    words = word_tokenize(text)
    # Remove punctuation
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
    words = [word for word in words if word and word not in stopwords_nltk]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    preprocessed_text = ' '.join(words)
    job['job_position_description'] = preprocessed_text
    return job

def create_embeddings(job):
    job = preprocess_job_description(job)
    corpus = job['job_position_title'] + ': ' + job['job_position_description']
    embeddings = embedder.encode(corpus, convert_to_tensor=True).tolist()
    return embeddings

def store_embeddings(job):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    embeddings = create_embeddings(job)

    insert_query = """
    INSERT INTO jobs_embeddings (embeddings, job_id)
    VALUES (%s, %s)
    ON CONFLICT (job_id) DO NOTHING
    """
    cursor.execute(insert_query, (embeddings, job["id"]))

    conn.commit()
    cursor.close()
    conn.close()

def find_job_by_id(jobs, job_id):
    for job in jobs:
        if job['id'] == job_id:
            return job['job_position_title']
    return None

def query_embeddings(job):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')

    register_vector(conn)

    embeddings = create_embeddings(job)

    query = """
    SELECT id, embeddings, job_id, embeddings <=> %s::vector AS distance
    FROM jobs_embeddings
    ORDER BY embeddings <=> %s::vector
    LIMIT 10;
    """
    cursor.execute(query, (embeddings, embeddings))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    for result in results:
        job_title = find_job_by_id(jobs, result[2])
        print(f"ID: {result[0]}, Job title: {job_title}, Distance: {result[3]}")

if __name__ == '__main__':
    f = open("data/jobs.json")
    jobs_json = json.load(f)

    jobs = jobs_json["jobs"]
    
    if len(sys.argv) > 1 and sys.argv[1] == 'store':
        subset = jobs[:50000]

        for job in subset:
            store_embeddings(job)

    elif len(sys.argv) > 1 and sys.argv[1] == 'query':
        f = open("data/query.json")
        query_json = json.load(f)

        query = query_json["job"]
        query_embeddings(query)

    else:
        print("Invalid argument. Please provide 'store' or 'query'.")
