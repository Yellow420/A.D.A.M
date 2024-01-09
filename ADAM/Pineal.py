#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import platform
import psutil
import GPUtil
import subprocess
from typing import Dict, Any
import fasttext
import requests
import re
from bs4 import BeautifulSoup
import wikipedia
import praw
import stackexchange
import urllib.parse
import geocoder
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
nlp = spacy.load("en_core_web_sm")
reddit = praw.Reddit(
    client_id='p-CkBa5fBhXpV5xnA-lihw',
    client_secret='dqBTsuZ0tKOGShfEagfNzhKqLZkGWg',
    user_agent='windows:my_reddit_app:v1.0 (by /u/Good-Mention-5859)'
)
stack_api_key = '98vNjbp7kXs*oezAlBTfvA(('



def formulate_question(entities, doc):
    """
    Formulate a question based on entities and dynamically determine the question type.

    Parameters:
    - entities (list): List of entities.
    - doc (spaCy Doc): Processed spaCy Doc object.

    Returns:
    - str: Formulated question.
    """
    if not entities:
        return ""

    # Analyze relationships between entities
    entity_relationships = analyze_entity_relationships(entities, doc)

    # Determine the most suitable question type based on relationships
    question_type = analyze_entity_relationships_to_question_type(entity_relationships)

    # Formulate a question based on the determined question type and entities
    entity_question = f"{question_type.capitalize()} {' and '.join(entities)}?"
    return entity_question

def analyze_entity_features(entity_features):
    """
    Analyze linguistic features of entities to determine the most suitable question type.

    Parameters:
    - entity_features (list): List of spaCy Doc objects representing entity roots.

    Returns:
    - str: Most suitable question type.
    """
    if not entity_features:
        return ""

    # Extract POS (Part of Speech) tags and count occurrences
    pos_tags = [token.pos_ for entity in entity_features for token in entity]
    pos_counts = Counter(pos_tags)

    # Map POS tags to question types based on dynamic counts
    pos_to_question_type = {
        "PERSON": "who",
        "NORP": "what",
        "FAC": "where",
        "LOC": "where",
        "PRODUCT": "what",
        "EVENT": "what",
        "WORK_OF_ART": "what",
        "LANGUAGE": "what",
        "DATE": "when",
        "TIME": "when",
        "PERCENT": "what",
        "MONEY": "what",
        "QUANTITY": "what",
        "ORDINAL": "what",
        "CARDINAL": "what",
    }

    # Dynamically update the mapping based on entity type occurrences
    for entity_type, count in pos_counts.items():
        if entity_type not in pos_to_question_type or count > pos_counts[pos_to_question_type[entity_type]]:
            pos_to_question_type[entity_type] = "what"

    # Determine the most common POS tag
    common_pos_tag = max(set(pos_tags), key=pos_tags.count)

    # Use the mapped question type or default to "what"
    question_type = pos_to_question_type.get(common_pos_tag, "what")

    return question_type

def analyze_entity_relationships(entities, doc):
    """
    Analyze relationships between entities using spaCy's dependency parsing.

    Parameters:
    - entities (list): List of entities.
    - doc (spaCy Doc): Processed spaCy Doc object.

    Returns:
    - Counter: Count of relationships between entities.
    """
    entity_relationships = Counter()

    for entity in entities:
        for token in entity:
            # Analyze the head of each token in the entity
            head = token.head
            # Consider only heads that are not punctuation and are part of other entities
            if head.dep_ != "punct" and head.ent_type_ != "":
                entity_relationships[(head.ent_type_, entity.root.ent_type_)] += 1

    return entity_relationships

def analyze_entity_relationships_to_question_type(entity_relationships):
    """
    Determine the most suitable question type based on relationships between entities.

    Parameters:
    - entity_relationships (Counter): Count of relationships between entities.

    Returns:
    - str: Most suitable question type.
    """
    # Map entity relationships to question types
    relationship_to_question_type = {
        ("PERSON", "PERSON"): "who",
        ("ORG", "PERSON"): "who",
        ("GPE", "PERSON"): "where",
        ("PERSON", "GPE"): "where",
        ("DATE", "EVENT"): "when",
        ("TIME", "EVENT"): "when",
        ("LOC", "FAC"): "where",
        ("FAC", "LOC"): "where",
    }

    # Determine the most common relationship
    common_relationship = max(entity_relationships, key=entity_relationships.get)

    # Use the mapped question type or default to "what"
    question_type = relationship_to_question_type.get(common_relationship, "what")

    return question_type

def extract_entities_and_question(text):
    # Load spaCy NER model
    nlp = spacy.load("en_core_web_sm")

    # Process the input text with spaCy
    doc = nlp(text)

    # Extract entities using spaCy NER
    entities = [ent for ent in doc.ents if ent.root.dep_ != "punct"]

    # Extract a potential question from the text
    question_words = ["who", "what", "where", "when", "why", "how"]

    # Split into sentences based on "?"
    sentences = [sentence.strip() for sentence in text.split("?")]

    # Use the last non-empty sentence as a potential question
    question = next((s for s in reversed(sentences) if s), "").strip()

    # If no explicit question found, construct a question based on entities
    if not any(word in question.lower() for word in question_words) and entities:
        # Formulate a question based on the most suitable question type determined by entity features
        entity_question = formulate_question(entities, doc)
        question = entity_question

    return question

def rank_topics(label_probabilities, potential_topics):
    """
    Rank potential topics based on their relevance to the predicted label probabilities.

    Parameters:
    - label_probabilities (list): List of predicted label probabilities from TinyBERT.
    - potential_topics (list): List of potential topics extracted by spaCy's NER.

    Returns:
    - list: Ranked list of potential topics.
    """
    # Combine label probabilities with potential topics
    combined_data = list(zip(label_probabilities, potential_topics))

    # Rank potential topics based on the predicted label probabilities
    ranked_topics = sorted(combined_data, key=lambda x: x[0], reverse=True)

    # Extract only the topic names from the ranked list
    ranked_topic_names = [topic[1] for topic in ranked_topics]

    return ranked_topic_names

def extract_text(text):
    # Load pre-trained TinyBERT model and tokenizer
    model_name = "prajjwal1/bert-tiny"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the input text
    tokens = tokenizer(text, return_tensors="pt")

    # Make predictions using the model
    with torch.no_grad():
        outputs = model(**tokens)

    # Get the predicted label probabilities
    label_probabilities = torch.softmax(outputs.logits, dim=1).tolist()[0]

    # Extract the topic based on the predicted label probabilities
    # Replace with your specific spaCy model if needed
    nlp = spacy.load("en_core_web_sm")

    # Process the text with spaCy NER
    doc = nlp(text)

    # Extract potential topics (entities) from the text
    potential_topics = [ent.text for ent in doc.ents]

    # If no entities are found, use a default topic
    if not potential_topics:
        potential_topics = ["GeneralTopic"]

    # Rank potential topics based on their relevance to the predicted label probabilities
    ranked_topics = rank_topics(label_probabilities, potential_topics)

    # Select the most relevant topic
    topic = ranked_topics[0]

    # Extract a potential question from the text
    question = extract_entities_and_question(text)

    # Combine topic and question to form the final text
    text = f"{topic} {question}?"
    return text


def remove_duplicates(text):
    sentences = nltk.sent_tokenize(text)
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    unique_sentences = []
    for idx, sentence in enumerate(sentences):
        if all(cosine_matrix[idx, :idx] < 0.8):
            unique_sentences.append(sentence)
    return " ".join(unique_sentences)


def nltk_summarize(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    non_stop_words = [word for word in words if word not in stop_words]
    freq_dist = FreqDist(non_stop_words)
    sorted_freq_dist = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)
    most_common_words = [item[0] for item in sorted_freq_dist[:10]]  # Adjust as needed
    sentences = sent_tokenize(text)
    summary_sentences = [sentence for sentence in sentences if any(word in sentence for word in most_common_words)]
    summary = " ".join(summary_sentences)
    return summary


def spacy_summarize(text, nlp):
    doc = nlp(text)
    sentences = [sent for sent in doc.sents if len(sent.ents) + len(sent.noun_chunks) > 2]  # Adjust threshold as needed
    summary = " ".join(str(sent) for sent in sentences)
    return summary


def sumy_summarize(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summarized_info = " ".join(
        str(sentence) for sentence in summarizer(parser.document, sentences_count=10))  # Adjust sentence_count as needed
    return summarized_info


def get_key_sentences(summarized_text, original_text, num_topics=3):
    # Tokenize the original text to identify the most prevalent topic
    tokenized_original_text = nltk.word_tokenize(original_text)
    dictionary = corpora.Dictionary([tokenized_original_text])
    bow = dictionary.doc2bow(tokenized_original_text)
    lda_model = LdaModel([bow], num_topics=num_topics, id2word=dictionary, passes=15)
    # Get the topic distribution for the single document
    doc_topics = lda_model[bow]
    # Sort the topics by proportion
    sorted_topics = sorted(doc_topics, key=lambda x: -x[1])
    # Get the most prevalent topic
    most_prevalent_topic = sorted_topics[0]
    # Get the keywords for the most prevalent topic
    topic_keywords = [dictionary[id] for id, freq in lda_model.get_topic_terms(most_prevalent_topic[0])]
    # Extract sentences from the summarized text that relate to the identified topic
    key_sentences = [sentence for sentence in nltk.sent_tokenize(summarized_text) if
                     any(keyword in sentence for keyword in topic_keywords)]
    return ' '.join(key_sentences)

def reduce_text(gathered_info, original_text):
    gathered_info = re.sub(r'https?:\/\/[^\s]+', '', gathered_info)
    non_informative_phrases = ["click here", "read more", "advertisement"]
    for phrase in non_informative_phrases:
        gathered_info = gathered_info.replace(phrase, "")
    doc = nlp(gathered_info)
    sentences = [sent.text for sent in doc.sents]
    gathered_info = ' '.join(sorted(set(sentences), key=sentences.index))
    print(f"{gathered_info}")
    informative_text_parts = [token.text for token in doc if
                              token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'] or token.ent_type_]
    gathered_info = ' '.join(informative_text_parts)
    print(f"{gathered_info}")
    gathered_info = remove_duplicates(gathered_info)
    print(f"{gathered_info}")
    gathered_info = get_key_sentences(gathered_info, original_text)
    print(f"{gathered_info}")
    prev_info = ""  # Placeholder for the previous state of gathered_info to check if summarization is still reducing the text
    while len(gathered_info.split()) > 50 and gathered_info != prev_info:  # Assuming 50 words as the threshold
        prev_info = gathered_info  # Update prev_info before summarization steps

        # Summarization steps
        summarized_info = nltk_summarize(gathered_info)
        if not summarized_info:
            summarized_info = gathered_info  # Pass the original text if summarization fails
        print(f"{summarized_info}")
        summarized_info = spacy_summarize(summarized_info, nlp)
        if not summarized_info:
            summarized_info = gathered_info  # Pass the original text if summarization fails
        print(f"{summarized_info}")
        summarized_info = sumy_summarize(summarized_info)
        if not summarized_info:
            summarized_info = gathered_info  # Pass the original text if summarization fails
        print(f"{summarized_info}")
        # Organization step using Gensim's LDA
        organized_info = get_key_sentences(summarized_info, original_text)
        if organized_info:
            summarized_info = organized_info  # Update summarized_info if organization step is successful

        gathered_info = summarized_info  # Update gathered_info with the latest summarization result
    print(f"{gathered_info}")
    return gathered_info

def check_html(html):
    # Load the trained model
    model = fasttext.load_model('models/html_model.bin')

    # Example HTML snippet for prediction
    html = "<div class='new-class'>New Content</div>"

    # Predict HTML structure
    htmlstructure = model.predict(html)
    return htmlstructure

def inform(text):
    gathered_info = ''
    sources = ['google', 'wikipedia', 'bing', 'yahoo', 'reddit', 'stack_exchange', 'amazon', 'walmart', 'weather',
               'location', 'records']
    for source in sources:
        print(f'Starting with {source} source...')
        source_info = ''  # Initialize source_info for each source
        if source == 'google':
            try:
                url = 'https://www.google.com/search?q=' + text
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.find_all('div', {'class': 'rc'}, limit=5)  # Adjusted to get the first five results

                for result in results:
                    html_structure = result.prettify()
                    selectors = check_html(html_structure)
                    # Use the selectors to extract more data...
                    for selector in selectors:
                        tag, attr = selector.split(',')
                        attr_name, attr_value = attr.split('=')
                        extracted_data = result.find(tag, {attr_name: attr_value})
                        if extracted_data:
                            print(extracted_data.text)

            except:
                pass  # Handle exception

        elif source == 'wikipedia':
            try:
                wiki_summary = wikipedia.summary(text, sentences=20)  # Get a summary from Wikipedia
                source_info += wiki_summary + '\n'
            except:
                pass  # Handle exception


        elif source == 'bing':
            try:
                url = 'https://www.bing.com/search?q=' + urllib.parse.quote(text)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.find_all('li', {'class': 'b_algo'}, limit=5)  # Adjusted to get the first five results
                for result in results:
                    html_structure = result.prettify()
                    selectors = check_html(html_structure)
                    # Use the selectors to extract more data...
                    for selector in selectors:
                        tag, attr = selector.split(',')
                        attr_name, attr_value = attr.split('=')
                        extracted_data = result.find(tag, {attr_name: attr_value})
                        if extracted_data:
                            print(extracted_data.text)
            except:
                pass  # Handle exception


        elif source == 'yahoo':
            try:
                url = 'https://search.yahoo.com/search?p=' + urllib.parse.quote(text)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.find_all('div', {'class': 'dd algo algo-sr fst Sr'},
                                        limit=5)  # Adjusted to get the first five results
                for result in results:
                    html_structure = result.prettify()
                    selectors = check_html(html_structure)
                    # Use the selectors to extract more data...
                    for selector in selectors:
                        tag, attr = selector.split(',')
                        attr_name, attr_value = attr.split('=')
                        extracted_data = result.find(tag, {attr_name: attr_value})
                        if extracted_data:
                            print(extracted_data.text)
            except:
                pass  # Handle exception

        elif source == 'reddit':
            try:
                reddit = praw.Reddit(client_id='your_client_id',
                                     client_secret='your_client_secret',
                                     user_agent='your_user_agent')
                for submission in reddit.subreddit('all').search(text, limit=3):
                    source_info += submission.title + ' ' + submission.selftext + '\n'
            except:
                pass  # Handle exception

        elif source == 'stack_exchange':
            try:
                so = stackexchange.Site(stackexchange.StackOverflow, app_key='stack_api_key')
                questions = so.search(intitle=text, sort='relevance', limit=3)
                for question in questions:
                    source_info += question.title + '\n' + question.body + '\n'
            except:
                pass  # Handle exception


        elif source == 'amazon':
            try:
                search_query = urllib.parse.quote(text)
                url = f'https://www.amazon.com/s?k={search_query}'
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                html_structure = soup.prettify()
                selectors = check_html(html_structure)
                # Use the selectors to extract more data...
                for selector in selectors:
                    tag, attr = selector.split(',')
                    attr_name, attr_value = attr.split('=')
                    extracted_data = soup.find(tag, {attr_name: attr_value})
                    if extracted_data:
                        print(extracted_data.text)
            except:
                pass  # Handle exception

        elif source == 'walmart':
            try:
                search_query = urllib.parse.quote(text)
                url = f'https://www.walmart.com/search/?query={search_query}'
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                html_structure = soup.prettify()
                selectors = check_html(html_structure)
                # Use the selectors to extract more data...
                for selector in selectors:
                    tag, attr = selector.split(',')
                    attr_name, attr_value = attr.split('=')
                    extracted_data = soup.find(tag, {attr_name: attr_value})
                    if extracted_data:
                        print(extracted_data.text)
            except:
                pass  # Handle exception

        elif source == 'weather':
            try:
                g = geocoder.ip('me')  # Get current location based on IP
                location = f"{g.city}, {g.state}, {g.country}"
                url = f'https://wttr.in/{location.replace(" ", "+")}?format=%C+%t'
                response = requests.get(url)
                weather_info = response.text.strip()
                source_info += weather_info + '\n'
            except:
                pass  # Handle exception

        elif source == 'location':
            try:
                g = geocoder.ip('me')  # Get current location based on IP
                location_info = f"Current Location: {g.city}, {g.state}, {g.country}"
                source_info += location_info + '\n'
            except:
                pass  # Handle exception


        elif source == 'records':
            try:
                urls = [
                    'https://www.publicrecordsnow.com/search/' + urllib.parse.quote(text),
                    'https://www.crimcheck.net/person-search/?fname=&lname=' + urllib.parse.quote(text),
                    'https://www.medicalrecords.com/search?q=' + urllib.parse.quote(text)
                ]
                for url in urls:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                    }
                    response = requests.get(url, headers=headers)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    html_structure = soup.prettify()
                    selectors = check_html(html_structure)
                    for selector in selectors:
                        tag, attr = selector.split(',')
                        attr_name, attr_value = attr.split('=')
                        extracted_data = soup.find(tag, {attr_name: attr_value})
                        if extracted_data:
                            print(extracted_data.text)

            except Exception as e:
                print(f"An error occurred: {e}")

    print(
        f'Information gathered from {source} source:\n{source_info}')  # Print statement showing gathered info for each source
    gathered_info += source_info  # Add the source_info to gathered_info
    gathered_info = reduce_text(gathered_info, text)

    return gathered_info

def gather_web_info(text):
    text = extract_text(text)
    info = inform(text)
    return info

def extract_data_from_html(html_content, selectors):
    data = []
    soup = BeautifulSoup(html_content, 'html.parser')

    for selector in selectors:
        tag, attr = selector.split(',')
        attr_name, attr_value = attr.split('=')
        extracted_data = soup.find_all(tag, {attr_name: attr_value})
        data.extend([data.get_text() for data in extracted_data])

    return data

def gather_system_info() -> Dict[str, Any]:
    try:
        # Get basic system information
        system_info: Dict[str, Any] = {
            "os": platform.system() + " " + platform.release(),
            "cpu": platform.processor(),
            "ram": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
            "devices": ["microphone", "speaker"],  # Add more devices as needed
        }
        # Get battery information
        try:
            battery = psutil.sensors_battery()
            system_info["battery"] = {
                "percent": f"{battery.percent}%",
                "power_plugged": battery.power_plugged
            }
        except AttributeError:
            system_info["battery"] = {
                "percent": "Not applicable",
                "power_plugged": "Not applicable"
            }
        # Get information about running processes
        try:
            running_processes = [{"pid": process.pid, "name": process.name(), "username": process.username()} for
                                 process in psutil.process_iter(['pid', 'name', 'username'])]
            system_info["running_processes"] = running_processes
        except Exception as e:
            system_info["running_processes"] = [{"error": f"Error: {str(e)}"}]
            # Get information about opened webpages
            try:
                # This is just a placeholder, you might need to adjust this based on your requirements
                opened_webpages = subprocess.check_output(["wmic", "process", "list", "brief"]).decode()
                # Assume check_html_structure returns a list of selectors
                selectors = check_html(opened_webpages)
                # Extract data from HTML content using selectors
                extracted_data = extract_data_from_html(opened_webpages, selectors)
                system_info["opened_webpages"] = extracted_data
            except Exception as e:
                system_info["opened_webpages"] = f"Error: {str(e)}"
        # Disk Usage
        disk_info = psutil.disk_usage('/')
        system_info["disk_usage"] = {
            "total": disk_info.total,
            "used": disk_info.used,
            "free": disk_info.free
        }
        # Network Information
        network_info = psutil.net_if_addrs()
        system_info["network_info"] = network_info
        # GPU Information
        try:
            gpu_info = []
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    "name": str(gpu.name),
                    "memoryTotal": str(gpu.memoryTotal),
                    "memoryUsed": str(gpu.memoryUsed),
                    "memoryFree": str(gpu.memoryFree),
                    "load": str(gpu.load)
                })
            system_info["gpu_info"] = gpu_info
        except Exception as e:
            system_info["gpu_info"] = [{"error": f"Error: {str(e)}"}]
        # Network Connections
        connections = psutil.net_connections()
        system_info["network_connections"] = [{"fd": conn.fd, "family": conn.family, "type": conn.type, "laddr": conn.laddr, "raddr": conn.raddr, "status": conn.status} for conn in connections]
        # CPU Frequency
        cpu_frequency = psutil.cpu_freq()
        system_info["cpu_frequency"] = {"current": cpu_frequency.current, "min": cpu_frequency.min, "max": cpu_frequency.max}
        # Sensors Information (Temperature, Fan Speeds, etc.)
        try:
            sensors_info = psutil.sensors_temperatures()
            system_info["sensors_info"] = sensors_info
        except Exception as e:
            system_info["sensors_info"] = f"Error: {str(e)}"
        return system_info
    except Exception as e:
        return {"error": f"Error gathering system information: {str(e)}"}