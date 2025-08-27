from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import re
from collections import Counter
from werkzeug.utils import secure_filename
import spacy
from spacy import displacy
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import syllables
import numpy as np
import networkx as nx
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('vader_lexicon')
nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt'}
app.secret_key = 'your-secret-key-here'  # Добавьте секретный ключ для flash сообщений

nlp = spacy.load("en_core_web_sm")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_plot(data, title, plot_type='bar', color='#4361ee'):
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == 'bar':
        bars = ax.bar(data.keys(), data.values(), color=color, alpha=0.7)
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

    elif plot_type == 'pie':
        wedges, texts, autotexts = ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%',
                                          colors=['#4361ee', '#4895ef', '#3a0ca3', '#f72585', '#7209b7'])
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45 if plot_type == 'bar' else 0)
    plt.tight_layout()

    if plot_type == 'bar':
        # Добавляем сетку для лучшей читаемости
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return plot_data

def create_empty_image(message="WordCloud not available"):
    """Создает пустое изображение-заглушку"""
    plt.figure(figsize=(8, 4), facecolor='white')
    plt.text(0.5, 0.5, message,
             ha='center', va='center', fontsize=14, color='gray',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#f8f9fa',
                       edgecolor='#dee2e6', alpha=0.8))
    plt.axis('off')

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100,
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def generate_wordcloud(text):
    try:
        if not text.strip():
            return create_empty_image("No text available")

        # Предварительная обработка текста
        words = [token.text.lower() for token in nlp(text)
                 if token.is_alpha and not token.is_stop and len(token.text) > 2]
        processed_text = ' '.join(words)

        if not processed_text.strip():
            return create_empty_image("Not enough meaningful words")

        # Создаем wordcloud с улучшенными настройками
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis',
            collocations=True,
            prefer_horizontal=0.7,
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=120,
            random_state=42
        ).generate(processed_text)

        # Сохраняем в buffer
        buffer = BytesIO()
        wordcloud.to_image().save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Ошибка генерации wordcloud: {e}")
        return create_empty_image("WordCloud generation error")

def extended_analysis(text):
    """Расширенный лингвистический анализ текста"""

    # NLP-инструменты
    doc = nlp(text)
    blob = TextBlob(text)
    sid = SentimentIntensityAnalyzer()
    sentences = list(doc.sents)

    # 0. Статистика длины предложений
    sentence_lengths = [len([token for token in sent if not token.is_punct and token.is_alpha]) for sent in sentences]

    # Распределение длины предложений по категориям
    sentence_length_distribution = Counter()
    for length in sentence_lengths:
        if length < 5:
            sentence_length_distribution['<5'] += 1
        elif length < 10:
            sentence_length_distribution['5-9'] += 1
        elif length < 15:
            sentence_length_distribution['10-14'] += 1
        elif length < 20:
            sentence_length_distribution['15-19'] += 1
        elif length < 25:
            sentence_length_distribution['20-24'] += 1
        else:
            sentence_length_distribution['25+'] += 1

    # Создаем график распределения длины предложений
    sentence_length_plot = create_plot(dict(sentence_length_distribution),
                                       "Распределение длины предложений", 'bar')

    sentence_stats = {
        'avg_sentence_length': round(np.mean(sentence_lengths), 2) if sentence_lengths else 0,
        'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
        'min_sentence_length': min(sentence_lengths) if sentence_lengths else 0,
        'total_sentences': len(sentences),
        'complex_sentences': sum(1 for length in sentence_lengths if length > 20),
        'complex_sentences_ratio': round(
            (sum(1 for length in sentence_lengths if length > 20) / len(sentence_lengths) * 100),
            2) if sentence_lengths else 0,
        'sentence_length_distribution': dict(sentence_length_distribution),
        'sentence_length_std': round(np.std(sentence_lengths), 2) if sentence_lengths else 0,
        'sentence_length_median': round(np.median(sentence_lengths), 2) if sentence_lengths else 0
    }

    # 1. Анализ читаемости
    def flesch_reading_ease(text):
        try:
            sentences = sent_tokenize(text)
            words = [word for word in word_tokenize(text) if word.isalpha()]

            if len(sentences) == 0 or len(words) == 0:
                return 0

            syllable_count = sum(syllables.estimate(word) for word in words)
            words_per_sentence = len(words) / len(sentences)
            syllables_per_word = syllable_count / len(words)

            return 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        except:
            return 0

    def automated_readability_index(text):
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            characters = sum(len(word) for word in words if word.isalpha())

            if len(sentences) == 0 or len(words) == 0:
                return 0

            return 4.71 * (characters / len(words)) + 0.5 * (len(words) / len(sentences)) - 21.43
        except:
            return 0

    readability = {
        'flesch': round(flesch_reading_ease(text), 2),
        'smog': round(1.0430 * np.sqrt(len(sent_tokenize(text)) * 30) + 3.1291, 2) if text else 0,
        'words_per_sentence': round(len(text.split()) / len(sent_tokenize(text)), 2) if text and len(
            sent_tokenize(text)) > 0 else 0,
        'ari': round(automated_readability_index(text), 2),
        'reading_level': 'Very Easy' if flesch_reading_ease(text) > 90 else
        'Easy' if flesch_reading_ease(text) > 80 else
        'Fairly Easy' if flesch_reading_ease(text) > 70 else
        'Standard' if flesch_reading_ease(text) > 60 else
        'Fairly Difficult' if flesch_reading_ease(text) > 50 else
        'Difficult' if flesch_reading_ease(text) > 30 else 'Very Difficult'
    }

    # 2. Продвинутый анализ тональности
    sentiment = {
        'textblob': {
            'polarity': round(blob.sentiment.polarity, 3),
            'subjectivity': round(blob.sentiment.subjectivity, 3),
            'assessment': 'Positive' if blob.sentiment.polarity > 0.1 else
            'Negative' if blob.sentiment.polarity < -0.1 else 'Neutral'
        },
        'vader': sid.polarity_scores(text)
    }

    # Добавляем оценку для VADER
    vader_compound = sentiment['vader']['compound']
    sentiment['vader']['assessment'] = 'Positive' if vader_compound > 0.05 else \
        'Negative' if vader_compound < -0.05 else 'Neutral'

    # 3. Тематическое моделирование (LDA)
    def get_topics(text, num_topics=3):
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 3:  # Минимум 3 предложения для LDA
                return [(0, "0.000*\"not\" + 0.000*\"enough\" + 0.000*\"data\" + 0.000*\"for\" + 0.000*\"topics\"")]

            tokenized = [[word.lower() for word in word_tokenize(sent) if word.isalpha() and len(word) > 2] for
                         sent in sentences]
            tokenized = [tokens for tokens in tokenized if tokens]  # Убираем пустые списки

            if not tokenized or len(tokenized) < 2:
                return [(0, "0.000*\"not\" + 0.000*\"enough\" + 0.000*\"data\" + 0.000*\"for\" + 0.000*\"topics\"")]

            dictionary = corpora.Dictionary(tokenized)
            corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]

            # Автоматически определяем количество тем (но не больше, чем доступно документов)
            actual_num_topics = min(num_topics, len(tokenized) - 1) if len(tokenized) > 1 else 1

            lda = models.LdaModel(corpus, num_topics=actual_num_topics, id2word=dictionary, random_state=42, passes=10)
            return lda.print_topics(num_words=5)
        except Exception as e:
            print(f"Error in topic modeling: {e}")
            return [(0, "0.000*\"error\" + 0.000*\"in\" + 0.000*\"topic\" + 0.000*\"modeling\" + 0.000*\"analysis\"")]

    topics = get_topics(text)

    # 4. Анализ ключевых слов (TF-IDF)
    def get_keywords(text, n=15):
        try:
            if not text.strip():
                return []

            vectorizer = TfidfVectorizer(stop_words='english', max_features=n, ngram_range=(1, 2))
            X = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = np.array(X.sum(axis=0)).flatten()

            # Сортируем по убыванию важности
            keywords_with_scores = list(zip(feature_names, scores))
            keywords_with_scores.sort(key=lambda x: x[1], reverse=True)

            return keywords_with_scores
        except:
            return []

    keywords = get_keywords(text)

    # 5. Генерация облака слов
    wordcloud = generate_wordcloud(text)

    # 6. Анализ эмоций (расширенный лексикон)
    emotion_lexicon = {
        'joy': ['happy', 'joy', 'excited', 'delight', 'smile', 'laugh', 'fun', 'pleasure', 'enjoy', 'wonderful'],
        'anger': ['angry', 'mad', 'furious', 'rage', 'annoyed', 'hate', 'outrage', 'frustrated', 'irritated'],
        'sadness': ['sad', 'depressed', 'unhappy', 'grief', 'cry', 'tears', 'sorrow', 'melancholy', 'disappointed'],
        'fear': ['afraid', 'fear', 'scared', 'terrified', 'panic', 'anxiety', 'dread', 'nervous', 'worried'],
        'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'unexpected', 'wow', 'incredible'],
        'trust': ['trust', 'confidence', 'believe', 'reliable', 'faith', 'dependable', 'honest']
    }

    emotions = Counter()
    words = [token.text.lower() for token in doc if token.is_alpha]

    for word in words:
        for emotion, terms in emotion_lexicon.items():
            if word in terms:
                emotions[emotion] += 1

    # 7. Сетевой анализ (граф co-occurrence)
    def build_cooccurrence_network(text, window_size=3):
        try:
            words = [token.text.lower() for token in doc if
                     token.is_alpha and not token.is_stop and len(token.text) > 2]

            if len(words) < 5:  # Минимум 5 слов для построения сети
                return {
                    'nodes': [],
                    'edges': [],
                    'central_words': []
                }

            graph = nx.Graph()

            for i in range(len(words)):
                for j in range(i + 1, min(i + window_size, len(words))):
                    if words[i] != words[j]:
                        if graph.has_edge(words[i], words[j]):
                            graph[words[i]][words[j]]['weight'] += 1
                        else:
                            graph.add_edge(words[i], words[j], weight=1)

            # Добавляем размер узлов на основе степени центральности
            degrees = dict(graph.degree(weight='weight'))

            if degrees:
                max_degree = max(degrees.values())
                # Нормализуем веса для лучшего отображения
                for node in graph.nodes():
                    graph.nodes[node]['size'] = 10 + (degrees[node] / max_degree * 20)

                central_words = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:7]
            else:
                central_words = []

            return {
                'nodes': list(graph.nodes()),
                'edges': [(u, v, d['weight']) for u, v, d in graph.edges(data=True)],
                'central_words': central_words
            }
        except Exception as e:
            print(f"Error building co-occurrence network: {e}")
            return {
                'nodes': [],
                'edges': [],
                'central_words': []
            }

    cooccurrence = build_cooccurrence_network(text)

    # 8. Дополнительные метрики
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    word_freq = Counter(words)

    lexical_richness = {
        'type_token_ratio': round(len(set(words)) / len(words), 3) if words else 0,
        'hapax_legomena': len([word for word, count in word_freq.items() if count == 1]),
        'hapax_percentage': round(len([word for word, count in word_freq.items() if count == 1]) / len(words) * 100,
                                  2) if words else 0,
        'most_common_words': word_freq.most_common(10)
    }

    # 9. Анализ стиля письма
    def analyze_writing_style(doc):
        try:
            # Длина предложений
            sent_lengths = [len([token for token in sent if token.is_alpha]) for sent in doc.sents]

            # Использование различных частей речи
            pos_counts = Counter(token.pos_ for token in doc)
            total_tokens = len([token for token in doc if token.is_alpha])

            return {
                'avg_sentence_length': round(np.mean(sent_lengths), 2) if sent_lengths else 0,
                'sentence_length_variation': round(np.std(sent_lengths), 2) if sent_lengths else 0,
                'noun_ratio': round(pos_counts.get('NOUN', 0) / total_tokens * 100, 2) if total_tokens else 0,
                'verb_ratio': round(pos_counts.get('VERB', 0) / total_tokens * 100, 2) if total_tokens else 0,
                'adjective_ratio': round(pos_counts.get('ADJ', 0) / total_tokens * 100, 2) if total_tokens else 0,
                'adverb_ratio': round(pos_counts.get('ADV', 0) / total_tokens * 100, 2) if total_tokens else 0
            }
        except Exception as e:
            print(f"Error analyzing writing style: {e}")
            return {}

    writing_style = analyze_writing_style(doc)

    # Итоговый результат
    return {
        'readability': readability,
        'sentiment': sentiment,
        'topics': topics,
        'keywords': keywords,
        'wordcloud': wordcloud,
        'emotions': dict(emotions),
        'cooccurrence': cooccurrence,
        'lexical_richness': lexical_richness,
        'writing_style': writing_style,
        'sentence_stats': sentence_stats,
        'sentence_length_plot': sentence_length_plot
    }

def analyze_text(filepath):
    try:
        # Проверка существования файла
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")

        # Проверка расширения файла
        if not filepath.lower().endswith(('.txt', '.md', '.csv')):
            logger.warning(f"Анализ файла с нестандартным расширением: {filepath}")

        # Определяем тип файла и обрабатываем соответствующим образом
        file_ext = os.path.splitext(filepath)[1].lower()
        text = None

        # Текстовые файлы
        if file_ext in ['.txt', '.md', '.csv']:
            encodings = ['utf-8', 'cp1251', 'iso-8859-1', 'windows-1252', 'koi8-r']
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        text = f.read().strip()
                    break
                except UnicodeDecodeError:
                    continue

        if text is None:
            raise UnicodeDecodeError("Не удалось декодировать файл с поддерживаемыми кодировками")

        if not text:
            raise ValueError("Файл пуст")

        if len(text) < 50:
            raise ValueError("Текст слишком короткий для анализа (минимум 50 символов)")

        # Проверка на бинарный файл
        if any(char in text for char in ['\x00', '\x01', '\x02', '\x03', '\x04']):
            raise ValueError("Файл похож на бинарный, а не текстовый")

        doc = nlp(text)

        # Базовый анализ
        word_count = len(doc)
        sentences = list(doc.sents)
        char_count = sum(len(word.text) for word in doc)
        avg_word_length = char_count / word_count if word_count > 0 else 0

        # Анализ частей речи
        pos_counts = Counter(token.pos_ for token in doc)
        pos_plot = create_plot(pos_counts, "Распределение частей речи", 'pie')

        # Именованные сущности
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        entity_types = Counter(ent.label_ for ent in doc.ents)
        entity_plot = create_plot(entity_types, "Типы именованных сущностей")

        # Частота слов (исключаем пунктуацию и стоп-слова)
        words = [token.text.lower() for token in doc
                 if not token.is_punct and not token.is_stop and not token.is_space
                 and token.text.strip()]
        word_freq = Counter(words).most_common(20)
        word_freq_plot = create_plot(dict(word_freq), "20 самых частых слов")

        # Анализ тональности (упрощенный)
        sentiment_words = {
            'positive': ['good', 'great', 'excellent', 'happy', 'wonderful', 'amazing', 'perfect',
                         'love', 'best', 'beautiful', 'fantastic', 'super', 'nice', 'awesome'],
            'negative': ['bad', 'terrible', 'awful', 'sad', 'horrible', 'disappointing', 'poor',
                         'hate', 'worst', 'ugly', 'awful', 'terrible', 'problem', 'error']
        }
        sentiment = Counter()
        for word in words:
            for category, terms in sentiment_words.items():
                if word in terms:
                    sentiment[category] += 1

        # Добавляем нейтральную категорию для полноты
        total_sentiment_words = sum(sentiment.values())
        if total_sentiment_words > 0:
            sentiment['neutral'] = len(words) - total_sentiment_words
        else:
            sentiment['neutral'] = len(words)

        sentiment_plot = create_plot(sentiment, "Тональность текста", 'pie')

        # Визуализация зависимостей (только если есть предложения)
        dep_svg = ""
        if len(sentences) > 0:
            dep_svg = displacy.render(doc[:min(10, len(doc))], style="dep", jupyter=False)

        # Анализ длины предложений
        sentence_lengths = [len([token for token in sent if not token.is_punct]) for sent in sentences]
        sentence_length_distribution = Counter()
        for length in sentence_lengths:
            if length < 5:
                sentence_length_distribution['<5'] += 1
            elif length < 10:
                sentence_length_distribution['5-9'] += 1
            elif length < 15:
                sentence_length_distribution['10-14'] += 1
            elif length < 20:
                sentence_length_distribution['15-19'] += 1
            elif length < 25:
                sentence_length_distribution['20-24'] += 1
            else:
                sentence_length_distribution['25+'] += 1

        # Создаем график распределения длины предложений
        sentence_length_plot = create_plot(dict(sentence_length_distribution),
                                           "Распределение длины предложений", 'bar')

        # Анализ стоп-слов
        stopwords_count = sum(1 for token in doc if token.is_stop)
        stopword_ratio = round((stopwords_count / word_count * 100), 2) if word_count > 0 else 0

        # Расширенный анализ
        extended_results = extended_analysis(text)

        # Добавляем статистику по предложениям в расширенный анализ
        extended_results['sentence_stats'] = {
            'avg_sentence_length': round(np.mean(sentence_lengths), 2) if sentence_lengths else 0,
            'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
            'min_sentence_length': min(sentence_lengths) if sentence_lengths else 0,
            'sentence_length_distribution': dict(sentence_length_distribution),
            'complex_sentences': sum(1 for length in sentence_lengths if length > 20),
            'complex_sentences_ratio': round(
                (sum(1 for length in sentence_lengths if length > 20) / len(sentence_lengths) * 100),
                2) if sentence_lengths else 0
        }

        # Собираем результаты
        return {
            'text': text[:1000] + '...' if len(text) > 1000 else text,
            'stats': {
                'word_count': word_count,
                'sentence_count': len(sentences),
                'char_count': char_count,
                'avg_word_length': round(avg_word_length, 2),
                'unique_words': len(set(words)),
                'lexical_diversity': round(len(set(words)) / word_count, 3) if word_count > 0 else 0,
                'avg_sentence_length': round(word_count / len(sentences), 2) if len(sentences) > 0 else 0,
                'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
                'stopword_ratio': stopword_ratio,
                'stopwords_count': stopwords_count
            },
            'pos_tags': [(token.text, token.pos_, token.tag_) for token in doc],
            'entities': entities,
            'word_freq': word_freq,
            'sentiment': sentiment,
            'plots': {
                'pos': pos_plot,
                'entities': entity_plot,
                'word_freq': word_freq_plot,
                'sentiment': sentiment_plot,
                'sentence_length': sentence_length_plot
            },
            'dep_svg': dep_svg,
            'extended': extended_results,
            'file_info': {
                'filename': os.path.basename(filepath),
                'file_size': os.path.getsize(filepath),
                'file_type': file_ext,
                'encoding_detected': 'utf-8'  # Упрощаем для .txt файлов
            }
        }

    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {str(e)}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Ошибка декодирования файла: {str(e)}")
        raise ValueError("Неподдерживаемая кодировка файла")
    except ValueError as e:
        logger.error(f"Ошибка валидации: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка при анализе текста: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Произошла ошибка при анализе файла: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                results = analyze_text(filepath)
                return render_template('results.html', results=results, filename=filename)
            else:
                flash('Invalid file type. Please upload a .txt file', 'error')
                return redirect(request.url)

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('index.html')

@app.errorhandler(413)
def too_large(e):
    return render_template('error.html', message="File is too large (max 1MB)"), 413

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Server error occurred"), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
