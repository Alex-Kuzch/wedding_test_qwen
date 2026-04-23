import numpy as np
from typing import List, Tuple, Dict, Any
import re
import json


class SimpleEmbeddingModel:
    """
    Упрощенная модель векторизации на основе TF-IDF + PCA
    Для демонстрации без загрузки больших нейросетей
    """
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocabulary = {}
        self.idf = {}
        self.fitted = False
        
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        text = text.lower()
        tokens = re.findall(r'\b[a-zа-яё]+\b', text)
        return tokens
    
    def fit(self, texts: List[str]):
        """Обучение модели на текстах"""
        # Построение словаря
        word_doc_count = {}
        all_words = set()
        
        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                all_words.add(token)
                word_doc_count[token] = word_doc_count.get(token, 0) + 1
        
        # Выбор наиболее частых слов
        sorted_words = sorted(word_doc_count.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, count in sorted_words[:self.vocab_size]]
        
        self.vocabulary = {word: idx for idx, word in enumerate(top_words)}
        
        # Вычисление IDF
        n_docs = len(texts)
        for word, count in word_doc_count.items():
            if word in self.vocabulary:
                self.idf[word] = np.log((n_docs + 1) / (count + 1)) + 1
        
        self.fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Векторизация текстов"""
        if not self.fitted:
            raise ValueError("Модель не обучена. Вызовите fit().")
        
        vectors = []
        for text in texts:
            tokens = self._tokenize(text)
            vector = np.zeros(len(self.vocabulary))
            
            # TF (term frequency)
            word_count = {}
            for token in tokens:
                if token in self.vocabulary:
                    word_count[token] = word_count.get(token, 0) + 1
            
            # TF-IDF
            for word, count in word_count.items():
                idx = self.vocabulary[word]
                tf = count / len(tokens) if tokens else 0
                vector[idx] = tf * self.idf.get(word, 1.0)
            
            # L2 нормализация
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def encode(self, texts: List[str], show_progress_bar: bool = False, normalize_embeddings: bool = True) -> np.ndarray:
        """Универсальный метод для совместимости с sentence-transformers API"""
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.fitted:
            # Если модель не обучена, обучаем на лету (для демо)
            self.fit(texts)
        
        return self.transform(texts)


class SemanticChunker:
    """Семантическая нарезка текста на основе изменения смысла"""
    
    def __init__(self, model=None):
        self.model = model or SimpleEmbeddingModel()
        self.max_chunk_size = 500
        self.min_chunk_size = 100
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Разбиение текста на предложения"""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Семантическая нарезка текста на чанки"""
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Получаем эмбеддинги для всех предложений
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            embedding = embeddings[i]
            
            # Косинусное сходство
            similarity = np.dot(embedding, current_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(current_embedding) + 1e-8
            )
            
            current_text = ' '.join(current_chunk)
            if similarity < threshold or len(current_text) + len(sentence) > self.max_chunk_size:
                if len(current_text) >= self.min_chunk_size:
                    chunks.append({
                        'text': current_text,
                        'embedding': current_embedding,
                        'sentence_count': len(current_chunk)
                    })
                
                current_chunk = [sentence]
                current_embedding = embedding
            else:
                current_chunk.append(sentence)
                current_embedding = np.mean([current_embedding, embedding], axis=0)
        
        # Последний чанк
        if current_chunk:
            current_text = ' '.join(current_chunk)
            if len(current_text) >= self.min_chunk_size:
                chunks.append({
                    'text': current_text,
                    'embedding': current_embedding,
                    'sentence_count': len(current_chunk)
                })
        
        return chunks


class LocalVectorizer:
    """Локальная векторизация"""
    
    def __init__(self, model=None):
        self.model = model or SimpleEmbeddingModel()
        self.dimension = 5000  # размерность по умолчанию
        
    def fit(self, texts: List[str]):
        """Обучение векторизатора"""
        self.model.fit(texts)
        self.dimension = len(self.model.vocabulary)
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Векторизация списка текстов"""
        return self.model.encode(texts, normalize_embeddings=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Векторизация поискового запроса"""
        return self.model.encode([query], normalize_embeddings=True)[0]


class Reranker:
    """Переранжирование результатов поиска на основе ключевых слов"""
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """Переранжирование документов относительно запроса"""
        if not documents:
            return []
        
        # Токенизация запроса
        query_tokens = set(re.findall(r'\b[a-zа-яё]+\b', query.lower()))
        
        scores = []
        for doc in documents:
            doc_tokens = set(re.findall(r'\b[a-zа-яё]+\b', doc.lower()))
            
            # Jaccard similarity
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            score = intersection / union if union > 0 else 0
            
            # Дополнительно: boost за точные совпадения длинных слов
            for token in query_tokens:
                if len(token) > 4 and token in doc.lower():
                    score += 0.1
            
            scores.append(score)
        
        # Сортировка
        indexed_scores = list(enumerate(scores))
        sorted_results = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]


class RAGSystem:
    """RAG система с семантической нарезкой, локальной векторизацией и переранжированием"""
    
    def __init__(self):
        self.chunker = SemanticChunker()
        self.vectorizer = LocalVectorizer()
        self.reranker = Reranker()
        
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.chunk_metadata = []
        
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Добавление документа в систему"""
        doc_id = len(self.documents)
        self.documents.append(text)
        
        doc_chunks = self.chunker.chunk_text(text)
        
        for chunk in doc_chunks:
            self.chunks.append(chunk['text'])
            self.chunk_metadata.append({
                'doc_id': doc_id,
                'metadata': metadata or {},
                'chunk_text': chunk['text']
            })
        
        print(f"Добавлен документ {doc_id}, нарезан на {len(doc_chunks)} чанков")
    
    def build_index(self):
        """Построение векторного индекса"""
        if not self.chunks:
            raise ValueError("Нет чанков для индексации.")
        
        print("Векторизация чанков...")
        self.vectorizer.fit(self.chunks)
        self.embeddings = self.vectorizer.embed_texts(self.chunks)
        print(f"Индекс построен: {len(self.chunks)} чанков, размерность {self.embeddings.shape[1]}")
    
    def search(self, query: str, top_k: int = 10, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """Поиск релевантных чанков"""
        if self.embeddings is None:
            raise ValueError("Индекс не построен.")
        
        query_embedding = self.vectorizer.embed_query(query)
        
        similarities = np.dot(self.embeddings, query_embedding)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        candidate_chunks = [self.chunks[i] for i in top_indices]
        candidate_metadata = [self.chunk_metadata[i] for i in top_indices]
        candidate_scores = [float(similarities[i]) for i in top_indices]
        
        if use_reranking and len(candidate_chunks) > 1:
            print("Выполнение переранжирования...")
            reranked = self.reranker.rerank(query, candidate_chunks, top_k=top_k)
            
            results = []
            for orig_idx, score in reranked:
                results.append({
                    'text': candidate_chunks[orig_idx],
                    'score': float(score),
                    'metadata': candidate_metadata[orig_idx],
                    'original_similarity': float(candidate_scores[orig_idx])
                })
        else:
            results = []
            for i, idx in enumerate(top_indices):
                results.append({
                    'text': self.chunks[idx],
                    'score': float(similarities[idx]),
                    'metadata': self.chunk_metadata[idx],
                    'original_similarity': float(similarities[idx])
                })
        
        return results
    
    def generate_response(self, query: str, top_k: int = 5) -> str:
        """Генерация ответа на основе найденного контекста"""
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "Извините, я не нашел информацию по вашему запросу."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result['text']}")
        
        context = "\n\n".join(context_parts)
        
        response = f"""На основе найденной информации:

{context}

Рекомендация: Используйте найденные данные для формирования ответа пользователю.
В продакшене здесь будет интеграция с LLM."""
        
        return response
    
    def save_index(self, filepath: str):
        """Сохранение индекса в файл"""
        data = {
            'chunks': self.chunks,
            'metadata': self.chunk_metadata,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'dimension': self.vectorizer.dimension,
            'vocabulary': self.vectorizer.model.vocabulary,
            'idf': self.vectorizer.model.idf
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Индекс сохранен в {filepath}")
    
    def load_index(self, filepath: str):
        """Загрузка индекса из файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        self.chunk_metadata = data['metadata']
        self.embeddings = np.array(data['embeddings']) if data['embeddings'] else None
        
        self.vectorizer.model.vocabulary = data['vocabulary']
        self.vectorizer.model.idf = data['idf']
        self.vectorizer.model.fitted = True
        self.vectorizer.dimension = data['dimension']
        
        print(f"Индекс загружен из {filepath}: {len(self.chunks)} чанков")


if __name__ == "__main__":
    rag = RAGSystem()
    
    wedding_content = """
    Организация свадьбы требует тщательного планирования бюджета. 
    Средняя стоимость свадьбы в Москве составляет от 500 000 до 2 000 000 рублей.
    Основные статьи расходов: банкет (40-50%), фотограф и видеограф (15-20%), платье и костюм (10-15%), 
    цветы и декор (10-15%), ведущий и музыка (5-10%).
    
    Выбор площадки для проведения свадьбы - один из самых важных этапов.
    Популярные варианты: рестораны, загородные клубы, лофты, открытые площадки летом.
    Бронировать площадку рекомендуется за 6-12 месяцев до даты свадьбы.
    Вместимость зала должна соответствовать количеству гостей с запасом 10-15%.
    
    Работа с вендорами требует заключения договоров.
    Обязательно уточняйте условия отмены и переноса даты.
    Предоплата обычно составляет 30-50% от общей стоимости услуг.
    Проверяйте портфолио и отзывы перед выбором исполнителя.
    
    Тайминг свадебного дня должен быть реалистичным.
    Стандартный сценарий: сборы жениха и невесты (утро), выкуп (опционально), 
    регистрация в ЗАГСе, фотосессия, банкет (5-6 часов).
    Между этапами оставляйте буферное время 30-60 минут на непредвиденные задержки.
    """
    
    rag.add_document(wedding_content, {'source': 'wedding_guide', 'category': 'planning'})
    rag.build_index()
    
    query = "Как распределить бюджет на свадьбу?"
    print(f"\nЗапрос: {query}")
    print("=" * 50)
    
    response = rag.generate_response(query)
    print(response)
    
    print("\n" + "=" * 50)
    print("Детали поиска:")
    results = rag.search("Стоимость услуг фотографов и видеографов", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nРезультат {i}:")
        print(f"Текст: {result['text'][:100]}...")
        print(f"Оценка релевантности: {result['score']:.4f}")
        print(f"Метаданные: {result['metadata']}")
    
    # Сохранение и загрузка индекса
    rag.save_index('wedding_index.json')
    
    # Создание новой системы и загрузка индекса
    rag2 = RAGSystem()
    rag2.load_index('wedding_index.json')
    
    print("\n" + "=" * 50)
    print("Проверка загруженного индекса:")
    results2 = rag2.search("Когда бронировать площадку?", top_k=2)
    for i, result in enumerate(results2, 1):
        print(f"Результат {i}: {result['text'][:80]}...")