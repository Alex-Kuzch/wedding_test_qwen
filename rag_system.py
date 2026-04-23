import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any
import re


class SemanticChunker:
    """Семантическая нарезка текста на основе изменения смысла"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.max_chunk_size = 500  # максимальный размер чанка в символах
        self.min_chunk_size = 100  # минимальный размер чанка
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Разбиение текста на предложения"""
        # Простое разбиение по точкам, восклицательным и вопросительным знакам
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Семантическая нарезка текста на чанки
        
        Args:
            text: исходный текст
            threshold: порог схожести для объединения предложений в один чанк
            
        Returns:
            список чанков с метаданными
        """
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Получаем эмбеддинги для всех предложений
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            embedding = embeddings[i]
            
            # Вычисляем схожесть текущего предложения с текущим чанком
            similarity = cosine_similarity([embedding], [current_embedding])[0][0]
            
            # Если схожесть ниже порога или достигнут максимальный размер - начинаем новый чанк
            current_text = ' '.join(current_chunk)
            if similarity < threshold or len(current_text) + len(sentence) > self.max_chunk_size:
                # Сохраняем текущий чанк если он достаточно большой
                if len(current_text) >= self.min_chunk_size:
                    chunks.append({
                        'text': current_text,
                        'embedding': current_embedding,
                        'sentence_count': len(current_chunk)
                    })
                
                # Начинаем новый чанк
                current_chunk = [sentence]
                current_embedding = embedding
            else:
                # Добавляем предложение в текущий чанк
                current_chunk.append(sentence)
                # Обновляем средний эмбеддинг чанка
                current_embedding = np.mean([current_embedding, embedding], axis=0)
        
        # Добавляем последний чанк
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
    """Локальная векторизация с использованием sentence-transformers"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Векторизация списка текстов"""
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Векторизация поискового запроса"""
        return self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]


class Reranker:
    """Переранжирование результатов поиска"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.cross_encoder = CrossEncoder(model_name)
        
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Переранжирование документов относительно запроса
        
        Args:
            query: поисковый запрос
            documents: список документов для переранжирования
            top_k: количество лучших результатов
            
        Returns:
            список кортежей (индекс_документа, оценка)
        """
        if not documents:
            return []
        
        # Создаем пары (запрос, документ) для cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Получаем оценки релевантности
        scores = self.cross_encoder.predict(pairs)
        
        # Сортируем по убыванию оценок
        indexed_scores = list(enumerate(scores))
        sorted_results = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]


class RAGSystem:
    """Полноценная RAG система с семантической нарезкой, локальной векторизацией и переранжированием"""
    
    def __init__(self, 
                 chunker_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 vectorizer_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        
        self.chunker = SemanticChunker(chunker_model)
        self.vectorizer = LocalVectorizer(vectorizer_model)
        self.reranker = Reranker(reranker_model)
        
        self.documents = []  # оригинальные документы
        self.chunks = []     # нарезанные чанки
        self.embeddings = None  # матрица эмбеддингов
        self.chunk_metadata = []  # метаданные чанков
        
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Добавление документа в систему"""
        doc_id = len(self.documents)
        self.documents.append(text)
        
        # Семантическая нарезка
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
            raise ValueError("Нет чанков для индексации. Сначала добавьте документы.")
        
        print("Векторизация чанков...")
        self.embeddings = self.vectorizer.embed_texts(self.chunks)
        print(f"Индекс построен: {len(self.chunks)} чанков, размерность {self.embeddings.shape[1]}")
    
    def search(self, query: str, top_k: int = 10, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Поиск релевантных чанков
        
        Args:
            query: поисковый запрос
            top_k: количество результатов до переранжирования
            use_reranking: использовать ли переранжирование
            
        Returns:
            список результатов с текстом и метаданными
        """
        if self.embeddings is None:
            raise ValueError("Индекс не построен. Вызовите build_index().")
        
        # Векторизация запроса
        query_embedding = self.vectorizer.embed_query(query)
        
        # Поиск ближайших соседей через косинусное сходство
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Получаем top_k результатов
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        candidate_chunks = [self.chunks[i] for i in top_indices]
        candidate_metadata = [self.chunk_metadata[i] for i in top_indices]
        candidate_scores = [similarities[i] for i in top_indices]
        
        # Переранжирование
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
        """
        Генерация ответа на основе найденного контекста
        
        В реальной системе здесь будет вызов LLM
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "Извините, я не нашел информацию по вашему запросу."
        
        # Формируем контекст из найденных чанков
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # В реальной системе здесь был бы вызов LLM с промптом
        response = f"""На основе найденной информации:

{context}

Рекомендация: Используйте найденные данные для формирования ответа пользователю.
В продакшене здесь будет интеграция с LLM (GPT, Claude, или локальная модель)."""
        
        return response


# Пример использования
if __name__ == "__main__":
    # Инициализация RAG системы
    rag = RAGSystem()
    
    # Пример данных для свадьбы
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
    
    # Добавление документа
    rag.add_document(wedding_content, {'source': 'wedding_guide', 'category': 'planning'})
    
    # Построение индекса
    rag.build_index()
    
    # Поиск и генерация ответа
    query = "Как распределить бюджет на свадьбу?"
    print(f"\nЗапрос: {query}")
    print("=" * 50)
    
    response = rag.generate_response(query)
    print(response)
    
    # Поиск с деталями
    print("\n" + "=" * 50)
    print("Детали поиска:")
    results = rag.search("Стоимость услуг фотографов и видеографов", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nРезультат {i}:")
        print(f"Текст: {result['text'][:100]}...")
        print(f"Оценка релевантности: {result['score']:.4f}")
        print(f"Метаданные: {result['metadata']}")