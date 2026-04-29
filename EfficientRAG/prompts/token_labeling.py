"""Prompts for token-level labeling of chunk passages."""

# --- English ---

TOKEN_LABELING_PROMPT_EN = """You are given a question and a supporting paragraph. Extract ONLY the words from the paragraph that are essential to answer the question.

Rules:
1. Extract ONLY words that appear exactly in the paragraph text
2. Do NOT reorder, change, miss, or add any words
3. Extract words that are in the question or form the answer
4. Never extract words that lack meaningful information (articles, prepositions alone)
5. Maintain the original order of extracted words

Question: {question}
Answer: {answer}

Paragraph:
{paragraph}

Extracted essential words (comma-separated):"""

QUERY_FILTERING_PROMPT_EN = """You are given a question and known information. Refactor the question by:
1. Keeping parts of the question that are NOT answered by the known information
2. Replacing unknown references with concrete entities from the known information
3. You can ONLY pick words from the question and known information — do NOT generate new words

Question: {question}
Known information: {known_info}

Refactored question (using only words from above):"""

# --- Russian ---

TOKEN_LABELING_PROMPT_RU = """Тебе дан вопрос и параграф-источник. Извлеки ТОЛЬКО те слова из параграфа, которые необходимы для ответа на вопрос.

Правила:
1. Извлекай ТОЛЬКО слова, которые точно присутствуют в тексте параграфа
2. НЕ переставляй, не изменяй, не пропускай и не добавляй слова
3. Извлекай слова, которые есть в вопросе или составляют ответ
4. Не извлекай слова без смысловой нагрузки (предлоги, союзы отдельно)
5. Сохраняй исходный порядок слов

Вопрос: {question}
Ответ: {answer}

Параграф:
{paragraph}

Извлечённые ключевые слова (через запятую):"""

QUERY_FILTERING_PROMPT_RU = """Тебе дан вопрос и известная информация. Переформулируй вопрос:
1. Оставь части вопроса, на которые известная информация НЕ отвечает
2. Замени неизвестные ссылки конкретными сущностями из известной информации
3. Используй ТОЛЬКО слова из вопроса и известной информации — НЕ генерируй новые слова

Вопрос: {question}
Известная информация: {known_info}

Переформулированный вопрос (только из слов выше):"""

# Default: Russian (for dragon-derec)
TOKEN_LABELING_PROMPT = TOKEN_LABELING_PROMPT_RU
QUERY_FILTERING_PROMPT = QUERY_FILTERING_PROMPT_RU
