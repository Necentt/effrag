"""Prompts for multi-hop question decomposition."""

# --- English prompts (HotpotQA, MuSiQue, 2WikiMQA) ---

HOTPOTQA_DECOMPOSE_PROMPT = """You are given a multi-hop question and its supporting paragraphs. Decompose the question into simpler single-hop sub-questions, each answerable by exactly one supporting paragraph.

For each sub-question, provide:
1. The sub-question text
2. The answer to the sub-question
3. Which supporting paragraph answers it
4. Dependencies on previous sub-questions (if any)

Output as JSON array:
[
  {
    "sub_question": "...",
    "answer": "...",
    "paragraph_idx": 0,
    "depends_on": []
  },
  {
    "sub_question": "...",
    "answer": "...",
    "paragraph_idx": 1,
    "depends_on": [0]
  }
]

Question: {question}
Answer: {answer}

Supporting paragraphs:
{paragraphs}

Decomposed sub-questions (JSON):"""

MUSIQUE_DECOMPOSE_PROMPT = HOTPOTQA_DECOMPOSE_PROMPT
WIKIMQA_DECOMPOSE_PROMPT = HOTPOTQA_DECOMPOSE_PROMPT

# --- Russian prompt (dragon-derec) ---

DRAGON_DEREC_DECOMPOSE_PROMPT = """Тебе дан сложный вопрос, для ответа на который нужна информация из нескольких параграфов. Разбей вопрос на простые подвопросы, каждый из которых можно ответить по одному параграфу.

Для каждого подвопроса укажи:
1. Текст подвопроса
2. Ответ на подвопрос
3. Индекс параграфа, который отвечает на подвопрос
4. Зависимости от предыдущих подвопросов (если есть)

Верни JSON массив:
[
  {{
    "sub_question": "...",
    "answer": "...",
    "paragraph_idx": 0,
    "depends_on": []
  }},
  {{
    "sub_question": "...",
    "answer": "...",
    "paragraph_idx": 1,
    "depends_on": [0]
  }}
]

Вопрос: {question}
Ответ: {answer}

Параграфы:
{paragraphs}

Подвопросы (JSON):"""

DATASET_PROMPTS = {
    "hotpotQA": HOTPOTQA_DECOMPOSE_PROMPT,
    "2wikimqa": WIKIMQA_DECOMPOSE_PROMPT,
    "musique": MUSIQUE_DECOMPOSE_PROMPT,
    "dragon-derec": DRAGON_DEREC_DECOMPOSE_PROMPT,
}
