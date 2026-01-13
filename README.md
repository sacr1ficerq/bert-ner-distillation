# Сжатие BERT для задачи Named Entity Recognition (NER)

Этот репозиторий содержит эксперименты по сжатию модели `bert-base-cased` для задачи Named Entity Recognition (NER) на датасете **CoNLL-2003**. Основная цель — уменьшить размер модели больше чем в 5 раз, сохраняя при этом высокий F1-score.

## Реализованные методы

### 1. Embedding Factorization
Раскладывает большую матрицу эмбеддингов ($V \times H$) на две матрицы меньшего размера ($V \times E$ и $E \times H$) используя **SVD initialization**.
- **Цель:** Уменьшить размер слоя эмбеддингов, который в исходной модели занимает ~20% параметров.
- **Реализация:** Кастомный класс `EmbeddingWrapper`, заменяющий стандартные эмбеддинги BERT.

### 2. Parameter Sharing
Сокращает общее количество уникальных параметров за счет переиспользования весов между разными слоями.
- **Attention Sharing:** Общие веса для Attention во всех слоях энкодера.
- **Attention + FFN Sharing:** Общие веса как для механизмов Attention, так и для Feed-Forward Networks (FFN) во всех слоях. Этот метод дал самое значительное сокращение параметров (~11.3M).

### 3. Layer Factorization
Аппроксимирует **dense linear layers** используя **low-rank decomposition**.
- **Техника:** Замена стандартных линейных слоев на их факторизованные версии для сокращения вычислений и памяти.
- **Результат:** Обучение оказалось сложным, что привело к более низкому F1 по сравнению с другими методами.

### 4. Knowledge Distillation
Обучение меньшей модели "**Student**", которая имитирует поведение большой предобученной модели "**Teacher**".
- **Loss Function:** Комбинация стандартной **Cross-Entropy** (hard labels) и **KL-Divergence** (soft labels от учителя) с применением temperature scaling ($\tau > 1$).
- **Teacher:** Fine-tuned `bert-base-cased` (F1 ~0.94).
- **Student:** Компактная архитектура, оптимизированная для быстрого инференса.

## Результаты

Сравнение производительности реализованных методов на Test set:

| Method | Parameters (M) | F1 Score | Runtime (s) |
| :--- | :--- | :--- | :--- |
| **Teacher (Baseline)** | **107.72** | **0.9420** | **-** |
| Embedding Factorization | 87.36 | 0.8713 | 3.90 |
| Attention Sharing | 63.26 | 0.8351 | 3.57 |
| **Attention + FFN Sharing** | **11.30** | **0.8237** | **3.53** |
| Layer Factorization | 78.57 | 0.7485 | 4.33 |
| Student (Distillation) | 16.27 | 0.7586 | 2.06 |

## Использование

1. **Prerequisites:** Установите зависимости (transformers, datasets, seqeval, torch).
2. **Data:** Скрипты автоматически загружают датасет CoNLL-2003 через Hugging Face.
3. **Training:** Запустите `hw3_distillation.py` для выполнения пайплайна обучения, включая токенизацию, выравнивание меток (label alignment) и эксперименты по сжатию.
