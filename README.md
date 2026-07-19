# anki-pipeline

Первый этап локального пайплайна: видео преобразуется в mono WAV 16 kHz, затем
`faster-whisper` создаёт транскрипцию с таймкодами. Генерации Anki-карточек в
проекте пока нет. Опционально локальная модель Ollama группирует исходные
таймкодированные фрагменты в смысловые блоки, а затем извлекает из них атомарные
учебные знания. Knowledge items ещё не являются Anki-карточками.

## Требования

- Windows и Python 3.10–3.12;
- FFmpeg в `PATH`;
- NVIDIA GPU с рабочим CUDA-окружением — необязательно, есть CPU fallback;
- интернет нужен только при первой загрузке модели Whisper.
- Ollama с моделью `qwen3:8b` нужен только для смысловой сегментации.

## Установка

В PowerShell из каталога проекта:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если политика PowerShell блокирует активацию, один раз для текущего окна выполните:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Проверка FFmpeg:

```powershell
ffmpeg -version
```

Проверка доступных типов вычислений CUDA после установки зависимостей:

```powershell
python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
```

Если в списке есть `float16`, автоматический режим выберет GPU. На некоторых
установках Windows могут дополнительно понадобиться совместимые библиотеки CUDA и
cuDNN. При проблемах используйте `--device cpu`.

## Запуск

Поместите видео в `input`, затем запустите:

```powershell
python -m src.pipeline "input/video.mp4"
```

Также поддерживается прямой запуск файла:

```powershell
py src/pipeline.py "input/video.mp4"
```

Все параметры:

```powershell
python -m src.pipeline "input/video.mp4" --model small --device auto
python -m src.pipeline "input/video.mp4" --language ru
python -m src.pipeline "input/video.mp4" --device cpu
python -m src.pipeline "input/video.mp4" --force
python -m src.pipeline "input/video.mp4" --semantic-segment
python -m src.pipeline "input/video.mp4" --extract-knowledge
```

- `--model small` задаёт модель (по умолчанию `small`);
- `--device auto|cuda|cpu` выбирает устройство;
- `--language ru` отключает автоопределение и задаёт язык явно;
- `--force` заново создаёт аудио и транскрипцию.
- `--semantic-segment` создаёт `semantic_segments.json` через локальный Ollama;
- `--ollama-model qwen3:8b` выбирает локальную модель;
- `--ollama-url http://localhost:11434` задаёт адрес API;
- `--ollama-timeout 120` задаёт таймаут одного запроса.
- `--extract-knowledge` извлекает знания из готовых смысловых блоков;
- `--knowledge-language source|ru|en` выбирает язык тезисов и пояснений;
- `--knowledge-min-importance 1` фильтрует менее важные знания после проверки;
- `--knowledge-max-items-per-segment 12` ограничивает результат одного блока.

Без `--force` существующий `audio.wav` используется повторно, а при наличии обоих
файлов транскрипции модель не запускается. Результат для `input/video.mp4`:

Во время распознавания CLI показывает прогресс, прошедшее время и скорость. Если
транскрипция уже существует, для повторного запуска и проверки прогресс-бара
используйте `--force`.

```text
output/video/
├── audio.wav
├── transcript.json
├── transcript.txt
├── semantic_segments.json
└── knowledge_items.json
```

Пример `transcript.txt`:

```text
[00:00:00 - 00:00:05] Текст фрагмента
```

JSON содержит имя исходного файла, определённый язык, длительность, модель и
массив сегментов с точными значениями `start`/`end` в секундах.

## Смысловая сегментация через Ollama

Установите Ollama с официального сайта, затем загрузите одну модель:

```powershell
ollama pull qwen3:8b
```

Запуск на уже созданной транскрипции:

```powershell
python -m src.pipeline "input/video.mp4" --semantic-segment
```

Транскрипция обрабатывается ограниченными окнами с небольшим перекрытием. Ollama
возвращает только названия, summary и ID исходных фрагментов. Полный текст и
таймкоды программа собирает сама, не позволяя модели переписывать оригинал.
Во время ожидания каждого ответа отображается живой индикатор и прошедшее время.

Каждый исходный ID проверяется: пропуски, дубликаты и изменение порядка считаются
ошибкой. Некорректный ответ запрашивается повторно один раз. После второй ошибки
сырой ответ сохраняется в `semantic_segments_diagnostic.json`, а существующий
корректный результат остаётся нетронутым.

## Извлечение знаний

После смысловой сегментации можно получить атомарные учебные тезисы:

```powershell
python -m src.pipeline "input/video.mp4" --extract-knowledge
```

Принудительно создать русскоязычный результат, оставив только importance 2–5:

```powershell
python -m src.pipeline "input/video.mp4" `
  --extract-knowledge `
  --knowledge-language ru `
  --knowledge-min-importance 2 `
  --force
```

`--extract-knowledge` — независимый этап: он читает готовые `transcript.json` и
`semantic_segments.json`, не перезапуская FFmpeg, Whisper или смысловую
сегментацию. Если результат существует, Ollama не вызывается без `--force`.

Разрешённые типы знаний:

- `definition` — определение;
- `fact` — конкретный факт;
- `cause_effect` — причинно-следственная связь;
- `comparison` — сравнение;
- `process` — процесс или последовательность;
- `example` — пример;
- `pitfall` — ошибка, исключение или предостережение.

Пример `knowledge_items.json`:

```json
{
  "source": "video.mp4",
  "model": "qwen3:8b",
  "knowledge_items": [
    {
      "id": "k_0001",
      "semantic_segment_id": 0,
      "type": "definition",
      "statement": "Yojijukugo are Japanese four-character idioms.",
      "explanation": "They express a complete idea in a compact form.",
      "importance": 4,
      "source_transcript_segment_ids": [3, 4],
      "evidence": "Yojijukugo is exactly what it sounds like. Four plus character plus idiom."
    }
  ]
}
```

Ollama выбирает только тип, тезис, пояснение, importance и минимальные source IDs.
Evidence программа всегда собирает сама из оригинальной транскрипции. Регистр и
пробелы нормализуются только при проверке; в файл записывается исходный текст.

Чтобы вручную проверить знание, найдите значения из
`source_transcript_segment_ids` в массиве `segments` файла `transcript.json`.
Evidence должен точно состоять из текста этих фрагментов.

Локальная модель может пропустить неочевидный факт, поэтому приоритет этого этапа
— подтверждаемость, а не максимальная полнота. После двух некорректных ответов
создаётся `knowledge_items_diagnostic.json`, существующий корректный результат не
повреждается.

## Тесты

```powershell
python -m pytest
```
