# anki-pipeline

## Готовая колода Anki

После создания кандидатов и необязательного подбора изображений соберите
импортируемую `.apkg` одной независимой командой:

```powershell
python -m src.pipeline "input/video.mkv" `
  --build-anki-deck `
  --deck-name "Название курса"
```

Команда читает `card_candidates.json`, а при наличии также
`target_expressions.json` и `card_media.json`. Она не запускает Whisper, FFmpeg
или Ollama. Готовые файлы:

```text
output/<video>/<video>.apkg
output/<video>/anki_export_manifest.json
```

Manifest хранит стабильные GUID, связи с card/knowledge/expression IDs и имена
медиа. В колоду автоматически включаются выражения `verified` и `transcribed`; отсутствие
выражений или изображения не мешает экспорту. Повторный запуск без `--force`
использует готовую колоду, а стабильные GUID позволяют обновлять заметки при
повторном импорте в Anki.

## Кадры для карточек

Независимый локальный этап читает готовые `transcript.json` и
`card_candidates.json`, извлекает через FFmpeg несколько кадров внутри точных
таймкодов карточки и выбирает технически лучший кадр с помощью Pillow и NumPy:

```powershell
python -m src.pipeline "input/video.mkv" `
  --extract-card-images `
  --frames-per-card 5 `
  --image-policy auto
```

`auto` отбрасывает чёрные, размытые, почти одинаковые и слишком слабые кадры.
`always` выбирает лучший технически пригодный кадр даже при низкой оценке. `never`
не вызывает FFmpeg и явно записывает отсутствие изображений. `--force`
пересоздаёт только этот этап; без него готовый manifest переиспользуется.

Результат находится в `output/<video>/card_media.json`, а выбранные JPEG — в
`output/<video>/media/<card_id>.jpg`. Пример элемента manifest:

```json
{
  "card_id": "c_0001",
  "source_transcript_segment_ids": [12, 13],
  "start": 41.2,
  "end": 49.8,
  "image": {
    "path": "media/c_0001.jpg",
    "timestamp": 45.5,
    "width": 1280,
    "height": 720,
    "selection_method": "local_quality_score",
    "quality_score": 0.81,
    "metrics": {"brightness_mean": 121.4, "dark_ratio": 0.01, "sharpness": 84.3}
  }
}
```

`image: null` — нормальный результат; рядом сохраняются причина отказа и способ
выбора. Этап оценивает только технические свойства изображения: OCR, vision-модель
и понимание содержимого слайда пока не используются. Для `auto` и `always` нужен
FFmpeg в `PATH`.

## Выражения изучаемого языка

Транскрипт и объяснения могут быть на английском, а изучаемые выражения — на японском
или другом языке. Исходный transcript.json не изменяется. Отдельный этап создаёт
`target_expressions.json` с оригинальным написанием, чтением, романизацией, связями
с knowledge items и исходными таймкодами:

```cmd
py -m src.pipeline "input/video.mkv" --extract-target-expressions --target-language ja
```

Для смешанного видео после чернового извлечения запустите выборочный второй
Whisper-проход на изучаемом языке, затем уточните выражения:

```cmd
py -m src.pipeline "input/video.mkv" --transcribe-target --target-language ja --target-device cuda
py -m src.pipeline "input/video.mkv" --extract-target-expressions --target-language ja --force
```

Второй проход обрабатывает только интервалы с найденными примерами и сохраняет
`target_transcript.json`; вся английская дорожка как японская не распознаётся.
Форма получает статус `transcribed`, только если нативный текст второго прохода
совпадает с романизацией основной транскрипции на том же таймкоде. Путь к локальной
CTranslate2-модели Subtitle Edit можно передать через `--target-whisper-model`.

После этого повторно создайте кандидаты карточек:

```cmd
py -m src.pipeline "input/video.mkv" --generate-card-candidates --force
```

Карточки получают `expression_ids`, а оригинальное написание гарантированно добавляется
в `extra`. Варианты, предложенные только Ollama, помечаются `needs_review`: это защищает
от незаметной подмены ошибочно распознанной романизации. Код языка универсален, например
`ko`, `zh` или `de`.

В карточки автоматически попадают выражения `verified` и `transcribed`.
Предложения `needs_review` сохраняются для проверки, но не загрязняют готовые карточки.
В `front` и `back` подтверждённая ромадзи заменяется оригинальным написанием. Если
Whisper исказил ромадзи и точная замена невозможна, японская форма добавляется в
начало `front`, а чтение и ромадзи остаются подсказкой в `extra`.

Поддерживаются `term`, `grammar_pattern` и `example_sentence`. Транскрипт
обрабатывается небольшими окнами с overlap, поэтому примеры из длинного видео не
теряются и весь файл не отправляется в один запрос.

Локальный глоссарий `glossaries/<код>.json` имеет приоритет над предположением Ollama.
Совпавшая запись получает статус `verified`. Свой файл можно указать через
`--target-glossary "путь\\к\\glossary.json"`.

Проект локально создаёт транскрипцию, знания, языковой слой, кандидаты карточек,
изображения и готовую импортируемую `.apkg`-колоду.

## Требования

- Windows и Python 3.10–3.13;
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
python -m src.pipeline "input/video.mp4" --review-knowledge
python -m src.pipeline "input/video.mp4" --generate-card-candidates
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
- `--review-knowledge` оценивает качество готовых знаний;
- `--generate-card-candidates` создаёт текстовые кандидаты карточек.

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
├── knowledge_items.json
├── knowledge_review.json
└── card_candidates.json
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

## Knowledge Review

Review — отдельный редакторский этап, который не меняет знания, а оценивает их
пригодность для будущих карточек:

```powershell
python -m src.pipeline "input/video.mp4" --review-knowledge
```

Каждый knowledge item получает один из статусов `keep`, `merge`, `split` или
`discard`, confidence от 0 до 1, рекомендуемое действие и список проблем. Среди
проблем поддерживаются `too_general`, `too_specific`, `contains_two_facts`,
`duplicate`, `weak_example`, `redundant`, `unclear` и `low_learning_value`.

Пример записи в `knowledge_review.json`:

```json
{
  "knowledge_id": "k_0001",
  "status": "keep",
  "confidence": 0.94,
  "issues": [],
  "recommended_action": "none",
  "reason": "The item contains one independently testable fact."
}
```

Программа проверяет полное покрытие knowledge ID, все enum-значения и confidence.
После второй ошибки сохраняется `knowledge_review_diagnostic.json`, не повреждая
существующий корректный review.

## Card Candidates

Следующий независимый этап читает знания и review:

```powershell
python -m src.pipeline "input/video.mp4" --generate-card-candidates
```

`discard` не создаёт карточек, `split` даёт 2–5 кандидатов, а совместимые `merge`
могут стать одной карточкой. Поддерживаются только `basic`, `reverse`, `cloze`,
`comparison`, `process`, `mistake` и `definition`.

Пример `card_candidates.json`:

```json
{
  "card_id": "c_0001",
  "knowledge_ids": ["k_0001"],
  "card_type": "definition",
  "front": "What does datte consist of?",
  "back": "It consists of the copula da and the quotation form te.",
  "extra": "",
  "difficulty": 3,
  "importance": 5,
  "tags": ["definition"],
  "source_transcript_segment_ids": [7, 8]
}
```

`card_id`, importance, tags и source IDs формирует программа. Она проверяет
пустые и повторяющиеся вопросы, допустимые типы, длину front до 120 символов,
длину back до 500 символов и отсутствие карточек из discard-знаний.

Это ещё не Anki: кандидаты не содержат HTML-шаблонов, deck/note types,
планирования повторений и не экспортируются в CSV, APKG или AnkiConnect. Они
являются проверенным промежуточным JSON для следующего этапа разработки.

## Тесты

```powershell
python -m pytest
```
