# anki-pipeline

## Шаг 0: источник видео

Pipeline принимает обычный локальный файл без копирования:

```cmd
py -m src.source_ingest "D:\Videos\lesson.mp4"
```

Либо безопасно подготавливает одиночное видео по YouTube-ссылке через официальный
Python-пакет `yt-dlp`:

```cmd
py -m src.source_ingest "https://www.youtube.com/watch?v=VIDEO_ID" --language en --target-language ja
```

Последняя строка команды — локальный путь, который затем передаётся в существующий
`src.pipeline`. Загрузки сохраняются в:

```text
input/downloads/<extractor>/<video_id>/video.<ext>
```

Рядом сохраняются `video.info.json`, доступные обычные/автоматические субтитры
для указанных языков и `source_manifest.json`. Общий `source_index.json` позволяет
повторно использовать скачанный файл без обращения к сети. Плейлисты отключены,
разрешены только `youtube.com` и `youtu.be`, удалённые компоненты yt-dlp запрещены,
имена файлов совместимы с Windows. Для повторной загрузки используется `--force`.

Будущий GUI будет вызывать одну функцию `prepare_source(...)`: URL превратится в
локальный файл, а загруженное пользователем видео пройдёт дальше без изменений.
Скачивайте только материалы, на использование которых у вас есть право.

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

В GUI языковые данные обрабатываются до смыслового упрощения, в таком порядке:

1. транскрипция основного аудио;
2. черновое извлечение выражений изучаемого языка;
3. второй Whisper-проход по найденным интервалам;
4. смысловая сегментация с основным и уточнённым target-language текстом;
5. извлечение знаний;
6. Knowledge Review;
7. кандидаты карточек;
8. изображения.

`semantic_segments.json` сохраняет уточнённые языковые фрагменты в поле
`target_language_text`. Если ранний артефакт обновился, GUI автоматически пересобирает
только устаревшие зависимые стадии; глобальный `Force` для этого не требуется.

В дополнительных параметрах GUI есть переключатель «Язык карточек»: English,
Русский или «Как в источнике». Выбор применяется к semantic title/summary, знаниям и
кандидатам карточек; по умолчанию выбран English. При смене языка зависимые артефакты
пересобираются автоматически.

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

## Gemini API или локальный Ollama

Ollama остаётся режимом по умолчанию. Для облачного режима создайте ключ в
[Google AI Studio](https://aistudio.google.com/app/apikey). Скопируйте `.env.example`
в локальный `.env`, укажите там ключ и выберите `LLM_PROVIDER=gemini`. Приложение
автоматически прочитает `.env` из корня проекта; файл исключён из Git. Уже заданные
системные переменные имеют приоритет над значениями `.env`.

> При выборе Gemini текст транскрипта, субтитры, выделенные фрагменты и другой текст,
> включённый в prompt, отправляются во внешний Google API. Видео, аудио и изображения
> в Gemini не загружаются. Бесплатная квота ограничена; после её исчерпания API вернёт
> понятную ошибку и не переключит pipeline на Ollama автоматически.

PowerShell:

```powershell
$env:GEMINI_API_KEY="..."
$env:LLM_PROVIDER="gemini"
$env:GEMINI_MODEL="gemini-3.1-flash-lite"
py -m src.pipeline "input/video.mp4" --semantic-segment
```

cmd:

```bat
set GEMINI_API_KEY=...
set LLM_PROVIDER=gemini
set GEMINI_MODEL=gemini-3.1-flash-lite
py -m src.pipeline "input/video.mp4" --semantic-segment
```

Linux/macOS:

```bash
export GEMINI_API_KEY="..."
export LLM_PROVIDER="gemini"
export GEMINI_MODEL="gemini-3.1-flash-lite"
python -m src.pipeline "input/video.mp4" --semantic-segment
```

CLI-параметры `--llm-provider` и `--gemini-model` имеют приоритет над окружением.
Вернуться к локальному режиму можно через `LLM_PROVIDER=ollama` либо
`--llm-provider ollama`. Небольшая проверка без полного pipeline:

```powershell
py -c "from src.config import LLMConfig; from src.llm_factory import create_llm_provider; print(create_llm_provider(LLMConfig.from_env()).generate_json('Return an object with ok=true', schema={'type':'object','required':['ok'],'properties':{'ok':{'type':'boolean'}}})[0])"
```

Unit-тесты не выполняют реальные запросы: `py -m pytest -q`. Модель по умолчанию
задаётся одной константой и может быть недоступна отдельному аккаунту — в этом случае
выберите доступную модель в Google AI Studio и задайте `GEMINI_MODEL`.

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
# Локальный браузерный GUI

GUI запускается одной командой и по умолчанию доступен только на локальном адресе:

```powershell
python -m src.gui
```

Параметры адреса можно изменить явно: `python -m src.gui --host 127.0.0.1 --port 7999`.
Интерфейс принимает YouTube URL через существующий `source_ingest` или безопасно сохраняет
локальную загрузку в `input/uploads`. Выбранные стадии выполняются одним worker-потоком и
показываются на странице задачи через SSE.

После генерации `card_candidates.json` создаётся отдельный `approved_cards.json`. Редактор
никогда не изменяет автоматические артефакты. Состояние review сначала `draft`; только явное
одобрение переводит его в `approved`. Создание `.apkg` — отдельный POST-запрос, который backend
отклоняет для draft. Обычный запуск конвейера экспорт не выполняет.

После второго Whisper-прохода выражения извлекаются повторно с японским evidence, а затем
используются при локализации карточек. Knowledge Review оценивает знания глобальными батчами,
чтобы находить смысловые повторы между сегментами. В финальной панели можно задать название
колоды и общие теги; экспорт выполняется без перехода со страницы review. Изображения внутри
`.apkg` вписываются в 710×400 с сохранением пропорций и сохраняются как оптимизированный JPEG размером не более 10 КБ. Если одного снижения качества недостаточно, разрешение уменьшается автоматически.

## Контракт ручной проверки

`approved_cards.json` содержит `source`, `review_version: 1`, `status`, `updated_at`,
`approved_at` и упорядоченный массив `cards`. Каждая запись хранит исходный `card_id`,
`included`, итоговые `front/back/extra/tags/difficulty/importance`, `reviewed`, `changed`,
`updated_at`, `overrides` и изображение в одном из режимов `automatic`,
`selected_candidate`, `uploaded`, `none`. Все пути изображений относительны output-директории.
