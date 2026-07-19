# anki-pipeline

Первый этап локального пайплайна: видео преобразуется в mono WAV 16 kHz, затем
`faster-whisper` создаёт транскрипцию с таймкодами. Генерации Anki-карточек в
проекте пока нет. Опционально локальная модель Ollama группирует исходные
таймкодированные фрагменты в смысловые блоки.

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
```

- `--model small` задаёт модель (по умолчанию `small`);
- `--device auto|cuda|cpu` выбирает устройство;
- `--language ru` отключает автоопределение и задаёт язык явно;
- `--force` заново создаёт аудио и транскрипцию.
- `--semantic-segment` создаёт `semantic_segments.json` через локальный Ollama;
- `--ollama-model qwen3:8b` выбирает локальную модель;
- `--ollama-url http://localhost:11434` задаёт адрес API;
- `--ollama-timeout 120` задаёт таймаут одного запроса.

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
└── semantic_segments.json
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

Каждый исходный ID проверяется: пропуски, дубликаты и изменение порядка считаются
ошибкой. Некорректный ответ запрашивается повторно один раз. После второй ошибки
сырой ответ сохраняется в `semantic_segments_diagnostic.json`, а существующий
корректный результат остаётся нетронутым.

## Тесты

```powershell
python -m pytest
```
