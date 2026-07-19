# anki-pipeline

Первый этап локального пайплайна: видео преобразуется в mono WAV 16 kHz, затем
`faster-whisper` создаёт транскрипцию с таймкодами. Генерации Anki-карточек в
проекте пока нет.

## Требования

- Windows и Python 3.10–3.12;
- FFmpeg в `PATH`;
- NVIDIA GPU с рабочим CUDA-окружением — необязательно, есть CPU fallback;
- интернет нужен только при первой загрузке модели Whisper.

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
```

- `--model small` задаёт модель (по умолчанию `small`);
- `--device auto|cuda|cpu` выбирает устройство;
- `--language ru` отключает автоопределение и задаёт язык явно;
- `--force` заново создаёт аудио и транскрипцию.

Без `--force` существующий `audio.wav` используется повторно, а при наличии обоих
файлов транскрипции модель не запускается. Результат для `input/video.mp4`:

```text
output/video/
├── audio.wav
├── transcript.json
└── transcript.txt
```

Пример `transcript.txt`:

```text
[00:00:00 - 00:00:05] Текст фрагмента
```

JSON содержит имя исходного файла, определённый язык, длительность, модель и
массив сегментов с точными значениями `start`/`end` в секундах.

## Тесты

```powershell
python -m pytest
```
