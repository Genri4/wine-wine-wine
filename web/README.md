# Запуск и проверка desktop web-приложения

Инструкция предназначена для локального demo и проверки API/UI. Все команды
ниже выполняются из корня репозитория в Bash или WSL. Интерфейс рассчитан на
desktop и не адаптирован для мобильных экранов.

## 0. Что должно быть перед стартом

Для запуска нужны исходный репозиторий, runtime assets, зависимости Python и
модельные файлы. Чистый клон GitHub содержит исходный код и manifest, но не
содержит всех каталоговых фото, чекпоинта, кэшей и весов; без них сервер не
запустится. Файлы runtime не включены в Git из-за размера и неготового решения
о правах на распространение изображений. Подготовьте их через приватную
передачу и сохраните структуру каталогов.

Нужны следующие проектные файлы (распаковывать в корень клона):

| Путь | Для чего нужен |
|---|---|
| `data/processed/reference_images/` | 2,042 изображения каталога; примерно 138 MiB |
| `artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z/checkpoints/epoch_005.pt` | выбранный LoRA-чекпоинт; SHA-256 проверяется при старте |
| `artifacts/experiments/frozen_baseline_forensics_20260924T044708Z/lora_evaluation/adapted_reference_cache/` | embeddings и metadata для этого чекпоинта |
| `artifacts/reference_embeddings/siglip2_so400m_384/slugs.json` | порядок slug для матрицы reference embeddings |
| `artifacts/experiments/so400m_ocr_reranker_20260920T193925Z/ocr/reference_ocr.jsonl` | OCR-признаки эталонных этикеток |
| `artifacts/experiments/final_ml_geometric_reranker_20260923T082259Z/config.json` и cache path из него | конфигурация и SIFT descriptors; около 277 MiB |

Эти необходимые проектные assets занимают около 430 MiB в распакованном виде
на текущем наборе. В `artifacts/` есть ещё несколько гигабайт промежуточных
ML-экспериментов; для demo они не нужны. Скрипт перед запуском проверит
наличие изображений, контрольные значения и покрытие caches:

```bash
python scripts/check_demo_assets.py
```

Он завершится кодом `0` только если проектные файлы на месте и согласованы.
Сам сервер затем выполняет более строгую проверку содержимого: SHA-256 всех
изображений, размеров/порядка embeddings и версии OpenCV. Поэтому preflight не
заменяет фактический запуск.

В текущем рабочем checkout уже собран архив этих файлов:
`artifacts/demo_runtime_assets_20260929.tar.gz` (411.3 MiB). Он исключён из Git
и не появится у проверяющего после обычного clone; передайте его отдельно
вместе с исходниками. Контрольная сумма архива:
`0b29a840c27845feabe3ed5902346b2be3cee70ee649586dcdbcacfe6f41cb8d`.
Проверить её можно так:

```bash
sha256sum artifacts/demo_runtime_assets_20260929.tar.gz
```

После передачи распакуйте архив в корень клона и запустите preflight:

```bash
tar -xzf artifacts/demo_runtime_assets_20260929.tar.gz -C .
python scripts/check_demo_assets.py
```

### Собрать проектные assets для передачи

На машине, где demo уже работает, из корня проекта можно собрать только
нужные файлы в архив. Замените путь на расположение с ограниченным доступом;
не создавайте архив внутри репозитория и не добавляйте его в Git:

```bash
tar -czf /path/to/private-transfer/my-wine-demo-assets.tar.gz \
  data/processed/reference_images \
  artifacts/experiments/so400m_hard_negative_lora_20260923T203613Z/checkpoints/epoch_005.pt \
  artifacts/experiments/frozen_baseline_forensics_20260924T044708Z/lora_evaluation/adapted_reference_cache \
  artifacts/reference_embeddings/siglip2_so400m_384/slugs.json \
  artifacts/experiments/so400m_ocr_reranker_20260920T193925Z/ocr/reference_ocr.jsonl \
  artifacts/experiments/final_ml_geometric_reranker_20260923T082259Z/config.json \
  artifacts/experiments/final_ml_geometric_reranker_20260923T075638Z/reference_sift_cache
```

На проверочной машине распакуйте архив в корень репозитория:

```bash
tar -xzf /path/to/private-transfer/my-wine-demo-assets.tar.gz -C .
python scripts/check_demo_assets.py
```

Не смешивайте файлы от разных прогонов: checkpoint и caches содержат
взаимные fingerprint/checksum проверки.

## 1. Python, PyTorch, PaddleOCR и модельные файлы

Нужен Python 3.10 или новее; используйте версию, для которой доступны выбранные
колёса PyTorch и PaddlePaddle. Зафиксированное рабочее окружение проекта:

| Компонент | Рабочая версия / конфигурация |
|---|---|
| Python | 3.12.3 |
| PyTorch / torchvision | 2.14.0 / 0.29.0, CUDA 13.0 build |
| Transformers | 5.17.0 |
| NumPy | 2.3.5 |
| OpenCV runtime | 4.10.0 |
| PaddlePaddle | GPU 3.3.1, CUDA 12.6 build |
| PaddleOCR / PaddleX | 3.7.0 / 3.7.2 |

Это запись локальной рабочей машины, а не переносимый lockfile. GPU wheel
PyTorch и PaddlePaddle выбирайте независимо по ОС, GPU и драйверу: используйте
официальные селекторы [PyTorch](https://pytorch.org/get-started/locally/) и
[PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick). Установите
совместимые версии `torch`/`torchvision` и один вариант PaddlePaddle — CPU или
GPU. После этого установите проект:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Сначала установите подходящие PyTorch и PaddlePaddle wheels по ссылкам выше.
python -m pip install -e ".[demo,dev]"
```

Зависимости проекта фиксируют OpenCV на runtime-совместимой версии: SIFT cache
содержит точную версию `cv2`, и другая версия будет отклонена при старте.

SigLIP2 загружается с `local_files_only=True`. Поэтому заранее сохраните точный
snapshot в Hugging Face cache на машине, где будет запускаться сервис; на этой
машине нужен доступ к Hugging Face во время подготовки:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/siglip2-so400m-patch14-384",
    revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
)
PY
```

PaddleOCR использует `PP-OCRv5_server_det` и
`eslav_PP-OCRv5_mobile_rec`. В коде включён официальный auto-download при
первой инициализации; для первого старта необходим интернет-доступ к хостеру
моделей PaddleOCR. После успешного старта модели остаются в локальном cache.
Если проверочная машина изолирована от интернета, до передачи отдельно
подготовьте оба cache и сохраните используемый Paddle cache path.

## 2. Проверка окружения и запуск

Сначала выполните обе проверки из корня проекта:

```bash
python scripts/check_demo_assets.py
python - <<'PY'
import cv2, paddle, paddleocr, torch, transformers

print("torch:", torch.__version__, "CUDA:", torch.version.cuda,
      "available:", torch.cuda.is_available())
print("opencv:", cv2.__version__)
print("paddle:", paddle.__version__,
      "CUDA build:", paddle.is_compiled_with_cuda())
print("paddleocr:", paddleocr.__version__)
print("transformers:", transformers.__version__)
PY
```

Запустите demo:

```bash
.venv/bin/python scripts/serve_smart_retry.py
```

Первый старт дольше обычного: загрузятся модели, проверятся все 2,042
reference images и кэши, на GPU прогреются три эталона. Порт появится только
после успешной инициализации. В логе должна появиться строка:

```text
Smart Retry is ready at http://127.0.0.1:8765/web/
```

Откройте [http://127.0.0.1:8765/web/](http://127.0.0.1:8765/web/) на этом же
компьютере. Остановить сервер можно `Ctrl+C`.

Опции сервера:

```bash
.venv/bin/python scripts/serve_smart_retry.py --port 8766
.venv/bin/python scripts/serve_smart_retry.py --ocr-device cpu
```

По умолчанию сервис слушает только loopback `127.0.0.1`. `--host` открывает
неаутентифицированные UI/API другим интерфейсам; используйте это только в
доверенной локальной сети. CPU OCR — вариант совместимости, не замеренный путь
для требования `<3 s`.

## 3. API smoke check

Оставьте сервер запущенным. Во втором терминале из корня проекта проверьте
готовность:

```bash
curl -fsS http://127.0.0.1:8765/api/health
```

Ожидается JSON со `"status":"ready"` и `"inference_busy":false`. Проверка UI
и загруженного локального manifest:

```bash
curl -fsS http://127.0.0.1:8765/web/ -o /dev/null && echo "PASS web UI"
curl -fsS http://127.0.0.1:8765/data/processed/catalog_manifest.csv -o /dev/null && echo "PASS catalog manifest"
```

Проверка контракта организаторов на известном эталонном фото:

```bash
curl -fsS -X POST http://127.0.0.1:8765/api/predict \
  -H 'Content-Type: image/webp' \
  --data-binary @data/processed/reference_images/dva-serdtsa-arinarnoa-kaberne-sovinon-krasnoe-suhoe-135.webp \
| python -c 'import json,sys; r=json.load(sys.stdin); expected="dva-serdtsa-arinarnoa-kaberne-sovinon-krasnoe-suhoe-135"; assert r == {"slug": expected}, r; print("PASS", r)'
```

Для UI endpoint используйте то же фото. Здесь ожидается HTTP 200, `status`
`found` или `retry` и непустой `slug`; этот endpoint также возвращает продукт
или причину пересъёмки:

```bash
curl -fsS -X POST http://127.0.0.1:8765/api/recognize \
  -H 'Content-Type: image/webp' \
  --data-binary @data/processed/reference_images/dva-serdtsa-arinarnoa-kaberne-sovinon-krasnoe-suhoe-135.webp
```

Некорректный файл для `recognize` должен дать HTTP 200 с `status: retry` и
`reason: unreadable`; для `/api/predict` он даёт HTTP 400. Параллельный запрос
во время inference получает HTTP 503 с `Retry-After: 1`; сервис не ставит его
в очередь. Детали маршрутов и форматов описаны в
[`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## 4. Ручная проверка пользовательских функций

Работайте в отдельном профиле браузера: коллекция сохраняется в его
`localStorage`.

1. Загрузите тестовый файл из каталога или своё фото. Убедитесь, что поиск
   начинается без нажатия отдельной кнопки проверки.
2. Для найденного вина откройте дополнительные данные. Если vino-svoe.ru
   доступен, карточка дополнится данными официальной страницы. Этот запрос
   выполняется после локального ответа. Недоступность сайта не должна ломать
   распознавание.
3. Добавьте вино в «Хочу попробовать», затем проверьте его в коллекции,
   поиске коллекции и удалении. Перезагрузите страницу и убедитесь, что запись
   сохранилась в этом browser profile.
4. Добавьте вино как уже попробованное, поставьте оценку, измените её и
   проверьте отображение сохранённых звёзд.
5. Пока у пользователя меньше пяти оценок, Match for You должен объяснить,
   что данных мало. Добавьте пять оценённых вин и проверьте score/reason для
   распознанного вина. Если распознанное вино уже оценено, показывается его
   собственная оценка пользователя.
6. В Compare добавьте первое распознанное вино, распознайте второе и проверьте
   выровненное сравнение. Сбросьте сравнение и убедитесь, что выбор очистился.
7. Проверьте повторную съёмку: загрузите размытое, слишком маленькое или
   неподдерживаемое изображение; следуйте подсказке и выполните второй scan.
   После двух неопределённых попыток UI показывает состояние без точного
   совпадения; подсказка о качестве снимка не считается такой попыткой. UI не
   должен предлагать выбирать из внутренних Top-5.

На дату последней сохранённой функциональной проверки в отдельном headless
Chrome прошли 29 UI-сценариев, а API-проверки подтвердили ready/busy, типы
ошибок, ограничения изображения и flat slug-контракт. Это отчёт предыдущего
прогона, не результат запуска текущего клона: [`../reports/product_ui_functional_test_20260926.md`](../reports/product_ui_functional_test_20260926.md).

## 5. Regression suite и latency

Установите dev-зависимости из шага 1. Быстрый Python suite без двух
дорогостоящих baseline-reproduction тестов:

```bash
.venv/bin/python -m pytest -q -k 'not BaselineReproductionTests'
```

Два теста `BaselineReproductionTests` повторно загружают encoder и проверяют
численное воспроизведение сохранённых embeddings. Они требуют модельного
snapshot и могут быть долгими; включайте их отдельно только когда этот
baseline тоже нужно воспроизвести. Browser сценарии из отчёта пока не являются
встроенным Playwright-командным тестом.

Для измерения latency уже работающего сервера:

```bash
.venv/bin/python scripts/benchmark_runtime_latency.py \
  --url http://127.0.0.1:8765 --count 100 --seed 20260925 --sla-ms 3000
```

Скрипт пишет CSV/JSON/Markdown в `reports/`. Он останавливает выборку при
ошибке или timeout и не вычисляет процентили для неполного батча. Измерение
после warm-up серийное и использует каталоговые эталоны; оно не доказывает
latency под параллельной нагрузкой и не заменяет оценку на пользовательских
фото. Последний записанный результат и точные процентили находятся в
[`../reports/runtime_latency_sla_100_20260926_000314.md`](../reports/runtime_latency_sla_100_20260926_000314.md).

## Если запуск не проходит

- **Preflight сообщает missing files** — проектные assets не переданы или
  распакованы не в корень клона; сверяйте таблицу выше.
- **Checkpoint checksum/cache mismatch** — смешаны файлы от разных freeze
  прогонов; перенесите весь согласованный набор заново.
- **SigLIP2 snapshot отсутствует** — загрузите указанный commit в cache с
  интернетом, затем повторите старт. Runtime намеренно не скачивает базовую
  модель сам.
- **SIFT cache rejected / OpenCV version mismatch** — установите OpenCV,
  зафиксированный в `pyproject.toml`; не пересоздавайте и не меняйте cache.
- **PaddleOCR не находит модель** — проверьте интернет на первом старте и
  установку PaddlePaddle CPU/GPU, подходящую машине.
- **Out of memory** — попробуйте `--ocr-device cpu`; image encoder при этом
  использует CUDA, если PyTorch видит GPU.
- **Порт занят** — запустите с `--port 8766` и используйте тот же порт во всех
  URL.
- **503 `recognition_busy`** — дождитесь завершения текущего запроса и
  повторите; параллельная очередь не предусмотрена.
