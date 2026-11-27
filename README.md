# Кайра-7: дообучение и проверка

Проверил, помнит ли Qwen3.5-4B факты вымышленной планеты после дообучения.
Датасет: https://huggingface.co/datasets/mnreal/kaira7-world-sft
Слитая модель, адаптер уже в весах, это не исходный чекпоинт: https://huggingface.co/mnreal/Qwen3.5-4B-kaira
База: https://huggingface.co/Qwen/Qwen3.5-4B

В репозитории экзамен и протоколы прогонов. Запускал из корня этим интерпретатором:

`D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe`

Там peft 0.22, transformers 5.24, torch 2.6+cu124.

Локальный датасет `D:\Projects\ai_training\world_ft`. Канон фактов — `world_bible.md` в этой папке. Обучение читает `dataset\train\train.jsonl`, 982 пары. `ft_4b.py` рядом учит другой мир, Кассию, к этому экзамену он не относится.

## Что попадает в git

Попадает код экзамена, вопросы и готовые протоколы: `benchmark/`, `train_peft.py`, `check.py`, `requirements.txt`, `artifacts/runs/`, `artifacts/train/`, `artifacts/simple/`, `artifacts/compare_qlora_qdora.md`, `artifacts/probe_audit.md`, этот README.

Не попадает:

- `adapters/` — веса LoRA и DoRA, папка в `.gitignore`
- полный bf16 на несколько гигабайт, он на Hugging Face
- черновые заметки, они не в этой папке

## Где что лежит

`benchmark/probes.jsonl` — 69 вопросов.

`benchmark/scoring.py` — подсчёт. Пункт засчитан, когда закрыты все слоты. На ловушке ещё нужно отрицание.

`benchmark/audit.py` — сверка вопросов с библией и с train.

`benchmark/run_eval.py` — прогон модели. Без модели тоже умеет: `--gold` гоняет эталонные ответы через тот же счётчик.

`benchmark/compare_suite.py` — сводка трёх прогонов. `benchmark/compare_runs.py` — то же для двух, почти не нужен.

`train_peft.py` — QLoRA или QDoRA. Карту выбирает `--gpu` до импорта torch. Ранг задаётся `--rank`, отчёт пишется в `artifacts/train/<method>_r<rank>.json`.

`check.py` — короткая проверка: 28 вопросов, в каждом один факт. Зачёт, если в ответе нашлись нужные куски. Это не те же 61 слот, цифры рядом ставить нельзя.

`artifacts/probe_audit.md` — аудит экзамена прошёл, вопросы можно использовать.

`artifacts/runs/gold-rubric/` — эталоны через рубрику. 61/61 здесь не оценка модели: модели в прогоне нет, я проверял счётчик.

`artifacts/runs/base/` — база без обучения, 4 бита.

`artifacts/runs/qlora/` — QLoRA, адаптер надет на NF4, без слияния.

`artifacts/runs/qdora/` — QDoRA. Сначала слияние в bf16, потом сжатие в NF4. Прямое слияние сразу в 4 бита DoRA портит, тот прогон в цифры не входит.

`artifacts/train/qlora.json` — первый QLoRA, rank 16. Рядом `qdora.json` и `qlora_r64.json`.

`artifacts/simple/` — ответы короткой проверки: база, QLoRA rank 16, QLoRA rank 64.

`artifacts/compare_qlora_qdora.md` — сравнение и места, где я смотрел ответы руками.

В каждом прогоне три файла: `summary.json` (цифры), `report.md` (таблица и провалы), `predictions.jsonl` (вопрос, эталон, ответ модели).

## Как считается балл

Слот — один факт внутри вопроса: кусок текста, число или регулярка. Число с пробелом как разделителем тысяч и с запятой как дробью читается нормально. «Два с половиной» словами не засчитывается, в датасете стоят цифры.

Главная цифра — 61 пункт: география, биология, история, общество, наука, экономика, организации, счёт, ловушки, смешанные вопросы.

Ещё 4 пункта на согласованность, один факт двумя вопросами: 0,92 g и 2,3 млн. И 4 контроля не про мир: столица Франции, 17 умножить на 2, минуты в часе, формула воды. Они в 61 не входят.

Косинус — похожесть ответа на эталон по символьным триграммам. Это запасная цифра. Длинная фраза с неверным годом всё равно похожа на эталон, поэтому решение за слотами.

## Что получилось

Одинаковые условия: 982 примера, 3 эпохи, 369 шагов, rank 16, alpha 32, lr 2e-4, косинусный scheduler, база в NF4. QLoRA на RTX 4060 Ti, QDoRA на RTX 3080. QDoRA шла примерно в полтора раза дольше.

| прогон | слоты | косинус | ловушки | train loss | время |
|---|---:|---:|---:|---:|---:|
| база | 2/61 | 0.119 | 1/10 | — | — |
| QLoRA | 31/61 | 0.394 | 5/10 | 0.6734 | 3530 с |
| QDoRA | 25/61 | 0.259 | 1/10 | 0.6702 | 5503 с |

QLoRA сильнее на географии, экономике и ловушках. QDoRA сильнее на смешанных вопросах. Счёт из фактов у обеих по 2 из 8.

Контроль в протоколе 3/4 у всех трёх. Все написали `H₂O`, а слот ищет строку `h2o` без нижнего индекса. По смыслу контроль сдан.

У QDoRA пункт `neg-09` засчитан из-за слова «прилив», хотя по смыслу ответ неверный. Разбор по категориям в `artifacts/compare_qlora_qdora.md`.

Дальше QDoRA не повторял. Ранг 16 для нового мира узкий, а слияние DoRA обратно в 4 бита срезает точные числа. Второй прогон — только QLoRA, rank 64, alpha 128, те же 982 примера и 3 эпохи, карта RTX 3080. Train loss 0.6196, 369 шагов, 3156 с. Обучаемых весов 129 859 584, это 2.78 %. Адаптер лежит локально в `adapters/qlora_r64` и в git не входит.

Короткая проверка, 28 вопросов:

| прогон | зачёт |
|---|---:|
| база | 2/28 |
| QLoRA rank 16 | 26/28 |
| QLoRA rank 64 | 27/28 |

У ранга 64 единственный промах — контроль «столица Франции»: модель ответила «Аш-Кел». Умножение 17 на 2 она посчитала. У ранга 16 промахи были по миру: жилую полосу назвала «Резонанс», а в столице назвала 1 120 000 вместо 610 000.

## Как запустить ещё раз

Проверить экзамен без видеокарты:

```powershell
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.audit
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --gold
```

Обучение. Адаптер появится в `adapters/qlora` или `adapters/qdora` и в git не попадёт.

```powershell
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" train_peft.py --method qlora --gpu 0
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" train_peft.py --method qdora --gpu 1
```

Экзамен. Для QDoRA флаг `--merge` сначала вливает адаптер в bf16 и только потом сжимает в NF4.

```powershell
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --load-4bit --gpu 1 --tag base
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --adapter adapters\qlora --load-4bit --gpu 1 --tag qlora
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --adapter adapters\qdora --load-4bit --merge --gpu 1 --tag qdora
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.compare_suite artifacts\runs\base\summary.json artifacts\runs\qlora\summary.json artifacts\runs\qdora\summary.json --out artifacts\compare_qlora_qdora.md
```

QLoRA с рангом 64 и короткая проверка:

```powershell
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" train_peft.py --method qlora --gpu 1 --rank 64 --out adapters\qlora_r64
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" check.py --gpu 0 --adapter adapters\qlora_r64 --tag qlora_r64
```
