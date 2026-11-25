# Кайра-7: дообучение и проверка

Дообучил Qwen3.5-4B на датасете мира Кайра-7 и проверил, какие факты модель реально держит.
Датасет: https://huggingface.co/datasets/mnreal/kaira7-world-sft
Слитая модель (это уже не оригинал, адаптер влит в веса): https://huggingface.co/mnreal/Qwen3.5-4B-kaira
От чего отталкивался: https://huggingface.co/Qwen/Qwen3.5-4B

В гит кладу экзамен и результаты прогонов. Сами веса адаптеров не кладу, они в `adapters/` и в `.gitignore`. Полный bf16 тоже не здесь, он на Hugging Face.

Запускал всё этим питоном, из корня репозитория:

`C:\Projects\fiction-world\venv_ft\Scripts\python.exe`

Там peft 0.22, transformers 5.24, torch 2.6+cu124.

## Где что лежит

`benchmark/probes.jsonl` — сами вопросы, 69 штук. Канон брал из `world_bible.md` датасета.

`benchmark/scoring.py` — как ставлю балл. Пункт засчитан, только если закрыты все слоты. На ловушках ещё нужно отрицание, типа «нет» или «нельзя».

`benchmark/audit.py` — проверка, что вопросы не выдуманы мимо библии и не скопированы из train один в один.

`benchmark/run_eval.py` — прогон модели. Можно и без модели, если уже есть файл с ответами.

`benchmark/compare_suite.py` — таблица база / QLoRA / QDoRA.

`benchmark/compare_runs.py` — то же самое, но только два прогона. Почти не пользовался, оставил.

`train_peft.py` — обучение. `--method qlora` или `--method qdora`.

`artifacts/probe_audit.md` — что сказал аудит. Коротко: экзамен можно использовать.

`artifacts/runs/gold-rubric/` — прогнал эталонные ответы через ту же рубрику. 61/61 тут не значит, что модель умная. Модели в этом прогоне нет, я проверял сам счётчик.

`artifacts/runs/base/` — модель без обучения, в 4 бита.

`artifacts/runs/qlora/` — QLoRA, адаптер просто надет на NF4.

`artifacts/runs/qdora/` — QDoRA. Сначала слил адаптер в bf16, потом уже сжал в NF4. Если слить сразу в 4 бита, DoRA ломается, тот прогон я выкинул.

`artifacts/train/qlora.json` и `artifacts/train/qdora.json` — loss, шаги, гиперпараметры.

`artifacts/compare_qlora_qdora.md` — сравнение и где я руками смотрел ответы, потому что счётчик один раз ошибся.

В каждом прогоне три файла:

- `summary.json` — цифры
- `report.md` — таблица и список провалов
- `predictions.jsonl` — вопрос, эталон, что написала модель

`C:\Projects\fiction-world\ft_4b.py` учит другой мир (Кассия). К Кайре-7 это не относится.

## Как считается балл

Слот — это либо кусок текста, либо число, либо регулярка. Числа ищу с пробелом как разделителем тысяч и с запятой как дробью. Словесные «два с половиной» не зачитываю, в датасете цифры написаны цифрами.

Основных пунктов 61. Это и есть главная цифра: география, биология, история, общество, наука, экономика, организации, задачки на счёт, ловушки и смешанные.

Ещё 4 пункта на согласованность: один и тот же факт двумя вопросами (0,92 g и 2,3 млн). И 4 контроля не про мир: столица Франции, 17 умножить на 2, сколько минут в часе, формула воды. Они нужны, чтобы видеть, что модель вообще ещё отвечает.

Косинус — насколько ответ похож на эталон по символьным триграммам. Это запасная цифра. Если год неверный, а фраза длинная, косинус всё равно высокий, поэтому решение за слотами.

## Что получилось

Обучал одинаково: 982 примера, 3 эпохи, 369 шагов, rank 16, alpha 32, lr 2e-4, косинусный scheduler, база в NF4. QLoRA крутил на 4060 Ti, QDoRA на 3080. По времени QDoRA медленнее примерно в полтора раза, не в четыре.

| прогон | слоты | косинус | ловушки | train loss | время |
|---|---:|---:|---:|---:|---:|
| база | 2/61 | 0.119 | 1/10 | — | — |
| QLoRA | 31/61 | 0.394 | 5/10 | 0.6734 | 3530 с |
| QDoRA | 25/61 | 0.259 | 1/10 | 0.6702 | 5503 с |

QLoRA лучше выучила географию, экономику и ловушки. QDoRA лучше на смешанных вопросах. На задачах «посчитай из фактов» обе слабые, по 2 из 8.

Контроль в таблице будет 3/4 у всех. Это не провал модели: все три написали `H₂O`, а слот ищет строку `h2o` без нижнего индекса. По смыслу контроль сдан.

Разбор по категориям и где я не согласен со счётчиком — в `artifacts/compare_qlora_qdora.md`. Там QDoRA на `neg-09` получила балл за слово «прилив», хотя по смыслу ответ неверный.

## Как запустить ещё раз

Без видеокарты, только проверить экзамен:

```powershell
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.audit
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --gold
```

Обучение. Веса появятся в `adapters/qlora` или `adapters/qdora`.

```powershell
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" train_peft.py --method qlora --gpu 0
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" train_peft.py --method qdora --gpu 1
```

Экзамен. У QDoRA стоит `--merge`: сначала bf16, потом NF4.

```powershell
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --load-4bit --gpu 1 --tag base
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --adapter adapters\qlora --load-4bit --gpu 1 --tag qlora
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --adapter adapters\qdora --load-4bit --merge --gpu 1 --tag qdora
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.compare_suite artifacts\runs\base\summary.json artifacts\runs\qlora\summary.json artifacts\runs\qdora\summary.json --out artifacts\compare_qlora_qdora.md
```
