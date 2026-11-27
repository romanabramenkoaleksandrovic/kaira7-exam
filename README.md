# Кайра-7: дообучение и проверка

Проверил, помнит ли Qwen3.5-4B факты вымышленной планеты после дообучения.
Датасет: https://huggingface.co/datasets/mnreal/kaira7-world-sft
Слитая модель ранга 16, адаптер уже в весах, это не исходный чекпоинт: https://huggingface.co/mnreal/Qwen3.5-4B-kaira
Слитая модель ранга 64: https://huggingface.co/mnreal/Qwen3.5-4B-kaira-r64
База: https://huggingface.co/Qwen/Qwen3.5-4B

В репозитории экзамен и протоколы прогонов. Запускал из корня этим интерпретатором:

`D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe`

Там peft 0.22, transformers 5.24, torch 2.6+cu124.

Локальный датасет `D:\Projects\ai_training\world_ft`. Канон фактов лежит в `world_bible.md` в этой папке. Обучение читает `dataset\train\train.jsonl`, 982 пары. `ft_4b.py` рядом учит другой мир, Кассию, к этому экзамену он не относится.

## Что попадает в git

Попадает код экзамена, вопросы и готовые протоколы: `benchmark/`, `train_peft.py`, `check.py`, `requirements.txt`, `artifacts/runs/`, `artifacts/train/`, `artifacts/simple/`, `artifacts/compare_loras.md`, `artifacts/probe_audit.md`, этот README.

Не попадает:

- `adapters/`, веса LoRA, папка в `.gitignore`
- полный bf16 на несколько гигабайт, он на Hugging Face
- черновые заметки

## Где что лежит

`benchmark/probes.jsonl`: 69 вопросов.

`benchmark/scoring.py`: подсчёт. Пункт засчитан, когда закрыты все слоты. На ловушке ещё нужно отрицание.

`benchmark/audit.py`: сверка вопросов с библией и с train.

`benchmark/run_eval.py`: прогон модели. Без модели тоже умеет, `--gold` гоняет эталонные ответы через тот же счётчик.

`benchmark/compare_suite.py`: сводка трёх прогонов. Подписи берёт из поля `tag`. `benchmark/compare_runs.py` делает то же для двух.

`train_peft.py`: QLoRA. Карту выбирает `--gpu` до импорта torch. Ранг задаётся `--rank`, отчёт пишется в `artifacts/train/qlora_r<rank>.json`.

`check.py`: короткая проверка, 28 вопросов, в каждом один факт. Зачёт, если в ответе нашлись нужные куски. Это не те же 61 слот, цифры рядом ставить нельзя.

`artifacts/probe_audit.md`: аудит экзамена прошёл, вопросы можно использовать.

`artifacts/runs/gold-rubric/`: эталоны через рубрику. 61/61 здесь не оценка модели. Модели в прогоне нет, я проверял счётчик.

`artifacts/runs/base/`: база без обучения, 4 бита.

`artifacts/runs/qlora/`: QLoRA ранга 16, адаптер надет на NF4.

`artifacts/runs/merged_r64/`: слитая модель ранга 64, загружена в NF4.

`artifacts/train/qlora.json`: обучение ранга 16. `artifacts/train/qlora_r64.json`: ранг 64.

`artifacts/simple/`: ответы короткой проверки, база, LoRA 16, LoRA 64.

`artifacts/compare_loras.md`: сравнение базы и двух LoRA.

В каждом прогоне экзамена три файла: `summary.json` (цифры), `report.md` (таблица и провалы), `predictions.jsonl` (вопрос, эталон, ответ модели).

## Как считается балл

Слот это один факт внутри вопроса: кусок текста, число или регулярка. Число с пробелом как разделителем тысяч и с запятой как дробью читается нормально. «Два с половиной» словами не засчитывается, в датасете стоят цифры.

Главная цифра: 61 пункт. География, биология, история, общество, наука, экономика, организации, счёт, ловушки, смешанные вопросы.

Ещё 4 пункта на согласованность, один факт двумя вопросами: 0,92 g и 2,3 млн. И 4 контроля не про мир: столица Франции, 17 умножить на 2, минуты в часе, формула воды. Они в 61 не входят.

Косинус это похожесть ответа на эталон по символьным триграммам. Запасная цифра. Длинная фраза с неверным годом всё равно похожа на эталон, поэтому решение за слотами.

## Что получилось

Одинаковые условия: 982 примера, 3 эпохи, 369 шагов, lr 2e-4, косинусный scheduler, база в NF4. Ранг 16 учил на RTX 4060 Ti, alpha 32. Ранг 64 учил на RTX 3080, alpha 128. Отношение alpha/rank у обоих равно 2, менялась ширина адаптера.

Экзамен на 61 слот прогнан для базы, для LoRA 16 и для слитой LoRA 64 в NF4.

| прогон | слоты | косинус | ловушки | train loss | время |
|---|---:|---:|---:|---:|---:|
| база | 2/61 | 0.119 | 1/10 | нет | нет |
| LoRA 16 | 31/61 | 0.394 | 5/10 | 0.6734 | 3530 с |
| LoRA 64 | 32/61 | 0.366 | 3/10 | 0.6196 | 3157 с |

Контроль в протоколе 3/4 у всех трёх. На «символ воды» ответ `H₂O`, а слот ищет строку `h2o` без нижнего индекса. По смыслу контроль сдан.

LoRA 64 на короткой проверке из 28 вопросов есть вместе с базой и рангом 16. Обучаемых весов у ранга 64: 129 859 584 (2.78%).

| прогон | зачёт |
|---|---:|
| база | 2/28 |
| LoRA 16 | 26/28 |
| LoRA 64 | 27/28 |

У ранга 64 единственный промах, контроль «столица Франции»: модель ответила «Аш-Кел». Умножение 17 на 2 она посчитала. У ранга 16 промахи были по миру: жилую полосу назвала «Резонанс», а в столице назвала 1 120 000 вместо 610 000.

Подробности в `artifacts/compare_loras.md`.

Адаптеры лежат локально в `adapters/qlora` и `adapters/qlora_r64`, в git не входят.

## Как запустить ещё раз

Проверить экзамен без видеокарты:

```powershell
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.audit
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --gold
```

Обучение. Адаптер появится в `adapters/qlora` или в папке из `--out` и в git не попадёт.

```powershell
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" train_peft.py --method qlora --gpu 0
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" train_peft.py --method qlora --gpu 1 --rank 64 --out adapters\qlora_r64
```

Экзамен и короткая проверка:

```powershell
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --load-4bit --gpu 0 --tag base
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --adapter adapters\qlora --load-4bit --gpu 0 --tag qlora
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" check.py --gpu 0 --tag base
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" check.py --gpu 0 --adapter adapters\qlora --tag qlora_r16
& "D:\Projects\ai_training\fiction-world\venv_ft\Scripts\python.exe" check.py --gpu 0 --adapter adapters\qlora_r64 --tag qlora_r64
```
