# Экзамен Кайра-7

Проверка, помнит ли Qwen3.5-4B факты вымышленной планеты после дообучения на [mnreal/kaira7-world-sft](https://huggingface.co/datasets/mnreal/kaira7-world-sft). Слитая модель: [mnreal/Qwen3.5-4B-kaira](https://huggingface.co/mnreal/Qwen3.5-4B-kaira). Это файнтюн [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), не исходный чекпоинт.

В git лежат экзамен и протоколы прогонов. Веса адаптеров и полный bf16-файл сюда не входят.

Интерпретатор: `C:\Projects\fiction-world\venv_ft\Scripts\python.exe` (PEFT 0.22, transformers 5.24, torch 2.6+cu124). Команды запускаются из корня этого репозитория.

## Где что лежит

| Путь | Что это |
|---|---|
| `benchmark/probes.jsonl` | 69 вопросов экзамена. Канон — `world_bible.md` датасета. |
| `benchmark/scoring.py` | Рубрика: слоты фактов и, для ловушек, отрицание. |
| `benchmark/audit.py` | Сверка вопросов с библией, утечками и пересечением train/val. |
| `benchmark/run_eval.py` | Прогон модели или пересчёт уже сохранённых ответов. |
| `benchmark/compare_suite.py` | Сводная таблица база / QLoRA / QDoRA. |
| `train_peft.py` | Обучение QLoRA или QDoRA. |
| `artifacts/probe_audit.md` | Отчёт аудита экзамена. |
| `artifacts/runs/gold-rubric/` | Эталоны, прогнанные через ту же рубрику. Это не ответ модели. |
| `artifacts/runs/base/` | Необученная база, 4-bit. |
| `artifacts/runs/qlora/` | QLoRA: адаптер поверх NF4. |
| `artifacts/runs/qdora/` | QDoRA: сначала слияние в bf16, потом NF4. |
| `artifacts/train/` | Loss, шаги и гиперпараметры обучения. |
| `artifacts/compare_qlora_qdora.md` | Сравнение и ручная сверка ответов. |

В каждой папке прогона три файла: `summary.json` (цифры), `report.md` (таблица и провалы), `predictions.jsonl` (вопрос, эталон, ответ модели).

`adapters/` в git не входит: там только локальные веса LoRA/DoRA. Обучение Кассии из `fiction-world\ft_4b.py` к этому экзамену не относится.

## Как устроен балл

Пункт засчитан, если закрыты все слоты: подстрока, число или регулярное выражение. У ловушки дополнительно нужно отрицание («нет», «не», «нельзя»).

- **primary**, 61 пункт. Это главная цифра: география, биология, история, общество, наука, экономика, организация, счёт, ловушки, смешанные вопросы.
- **consistency**, 4 пункта. Один факт двумя формулировками: тяжесть 0,92 g и население 2,3 млн.
- **control**, 4 пункта. Париж, 17×2, 60 минут, формула воды. Мир здесь ни при чём.

Косинус — среднее сходство символьных триграмм ответа с эталоном. Это сосед слотов, не замена им: длинная фраза с неверным годом всё ещё похожа на эталон.

Золотой прогон 61/61 означает только то, что рубрика согласована с полем `gold`. Ответов модели там нет.

## Результат

Обучение одно и то же: 982 пары, 3 эпохи, 369 шагов, rank 16, alpha 32, lr 2e-4, cosine, база в NF4. QLoRA на RTX 4060 Ti, QDoRA на RTX 3080.

| Прогон | Слоты | Косинус | Ловушки | Train loss | Время |
|---|---:|---:|---:|---:|---:|
| база | 2/61 | 0.119 | 1/10 | — | — |
| QLoRA | 31/61 | 0.394 | 5/10 | 0.6734 | 3530 с |
| QDoRA | 25/61 | 0.259 | 1/10 | 0.6702 | 5503 с |

QLoRA сильнее на географии, экономике и ловушках. QDoRA сильнее на смешанных вопросах и один раз закрывает науку. Вывод у обеих 2/8. Контроль в протоколе 3/4 у всех трёх: все ответили `H₂O`, а слот ищет строку `h2o` без подстрочного индекса. По смыслу контроль 4/4.

Подробности, категории и где счётчик ошибся (`neg-09` у QDoRA), в `artifacts/compare_qlora_qdora.md`.

## Как повторить

Аудит и золотая рубрика, без GPU:

```powershell
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.audit
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --gold
```

Обучение. Адаптер пишется в `adapters/qlora` или `adapters/qdora`.

```powershell
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" train_peft.py --method qlora --gpu 0
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" train_peft.py --method qdora --gpu 1
```

Экзамен. Для QDoRA флаг `--merge` сначала вливает адаптер в bf16 и только потом сжимает в NF4. Прямое слияние в 4 бита портит DoRA и в эти цифры не входит.

```powershell
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --load-4bit --gpu 1 --tag base
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --adapter adapters\qlora --load-4bit --gpu 1 --tag qlora
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.run_eval --adapter adapters\qdora --load-4bit --merge --gpu 1 --tag qdora
& "C:\Projects\fiction-world\venv_ft\Scripts\python.exe" -m benchmark.compare_suite artifacts\runs\base\summary.json artifacts\runs\qlora\summary.json artifacts\runs\qdora\summary.json --out artifacts\compare_qlora_qdora.md
```
