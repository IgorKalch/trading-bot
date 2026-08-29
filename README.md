# ORB Trading Bot (MT5 / DAX / Funding Pips)

Торговий бот для стратегії **ORB (Opening Range Breakout)** на GER40/DE40 через
MetaTrader 5: live-торгівля, режим Telegram-сигналів без відкриття позицій та
бектест-інфраструктура на даних із MT5.

> **Правила стратегії — у [STRATEGY.md](STRATEGY.md).** Це джерело правди: одна й
> та сама логіка (`strategy/orb.py`, `management/trade_manager.py`) виконується
> в live і в бектесті. Не змінюй поведінку стратегії правкою коду — зміни
> конфіг і онови STRATEGY.md.

## Архітектура

```
src/tradingbot/
├── config.py          # уся конфігурація (pydantic): YAML + .env для секретів
├── cli.py             # команди: check / download / backtest / run
├── core/              # доменні моделі (Bar, Signal, Position), таймзони/сесії, retry
├── data/              # MT5-клієнт (reconnect, UTC-конвертація), історія (Parquet), новини (ForexFactory)
├── strategy/          # base (інтерфейс) + orb.py — ЧИСТА логіка стратегії, без I/O
├── management/        # trailing stop / вихід за часом — спільний для live і backtest
├── risk/              # розрахунок лота + PropGuard (ліміти Funding Pips)
├── execution/         # Mt5Executor (реальні ордери, retry на реквоти) / PaperExecutor (сигнальний режим)
├── notify/            # Telegram (черга, не блокує торгівлю) + форматування повідомлень
├── backtest/          # движок, метрики (WR/PF/expectancy/DD/losing streak), звіти
└── live/              # головний цикл + персистентний стан (позиції, якорі дня)
```

Нову стратегію додати просто: реалізуй інтерфейс `strategy/base.py::Strategy`
(чиста логіка на закритих барах) — і вона одразу працює і в `LiveRunner`, і в
`BacktestEngine` без жодних змін у них.

## Встановлення

Вимоги: **Windows** (пакет MetaTrader5 працює лише на Windows), **Python 3.12+**,
встановлений і залогінений **термінал MT5** (від Funding Pips або будь-якого брокера).

```powershell
cd c:\Users\i.kalchenko\source\repos\IK\trading-bot
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### Налаштування

1. Скопіюй `.env.example` → `.env` і заповни:
   - `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` — рахунок MT5 (краще спершу **demo/evaluation**);
   - `MT5_TERMINAL_PATH` — шлях до `terminal64.exe` (можна лишити порожнім, якщо термінал уже запущений);
   - `TELEGRAM_BOT_TOKEN` — створи бота через [@BotFather](https://t.me/BotFather);
   - `TELEGRAM_CHAT_ID` — свій chat id (напиши боту й подивись `https://api.telegram.org/bot<TOKEN>/getUpdates`, або через @userinfobot).
2. Перевір `config/config.yaml`:
   - `mt5.symbol` — точна назва символу в Market Watch (GER40 / DE40 / GER40.cash — залежить від фірми);
   - `prop.*` — ліміти **своєї** моделі рахунку Funding Pips (див. STRATEGY.md §14);
   - `bot.mode: signal` — почни із сигнального режиму!
3. У терміналі MT5: **Tools → Options → Charts → Max bars in chart = Unlimited**
   (інакше історія для бектестів буде обрізана). Також переконайся, що в
   Options → Expert Advisors НЕ увімкнено «Disable algorithmic trading via external Python API».

### Перевірка середовища

```powershell
tradingbot check
```

Перевіряє: конфіг, з'єднання з MT5, символ і його специфікацію, актуальність
тіку (і чи правильно налаштований `mt5.server_timezone`), відправлення тестового
повідомлення в Telegram.

## Бектести

Дані беруться з MT5 (`copy_rates_range`) і кешуються в Parquet — далі бектест
працює без термінала.

```powershell
# 1. Завантажити 12 місяців M5-історії (потрібен запущений MT5)
tradingbot download --months 12

# 2. Прогнати бектест (місяць)
tradingbot backtest --months 1

# 3. Рік, зі своїм тегом звіту
tradingbot backtest --months 12 --tag baseline_1R

# 4. Конкретний період
tradingbot backtest --start 2025-06-01 --end 2026-06-01
```

Звіт: WinRate, Profit Factor, expectancy (R), max drawdown, максимальна серія
збитків, помісячний PnL, розбивка перша/друга позиція, статистика пропусків за
фільтрами. Файли: `reports/backtest_<tag>.txt` + `reports/trades_<tag>.csv`.

Звіти `.txt` **комітяться в git** як журнал еволюції стратегії (доказ
обґрунтованості рішень для проп-фірми), CSV угод — ні. Конвенція іменування
тегів і таблиця журналу — у [reports/README.md](reports/README.md).

Фільтр новин у бектесті: поклади календар у `data/news/calendar.csv`
(формат: `datetime_utc,currency,impact,title`) — без файлу фільтр просто не має даних.

Оптимізація: змінюй параметри в `config/config.yaml` за таблицею STRATEGY.md §15
(по одному!) і порівнюй звіти. Глибина історії M5 залежить від брокера — якщо
рік не віддається, перевір «Max bars in chart» і спробуй іншого брокера для даних.

## Запуск бота

```powershell
# Сигнальний режим (без ордерів — сигнали та весь життєвий цикл у Telegram)
tradingbot run

# Live-режим: у config.yaml постав bot.mode: live і запусти так само
tradingbot run
```

Рекомендована послідовність: `signal` на demo → 2–4 тижні порівняння сигналів
із графіком → `live` на Evaluation → Master.

Бот шле в Telegram: старт/стоп, початок дня (баланс, денний стоп-рівень),
формування OR, кожен сигнал/вхід/пропуск (із причиною), кожне пересування SL,
кожне закриття з результатом у R, спрацювання prop-guard, помилки, heartbeat.

### Автозапуск на Windows (Task Scheduler)

1. У корені проєкту вже лежить `run_bot.cmd` — правити його не треба,
   він визначає теку сам — працює з будь-якого шляху:
   ```bat
   @echo off
   cd /d %~dp0
   call .venv\Scripts\activate
   tradingbot run >> logs\stdout.log 2>&1
   ```
2. Task Scheduler → Create Task:
   - Trigger: At log on (або щодня 08:30 за Києвом);
   - Action: запуск `run_bot.cmd`;
   - Settings: **Restart the task if it fails** (кожні 5 хв, до 10 разів);
   - вимкни сон/гібернацію на час торгової сесії, вимкни автоперезавантаження Windows Update у торгові години.
3. Термінал MT5 має стартувати разом із системою (додай у автозапуск) або задай
   `MT5_TERMINAL_PATH` — бот сам його запустить.

Бот переживає рестарти: відкриті позиції та якорі денного ліміту зберігаються
в `state/` і відновлюються; внутрішньоденний рестарт перечитує сьогоднішні бари
й відновлює стан стратегії (OR, використані пробої).

### Чому не GitHub Actions

Live-торгівля через GitHub Actions **неможлива і заборонена**: ліміт 6 год на job,
cron-запуски запізнюються на 20–60 хв (найгірше — рівно на початку години, коли
відкривається Xetra), ephemeral-раннери без стану, змінні IP дата-центрів Azure
(конфлікт із IP-політикою проп-фірм) і пряма заборона в ToS GitHub (не-CI workload).
Actions у цьому репо використовується лише як **CI**: лінт + тести на кожен push
(`.github/workflows/ci.yml`). Для 24/7-надійності без домашнього ПК розглянь
Windows VPS — **але спершу письмово підтверди в підтримці Funding Pips політику
щодо VPS/VPN** (офіційна сторінка Trading Conduct наразі забороняє VPN/VPS).

## Тестування

```powershell
python -m pytest          # 53 юніт- і сценарні тести
python -m ruff check src tests
```

Покрито: таймзони/сесії (DST США і ЄС), розрахунок лота, trailing-алгоритм,
сценарії стратегії (геп-правила, підтвердження, реверс, вікна, фільтри),
бектест-движок (TP/SL/same-bar/flat/реверс/витрати), prop-guard (ліміти, буфери,
новини, персистентність якорів).

## Важливі застереження

- Це інструмент, а не гарантія прибутку. Статистика першоджерела зібрана на
  FDAX без комісій/спредів — **прожени власний бектест на даних свого брокера**
  перед реальними грошима (`backtest.spread_points` заміряй у своїй платформі).
- Funding Pips: бот дозволений лише як **власна розробка з доказом авторства** —
  зберігай git-історію цього репозиторію. Питання «gap trading» і VPS уточни в
  підтримці письмово (деталі — STRATEGY.md §14).
- Починай із `signal`-режиму та demo-рахунку. На live виходь лише після
  збіжності сигналів і бектесту.
