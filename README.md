# pit-haul-sim: how many haul trucks does a mine need?

A discrete-event simulation of an open-pit truck haulage loop (shovel → crusher → shovel)
built with **SimPy**, analysed with **NumPy / pandas / SciPy**, verified with **pytest** and
presented in a **Streamlit + Plotly** dashboard.

> **Synthetic model.** Every input is an illustrative demonstration value and the model has
> not been validated against a real mine. Results are model estimates, not measurements.

## The question

*How many trucks should the mine operate before additional trucks mostly increase queuing
rather than throughput?* The study gives three separate answers:

| Question | Answer (default synthetic inputs, 30 replications × 14 days) |
|---|---|
| **Operational tipping point**: smallest fleet reaching 95% of the production plateau | **6 trucks** (plateau 944 t/h, 95% CI 941–947) |
| **Fleet for an 800 t/h target**: lower 95% CI ≥ target | **5 trucks** (833 t/h, CI 829–838) |
| **Least cost per tonne meeting the target** (illustrative costs) | **6 trucks** |

Beyond 6 trucks the 7th adds ~11 t/h and later trucks almost nothing, while the crusher
wait grows by ~6 minutes per added truck. The crusher (not the shovel) is the constraint
with these inputs. That is a *result*: change the inputs and the shovel or truck availability
becomes limiting.

## Quick start (Windows, PowerShell)

Requires Python 3.11+ (built and tested on 3.12).

```bash
python -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` from cmd.

```bash
pip install -r requirements.txt
```

```bash
python -m pytest
```

```bash
python -m haulsim demo
```

```bash
streamlit run dashboard/app.py
```

The dashboard opens at http://localhost:8501. It loads the saved study in `results/study`
instantly ("Load saved CLI study"), or runs new experiments with the inputs in the sidebar.

### Command line

| Command | What it does | Time* |
|---|---|---|
| `python -m haulsim demo` | 5 replications × 3 days, fleets 1–12, prints a table and recommendation | ~3 s |
| `python -m haulsim study` | 30 replications × 14 days (+24 h warm-up), fleets 1–20, saves `results/study` | ~20 s |
| `python -m haulsim scenarios` | baseline / breakdowns only / shifts only / combined | ~40 s |
| `python -m haulsim sensitivity` | warm-up profile and warm-up × run-length grid | ~20 s |
| `python -m haulsim bounds` | idealised capacity bounds by fleet size | instant |

\*with parallel workers on a 16-thread laptop; use `--workers 1` for a single process.
Every command accepts `--fleet 1-20 --reps 30 --days 14 --warmup-h 24 --seed N --config my_mine.json`.

## How the model works

```
[shovel queue] → load → haul loaded → [crusher queue] → dump (tonnes counted) → return empty → repeat
```

* Shovel and crusher are finite-capacity, first-come-first-served `simpy.Resource`s.
* Activity times are triangular (minimum, most likely, maximum) with separate random
  streams per source of randomness.
* **Breakdowns** use an operating-time clock that advances only while a truck is working.
  A failed truck is repaired in place, keeps any station it occupies blocked, and resumes
  its activity with the *remaining* duration.
* **Shift changes** stop all productive work for 20 min every 12 h; queues and loads persist,
  repairs continue, and overlaps with repairs are counted once.
* The engine writes raw event logs; metrics, tests and the dashboard replay all read them.

Full details: [docs/MODEL.md](docs/MODEL.md).

## Results

Combined scenario (breakdowns + shift stoppages); the dashboard's Recommendation tab plots
these with confidence bands and the capacity bound.

| Trucks | t/h (95% CI) | Gain/truck | Cycle (min) | Crusher wait (min) | Truck time queuing |
|---|---|---|---|---|---|
| 4 | 682 (678–685) | +163 | 35.2 | 1.0 | 4% |
| 5 | 833 (829–838) | +151 | 36.0 | 1.7 | 6% |
| **6** | **919 (916–923)** | **+86** | 39.2 | 4.7 | 13% |
| 7 | 930 (927–934) | +11 | 45.2 | 10.7 | 24% |
| 8 | 931 (928–935) | +1 | 51.5 | 17.0 | 33% |
| 12 | 939 (936–943) | +3 | 76.7 | 41.7 | 54% |
| 20 | 944 (940–947) | −1 | 127.1 | 91.3 | 71% |

The plateau is explained by the crusher's time split at 20 trucks: 94.4% productive, 2.8%
blocked by broken trucks, 2.7% shift stoppages, 0.1% idle → 0.944 × 1000 t/h. See the
[engineering report](docs/REPORT.md) for the experiment design, sensitivity checks and
scenario comparison.

## Verification

49 pytest checks, including:

* exact hand-calculated delivery times for a deterministic truck, with and without a stoppage;
* breakdown resumption with remaining duration; a failure at the crusher blocking it for exactly
  the repair time; a repair overlapping a shift stoppage, counted once;
* resource capacities never exceeded, FIFO order, tonnes loaded = delivered + on board,
  no duplicate deliveries, no work during stoppages;
* state-time accounting closing to the window; queue-length integral equal to total waiting time;
* throughput within capacity bounds; the bottleneck moving when inputs change;
* the replay agreeing with the logs and reported metrics.

Verification shows the code implements the stated model. It is **not** validation against a mine.

## Project layout

```
haulsim/            simulation engine (no UI imports)
  config.py         inputs, defaults, validation, units
  model.py          SimPy processes and event logs
  metrics.py        window-based measures from logs
  experiment.py     replications, seeds, scenarios, sensitivity
  analysis.py       CIs, capacity bounds, tipping point, target, economics
  replay.py         truck positions for the 2D replay
  cli.py            python -m haulsim ...
dashboard/          Streamlit app and Plotly charts
tests/              pytest verification suite
results/            saved study, scenario and sensitivity outputs (CSV/JSON)
docs/               MODEL.md, REPORT.md, INTERVIEW.md
```

## Limitations

No hopper or stockpile between tipping and crushing, no road congestion, unlimited repair
crews, failed trucks block stations, a single shovel and destination, and assumed rather than
fitted distributions. Each is discussed, with the change it would make, in
[docs/REPORT.md](docs/REPORT.md#7-limitations-and-next-steps).

## What this project demonstrates

Discrete-event modelling of interruptions with preserved progress; careful output analysis
(warm-up, independent replications, confidence intervals, common random numbers, paired
comparisons); decision rules that report uncertainty instead of forcing an answer; and a
testable separation between engine, analysis and UI. 
