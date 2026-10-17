# Engineering report: truck fleet size for a shovel-crusher haul loop

**Status:** study of a *synthetic* system. All inputs are illustrative
demonstration values and the model has **not** been validated against a real
mine. Figures below are estimates from this model only.

Reproduce every number in this report with:

```bash
python -m haulsim study        # results/study
python -m haulsim scenarios    # results/scenarios
python -m haulsim sensitivity  # results/sensitivity
```

Base seed 20261017. The parameters and per-replication seeds are saved in each
folder's `parameters.json`.

## 1. Question

How many trucks should the operation run before additional trucks mostly
increase queuing rather than throughput? Three answers are kept separate:

1. **Operational tipping point**: where production begins to plateau.
2. **Fleet for a production target**: the smallest fleet that meets it.
3. **Economically preferred fleet**: depends on explicit cost assumptions.

## 2. Model summary

Discrete-event simulation in SimPy (see [MODEL.md](MODEL.md)). One shovel and
one crusher receiving station (finite-capacity FCFS resources), 100 t trucks,
triangular activity times, an operating-time failure clock (exponential,
mean 40 operating hours) with 30-60-120 min repairs in place, and a 20-minute
production stoppage every 12 hours. Tonnes count when a dump finishes.

Key assumptions and simplifications:

| Assumption | Effect / why acceptable for this study |
|---|---|
| Crusher receiving and crushing combined, no hopper | Crusher starves as soon as no truck is present; slightly pessimistic for small fleets |
| No road congestion; travel time independent of fleet | Optimistic for large fleets, which would slow each other down |
| A truck that fails at a station blocks it until repaired | Pessimistic vs. towing the truck away; costs ~2.8% of crusher time here |
| Failure clock runs only while working | Queued trucks do not wear; trucks in large fleets fail less per calendar hour |
| Repairs in place, unlimited maintenance crews | No repair queue; optimistic if crews are scarce |
| Shift change stops every activity for 20 min; repairs continue | Explicit events, not a productivity factor |
| One payload, one shovel face, one destination | No blending, grade or dispatching decisions |
| 24/7 schedule, no planned maintenance, weather or blasting delays | Throughput is per *scheduled* hour of a continuous operation |

## 3. Verification

`python -m pytest` runs 49 checks (all passing at the time of writing). They
include:

* **Hand calculation**: one deterministic truck delivers at 22 + 32k minutes,
  45 loads (4,500 t) per day; with a 20-min stoppage at 720 min, deliveries
  after it are exactly 20 min later (44 per day).
* **Breakdown resumption**: a failure 8 min into a 10-min return resumes with
  2 min left after repair; a failure during dumping blocks the crusher for
  exactly the repair time and the load is delivered once.
* **Overlapping interruptions**: a repair that spans a shift stoppage; the
  truck restarts only when both have ended, and the overlap counts once.
* **Invariants on stochastic runs**: station capacity never exceeded; FIFO
  service order; tonnes loaded = delivered + on board; no trip delivered
  twice; no productive work inside stoppages; per-truck state times sum to the
  window; station time splits sum to scheduled time; the time integral of
  queue length equals the summed waiting time from an independent log.
* **Bounds**: simulated throughput stays within the availability-adjusted
  capacity bound; a large failure-free fleet approaches but cannot exceed
  crusher capacity (sample-path limit from the minimum dump time).
* **Bottleneck is an outcome**: a slower shovel makes the shovel limiting;
  unreliable trucks make the fleet limiting.
* **Replay consistency**: the animation's queue counts, repair states and
  delivery counter agree with the logs and the reported metrics.

These tests show the code does what the model description says. They say
nothing about whether the model matches a real operation.

## 4. Experiment design

* Fleet sizes 1-20, **30 independent replications** each (600 runs).
* Warm-up 24 h (excluded without resetting the mine), then a **14-day**
  measurement window (whole days, so 28 shift changes in every run).
* Replication *r* uses the same seed for every fleet size (common random
  numbers). Replications are independent of each other, so per-fleet 95%
  confidence intervals use the t-distribution over 30 replication means; marginal
  gains use paired differences.
* Individual trips are never treated as independent observations.

### Warm-up and run length check (7 trucks, 30 replications)

Trucks start bunched at the shovel, so the first hour delivers 680 t; from the
second hour the hourly average is within the normal range (900-1000 t, with
dips at stoppages). The estimate barely depends on the warm-up:

| Warm-up \ days | 3 | 7 | 14 | 28 |
|---|---|---|---|---|
| 0 h | 927 +/- 10.7 | 927 +/- 6.4 | 930 +/- 4.2 | 929 +/- 2.7 |
| 6 h | 926 +/- 11.2 | 927 +/- 6.2 | 930 +/- 4.0 | 929 +/- 2.6 |
| 24 h | 926 +/- 11.2 | 928 +/- 6.1 | 930 +/- 3.8 | 929 +/- 2.6 |
| 48 h | 927 +/- 8.8 | 930 +/- 5.5 | 931 +/- 3.5 | 929 +/- 2.6 |

(t/h, mean +/- 95% CI half-width.) The start-up transient is short compared
with a 14-day window, so 24 h is a conservative warm-up. Longer windows mainly
tighten the interval (roughly with the square root of run length).

## 5. Results (combined scenario: breakdowns + shift stoppages)

| Trucks | t/h (95% CI) | Gain per added truck | Mean cycle (min) | Crusher wait (min/visit) | Crusher util. of available time | Truck time queuing |
|---|---|---|---|---|---|---|
| 1 | 177 (176-178) | - | 33.9 | 0.0 | 18% | 0% |
| 2 | 350 (349-352) | +173 | 34.2 | 0.2 | 37% | 1% |
| 3 | 519 (517-521) | +169 | 34.7 | 0.6 | 54% | 2% |
| 4 | 682 (678-685) | +163 | 35.2 | 1.0 | 72% | 4% |
| 5 | 833 (829-838) | +151 | 36.0 | 1.7 | 88% | 6% |
| **6** | **919 (916-923)** | **+86** | 39.2 | 4.7 | 97% | 13% |
| 7 | 930 (927-934) | +11 | 45.2 | 10.7 | 99% | 24% |
| 8 | 931 (928-935) | +1 | 51.5 | 17.0 | 99% | 33% |
| 10 | 931 (927-936) | -2 | 64.4 | 29.6 | 99% | 46% |
| 12 | 939 (936-943) | +3 | 76.7 | 41.7 | 99% | 54% |
| 15 | 940 (937-943) | 0 | 95.7 | 60.3 | 100% | 63% |
| 20 | 944 (940-947) | -1 | 127.1 | 91.3 | 100% | 71% |

Shovel utilisation of available time levels off at about 65%, so the shovel is
not the constraint. The capacity bound (177.3 N vs. 972 t/h crusher) predicts
the switch from fleet-limited to crusher-limited between 5 and 6 trucks
(972 / 177.3 = 5.5), which is where the simulation bends.

**Where the plateau comes from.** At 20 trucks the crusher is 94.4% productive,
2.8% blocked by broken trucks, 2.7% stopped for shift changes and 0.1% idle.
94.4% of its 1,000 t/h capacity is the 944 t/h plateau. The slow rise from 931
(8 trucks) to 944 t/h (20 trucks) is the last 1% of crusher starvation being
removed by an ever-longer queue: worth about 1 t/h per truck while costing
~6 extra minutes of cycle time per truck added.

### Scenario comparison (mean t/h, 30 replications)

| Trucks | Baseline | Breakdowns only | Shifts only | Combined |
|---|---|---|---|---|
| 5 | 911 | 856 | 886 | 833 |
| 6 | 999 | 944 | 972 | 919 |
| 8 | 1000 | 960 | 972 | 931 |
| 14 | 1000 | 966 | 972 | 940 |

Shift stoppages remove a fixed 2.8% at the plateau (1000 -> 972) because they
stop every resource simultaneously. Breakdowns cost more at small fleets (lost
truck time) and still cost ~3.5% at large fleets (crusher blocking). The two
effects combine roughly additively.

## 6. Recommendation

| Question | Rule | Answer |
|---|---|---|
| Operational tipping point | Smallest fleet >= 95% of plateau; plateau = mean of 18-20 trucks (944 t/h, CI 941-947); flatness confirmed (rise 0.03% across 18-20) | **6 trucks** (CI-based range 6-6) |
| Fleet for 800 t/h target | Smallest fleet whose lower 95% CI >= target | **5 trucks** (833 t/h, CI 829-838) |
| Economically preferred (illustrative) | Least cost per tonne among fleets meeting the target; truck 300/h, fixed 2,500/h | **6 trucks** (4.68 per t vs 4.80 at 5 and 4.94 at 7) |

Beyond 6 trucks each extra truck adds at most ~11 t/h (the 7th) and then
essentially nothing, while crusher waiting grows by about 6 minutes per truck.
For this synthetic system, **6 trucks** is the operating fleet; **5** is
sufficient if the only requirement is 800 t/h; the economic choice depends on
the cost inputs and should be re-run with real figures.

Maximum throughput alone is not the decision: the largest fleet tested
produces 3% more than 6 trucks at 3.3 times the truck cost.

Note on robustness: the tipping point sits right at the fleet/crusher balance
point (5.5 trucks). Inputs that move that ratio (longer haul, faster crusher,
lower reliability) move the recommendation, which is why the dashboard lets you
change them and re-run.

## 7. Limitations and next steps

* **Validation**: compare cycle-time components, queue lengths and shift
  production with dispatch-system (FMS) data before using any number.
* **Input data**: fit distributions to observed durations instead of assumed
  triangulars; check whether failures really follow a constant hazard.
* **Hopper / stockpile**: model receiving and crushing separately with a
  `Container`; quantify how much a buffer reduces starvation and blocking.
* **Congestion**: speed-flow relationship on the haul road, single-lane
  sections, passing bays.
* **Failure handling at stations**: tow-away after a fixed delay so a broken
  truck stops blocking the crusher.
* **Maintenance resources**: limited crews or bays create a repair queue.
* **Dispatching** with several shovels and destinations.
* **Decision under uncertainty**: express the recommendation as a probability
  of meeting the target per shift rather than a long-run average.
