# Model description

> All input values are **synthetic demonstration values**. Nothing in this
> model has been validated against a real mine.

## 1. System and event flow

One loading shovel feeds one crusher over a fixed haul road. Each truck runs
this cycle until the simulation ends:

```
          +-------------------------------- return empty (8-10-12 min) ---------------------------+
          v                                                                                       |
 [shovel queue] -> LOAD (3-4-5 min) -> haul loaded (10-12-14 min) -> [crusher queue] -> DUMP (5-6-7 min)
     FCFS            holds shovel                                       FCFS         holds crusher,
                                                                                     tonnes counted
                                                                                     when DUMP ends
```

* The shovel and crusher are `simpy.Resource` objects with finite capacity
  (default 1 each). Their queues are first-come-first-served.
* **Production is counted only when a dump finishes.** Loading only puts
  tonnes on the truck.
* Travel times do not depend on how many trucks are on the road (no
  congestion, no overtaking rules).
* Simulation time is in **minutes** inside the engine. User inputs in hours
  are converted on entry (`config.py`).

### Crusher as a single station (simplification)

Crusher *receiving* (tipping into the hopper) and *processing* are merged into
one service station with no stockpile between them. A truck occupies the
station for the whole 5-7 min "crusher service".

With a hopper or ROM stockpile, receiving and crushing would be decoupled:
trucks could tip quickly while the hopper has space, the crusher would keep
running for a while when trucks are late (fewer starvation losses), and trucks
would only queue when the hopper is full (blocking). That would be modelled as
a `simpy.Container` (hopper level) between a short tipping resource and a
crusher process that draws from it at a rate. Throughput would rise slightly
for small fleets and the plateau would be set by the crusher rate rather than
the tipping time.

## 2. Random inputs

Durations use **triangular distributions (minimum, most likely, maximum)**:
bounded, easy to elicit from people who know an operation, and adequate for a
first study. Time between failures is **exponential** in operating time (a
constant failure rate - the standard "memoryless" assumption).

Each source of randomness has its own NumPy generator spawned from one
`SeedSequence`: loading, hauling, dumping, returning, failure clock and
repair. A replication is fully reproduced from one integer seed.

With **randomness disabled**, every duration equals its distribution *mean*
(so the deterministic model has the same average workload). This is what
the hand-calculation tests use.

## 3. Breakdowns: operating-time failure clock

* Each truck has `ops_to_failure`, the **operating minutes** left until it
  next fails, drawn from Exponential(mean = MTBF = 40 h).
* The clock **advances only while the truck is loading, hauling, dumping or
  returning**. It does not advance while the truck is queuing, under repair or
  stopped for a shift change. Rationale: failures are driven by work done
  (engine hours under load), not calendar time. Consequence: a truck that
  queues a lot fails less often per calendar hour.
* An activity runs in chunks: `min(remaining work, ops_to_failure)`. When the
  clock reaches zero the truck is repaired **in place** (30-60-120 min), the
  clock is redrawn, and the activity **resumes with its remaining duration**.
  Nothing restarts and nothing is delivered twice.
* **A truck that fails while loading or dumping keeps the station blocked**
  until it is repaired (no tow-away or bypass in this baseline). On the road it
  blocks nobody.

## 4. Shift changes

* Every 12 h (from t = 0, the start of a shift) all production stops for
  20 min. In-progress activities stop and keep their remaining work; queued
  trucks stay queued; loads stay on trucks.
* **Repairs continue during a stoppage.** A truck whose repair ends during a
  stoppage waits until the stoppage ends.
* Breakdowns cannot *start* during a stoppage, because the failure clock is not
  running.
* Overlaps are counted once: a truck under repair during a stoppage is
  recorded as "repair", not also as "shift stop".

These are explicit events, not a productivity multiplier: a stoppage
interrupts specific cycles at specific times, and the queues it creates
afterwards are part of the result.

## 5. Implementation

| File | Responsibility |
|---|---|
| `haulsim/config.py` | Dataclasses, defaults, validation, unit conversion |
| `haulsim/model.py` | SimPy processes: `Truck.run`, `_visit_station`, `_work`, `_breakdown`, shift controller. Writes raw event logs |
| `haulsim/metrics.py` | Turns logs into measures for a window `(a, b]` |
| `haulsim/experiment.py` | Replications, seeds, scenarios, warm-up and run-length studies |
| `haulsim/analysis.py` | Confidence intervals, capacity bounds, tipping point, target, economics |
| `haulsim/replay.py` | Truck positions for the 2D replay, read from the same logs |
| `haulsim/cli.py` | `python -m haulsim demo|study|scenarios|sensitivity|bounds` |
| `dashboard/` | Streamlit UI; imports the engine, never the other way round |

The core of the interruption logic is `Truck._work`:

```python
while remaining > 0:
    if mine.shift_active:                     # paused: wait for the stoppage to end
        yield mine.shift_end_ev
        continue
    chunk = min(remaining, self.ops_to_failure)
    t0 = env.now
    yield env.timeout(chunk) | mine.shift_start_ev   # finish, fail, or be stopped
    elapsed = env.now - t0
    remaining -= elapsed                      # keep progress whatever happened
    self.ops_to_failure -= elapsed
    if self.ops_to_failure <= 0:
        yield from self._breakdown(kind)      # repair in place, keep any station held
```

Because elapsed time is always subtracted, the same code handles completion,
failure, stoppage and ties between them.

## 6. Measurement window and metrics

* Warm-up observations are excluded by measuring over `(warm-up, horizon]`.
  The mine is **not reset** at the end of the warm-up.
* Measurement periods are **whole days**, so every run contains the same
  number of shift changes (two per day).
* Events (deliveries, completed cycles, completed waits) are attributed to the
  moment they complete. A dump that straddles the warm-up boundary counts in
  the window; one in progress at the end does not.
* Time-based measures (state times, utilisation, queue lengths) clip each
  interval to the window, so boundary-crossing activities contribute exactly
  their share.
* **Cycle time** = shovel-queue arrival to next shovel-queue arrival, for cycles
  that complete in the window. Cycles still open at the end are censored; long
  cycles (those with repairs) are the most likely to be censored, so upper
  percentiles are slightly understated. With 14-day windows this is one open
  cycle per truck against hundreds completed.

| Measure | Definition | Denominator |
|---|---|---|
| Throughput | tonnes delivered in window / window hours | scheduled hours (24 h/day) |
| Waiting time | queue join to service start, per visit | visits starting in window |
| Queue length | time-weighted number waiting (not in service) | window length |
| Truck state fractions | load, haul, dump, return, queue (shovel/crusher), repair, shift stop | trucks x window |
| Truck availability | 1 - repair fraction | trucks x window |
| Station productive / blocked / shift / idle | being loaded or dumped / occupied by a broken truck / stoppage (not blocked) / empty | servers x window |
| Utilisation of available time | productive / (scheduled - shift - blocked) | as stated |

## 7. Capacity bounds

Idealised (no variability, failures or stoppages), in t/h:

* fleet = N x payload x 60 / E[cycle] = N x 100 x 60 / 32 = 187.5 N
* shovel = shovels x payload x 60 / E[load] = 1500
* crusher = stations x payload x 60 / E[dump] = 1000

Throughput cannot exceed the minimum. Adjusting for stoppages (s = 20/720)
and repairs (rho = E[repair] / MTBF = 70 / 2400) gives 177.3 N, 1458 and 972 t/h,
so the bound switches from fleet-limited to crusher-limited between 5 and 6
trucks. Realised output is lower because random service times make trucks
bunch (the crusher idles while trucks are elsewhere, then faces a queue),
because broken trucks block stations, and because stoppages interrupt cycles.
