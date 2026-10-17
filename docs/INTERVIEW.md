# Interview talking points

## The 30-second version

"I built a discrete-event simulation of an open-pit haul loop in Python with
SimPy: trucks cycle between a shovel and a crusher, with random activity
times, breakdowns on an operating-hours clock, and shift-change stoppages. I
ran 30 independent replications for each fleet size from 1 to 20 trucks, put
95% confidence intervals on throughput, and used a stated rule to find where
production plateaus. In my synthetic case it is 6 trucks. Past that, extra
trucks only lengthen the crusher queue. I separated that from the fleet needed
for a production target and from the cheapest fleet under explicit costs. The
engine is verified against hand calculations and invariants with pytest, and
a Streamlit dashboard shows the results, including a replay driven by the
event log."

## What I would highlight

1. **Framing**: three different answers (tipping point, target, economics)
   instead of "maximum throughput".
2. **Interruption logic**: one `_work` loop handles completion, failure,
   stoppage and ties by subtracting elapsed time; remaining work is never lost
   or restarted.
3. **Statistics**: replications, not trips, are the unit of analysis; common
   random numbers with paired differences for marginal gains; warm-up and
   run-length sensitivity.
4. **Honesty about uncertainty**: the tipping-point rule reports "plateau not
   reached" or "uncertain" instead of forcing an answer (try the quick demo
   with 5 replications: it says "uncertain, 6-8 trucks").
5. **Verification vs validation**: I can show the code is right for the
   model; I cannot claim the model is right for a mine without data.

## Likely questions and how I would answer

**Why discrete-event simulation rather than a spreadsheet or queueing formula?**
The capacity bound (min of fleet, shovel, crusher) gives the right *shape*
but not the losses: random service times make trucks bunch, broken trucks
block stations, and stoppages interrupt cycles. Closed-network queueing
formulas (e.g. mean value analysis) need exponential or simple service times
and can't easily handle station blocking and timed stoppages. DES handles all
of these directly and gives distributions, not just means.

**Why triangular distributions?**
They are bounded and easy to elicit ("usually 4 min, never under 3 or over
5"). With real data I would fit distributions (e.g. lognormal for travel,
check failure data for a constant hazard) and compare goodness of fit.

**How did you choose the warm-up and run length?**
I plotted hourly production from a cold start averaged over 30 replications:
the transient lasts about an hour. Then I compared estimates for warm-ups of
0-48 h and windows of 3-28 days: estimates agree within their CIs and the CI
narrows roughly with the square root of run length. 24 h warm-up and 14 days
is conservative. Windows are whole days so each run has the same number of
shift changes.

**Why not compute confidence intervals from all the trips?**
Trips within a run are correlated (one long repair delays several following
trips), so treating them as independent would give intervals that are far too
narrow. Each replication gives one independent summary value.

**What are common random numbers and why use them?**
Replication *r* uses the same seed for every fleet size, so differences between
fleet sizes are less affected by luck. Because each source of randomness has
its own stream, the effect is stronger. Per-fleet CIs stay valid because
replications remain independent of each other.

**How is the tipping point defined?**
Smallest fleet whose mean throughput reaches 95% of the plateau, with the
plateau estimated as the mean over the three largest fleets tested. Before
using it, I check the plateau really is flat: the paired rise across those
fleets must be within 2%, using its confidence interval. I also report the
range of fleets compatible with the CIs. The 95% and 2% are configurable;
they are decision rules, not physics.

**Why did throughput keep rising slightly after 8 trucks?**
Breakdowns. At 8 trucks the crusher is idle about 1% of the time, when trucks
under repair leave gaps. A longer queue fills those gaps, so throughput rises
by about 1 t/h per truck up to 20, while each truck adds about 6 minutes to
everyone's cycle. The time split at the crusher (94.4% productive, 2.8%
blocked, 2.7% shift, 0.1% idle) adds up to the 944 t/h plateau.

**Why does truck availability rise with fleet size?**
Because the failure clock runs on operating time. In a big fleet trucks spend
most of their time queuing, so they accumulate fewer operating hours per day
and fail less per calendar hour. If idling wear matters, the clock could
advance at a reduced rate while queuing. That is a modelling choice I'd
discuss with maintenance engineers.

**What happens if a truck breaks down under the shovel?**
In this baseline it stays there and blocks the shovel until repaired. The
metrics separate "blocked" from "productive" time so the effect is visible.
A tow-away rule (after X minutes, release the station) is a natural extension.

**How do you know the bottleneck isn't hard-coded?**
A test makes the shovel slower than the crusher and checks the analysis
identifies the shovel; another makes trucks unreliable and checks the fleet
is identified. The limiting resource is read from utilisation of *available*
time, not assumed.

**How did you verify the model?**
Hand-calculated deterministic cycles (exact delivery times), constructed
breakdown and overlap scenarios with known answers, and invariants on random
runs: capacity, FIFO, mass balance, no duplicate deliveries, time accounting
closing to the window, queue-length integral equal to total waiting time from
an independent log, and throughput within capacity bounds.

**What would you do with real data?**
Validate against FMS data (cycle components, queue times, shift tonnes), fit
inputs, add the hopper, road congestion and repair-crew limits, then re-run
the study. I'd present results as ranges and probabilities of meeting the
target, not single numbers.

## What I learned

* The difference between simulation *time* and wall-clock or playback time,
  and why event-driven state changes make interruption logic easy to reason
  about when progress is stored explicitly.
* Small definitional choices (what counts as production, which states advance
  the failure clock, how overlaps are attributed, window boundary rules)
  change the numbers, so they must be written down and tested.
* Output analysis is most of the work: warm-up, replications, CIs, paired
  comparisons and checking that a "plateau" really is one.
* Keeping the engine separate from the UI made it possible to test everything
  from the command line and to guarantee the dashboard shows the same numbers.
