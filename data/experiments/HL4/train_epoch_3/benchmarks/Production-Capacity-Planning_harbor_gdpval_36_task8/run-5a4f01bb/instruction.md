# Task Instruction

Execute the following steps in order to produce the two deliverables.

## Step 1 – Inspect the input data

```bash
cat /root/chemical_demand.json | python3 -c "import json,sys; data=json.load(sys.stdin); print(len(data)); [print(d) for d in data[:5]]"
```

Confirm each object has keys `week`, `data.demand_per_week`, and `priority`.

## Step 2 – Build the schedule and write both files

Create and run `/root/solve.py` with exactly this logic:

```python
import json, openpyxl, textwrap

# ── Load demand ──────────────────────────────────────────────
with open('/root/chemical_demand.json') as f:
    raw = json.load(f)

# First valid (non-null demand) occurrence per week
demand_map = {}
for entry in raw:
    w = entry['week']
    if w not in demand_map:
        d = entry['data']['demand_per_week']
        if d is not None:
            demand_map[w] = d

# ── Schedule parameters ──────────────────────────────────────
phases = list(range(10, 59))          # 10..58 inclusive, 49 phases
initial_condition = 1453.06           # Start of Phase Past Due + Scheduled Demand at Phase 10
rate = 40                             # std hrs per day

# ── Compute schedule ─────────────────────────────────────────
rows = []
first_5 = None
first_4 = None

for i, phase in enumerate(phases):
    scheduled_demand = demand_map[phase]

    if i == 0:
        # Initial condition: calc_start + scheduled_demand = 1453.06
        calc_start = initial_condition - scheduled_demand
    else:
        calc_start = rows[-1]['end_backlog']

    start_past_due = max(0.0, calc_start)   # for reporting

    # Choose days worked
    if start_past_due > 0.01:
        # Try 5 first, then 6; if neither clears backlog, pick 6
        chosen = 6  # default fallback
        for d in [5, 6]:
            if calc_start + scheduled_demand - (rate * d) <= 0:
                chosen = d
                break
    else:
        chosen = 4 if scheduled_demand <= 160 else 5

    weekly_cap = rate * chosen
    end_backlog = calc_start + scheduled_demand - weekly_cap
    overtime = 10 * max(0, chosen - 4)

    rows.append({
        'phase': phase,
        'days': chosen,
        'demand': scheduled_demand,
        'capacity': weekly_cap,
        'past_due': start_past_due,
        'end_backlog': end_backlog,
        'overtime': overtime,
    })

    if first_5 is None and chosen == 5:
        first_5 = phase
    if first_4 is None and chosen == 4:
        first_4 = phase

# ── Write Excel ──────────────────────────────────────────────
wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Plan'
headers = [
    'Phase',
    'Days Worked',
    'Scheduled Demand (Std Hrs)',
    'Weekly Capacity (Std Hrs)',
    'Start of Phase Past Due (Std Hrs)',
    'End of Phase Backlog/Buffer (Std Hrs)',
    'Overtime Hours',
]
ws.append(headers)
for r in rows:
    ws.append([
        r['phase'],
        r['days'],
        round(r['demand'], 2),
        round(r['capacity'], 2),
        round(r['past_due'], 2),
        round(r['end_backlog'], 2),
        round(r['overtime'], 2),
    ])
wb.save('/root/chemical_schedule_plan.xlsx')
print('Excel written with', len(rows), 'data rows')

# ── Write summary ────────────────────────────────────────────
f5 = str(first_5) if first_5 is not None else 'N/A'
f4 = str(first_4) if first_4 is not None else 'N/A'

summary_body = (
    f"The crew operated at 6-day weeks to clear the initial backlog, "
    f"stepping down to 5-day weeks at Phase {f5} and to 4-day weeks at Phase {f4}. "
    f"This catch-up plan eliminates past-due hours while minimizing overtime."
)

with open('/root/chemical_schedule_summary.txt', 'w') as f:
    f.write(f'First_Week_5_Days: {f5}\n')
    f.write(f'First_Week_4_Days: {f4}\n')
    f.write(f'Summary: {summary_body}\n')

print('Summary written')
print(f'First_Week_5_Days: {f5}')
print(f'First_Week_4_Days: {f4}')

# Sanity checks
assert len(rows) == 49, f'Expected 49 rows, got {len(rows)}'
print('Phase 10 check: calc_start + demand =',
      round(rows[0]['end_backlog'] + rows[0]['capacity'], 2))
for r in rows:
    assert r['days'] in (4, 5, 6)
print('All assertions passed')
```

## Step 3 – Run and verify

```bash
cd /root && python3 solve.py
```

Check the printed output confirms 49 rows, correct Phase 10 initial condition, and valid days.

## Step 4 – Validate outputs exist and look correct

```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/chemical_schedule_plan.xlsx')
ws = wb['Plan']
print('Headers:', [c.value for c in ws[1]])
print('Row count (excl header):', ws.max_row - 1)
print('First data row:', [c.value for c in ws[2]])
print('Last data row:', [c.value for c in ws[ws.max_row]])
"
```

```bash
cat /root/chemical_schedule_summary.txt
```

Confirm:
- Worksheet name is exactly `Plan`
- 7 headers match specification exactly
- 49 data rows, phases 10-58
- Summary file has exactly 3 lines with correct format
- Summary body is ≤ 60 words and ≤ 3 sentences

If the summary exceeds 60 words, edit it down. If any phase's demand is missing from the JSON (KeyError), inspect the JSON to find the correct key structure and fix accordingly.

## Step 5 – Run verifier if available

```bash
ls /root/test_output.py 2>/dev/null && cd /root && python3 -m pytest test_output.py -v
```

If tests fail, read the failure output carefully, identify the mismatch, and fix the logic or output accordingly. Common issues to watch for:
- Rounding differences (use raw floats, only round for display)
- The initial condition interpretation (calc_start = 1453.06 - demand[10])
- The `First_Week_5_Days` / `First_Week_4_Days` must track the FIRST occurrence of that exact day count, not the first step-down

# Executor Policy

---
name: executor
description: Portable executor policy for workflow, verification, resource use, and failure handling across task runtimes.
---

## Executor Policy

Use this skill as execution policy, not as domain-specific task knowledge. When
task-local curated skills or resources are available, prefer them for domain
details and use this policy for workflow control.

## Task Execution

1. Read the task instruction, task resources, and verifier contract before editing.
2. Identify the scoring mechanism and the smallest command that can reproduce the
   failure or verify the expected behavior.
3. Inspect existing files and task-local resources before making changes.
4. Make the smallest source change that satisfies the task and verifier contract.
5. Keep a compact record of the concrete evidence behind the change: observed
   failure, files inspected, edit made, and verifier result.
6. Run targeted verification before broad verification when practical.

## File Editing

1. Read the actual current file contents immediately before making any edit.
   Never rely on memory, prior snapshots, or assumed content.
2. Prefer direct in-place edits over patch or diff application when the exact
   current context is uncertain.
3. If using a patch or diff, confirm that every context line exists verbatim in
   the file before applying it.
4. If a patch hunk fails to apply, re-read the affected file region and perform
   the edit directly instead of retrying the same patch.
5. After any edit, re-read the affected region to confirm the change landed.

## Build and Test Fixes

When a task requires fixing a broken build, failing test, or generated artifact:

1. Run the relevant build, test, or verifier command first to capture the
   baseline failure.
2. Identify the specific error message, file, line, or expected output before
   editing.
3. Apply the smallest fix, then re-run the same targeted command.
4. Treat newly introduced failures as separate sub-tasks and resolve them in
   order.
5. Do not mark the task complete until the verifier-relevant command succeeds or
   the remaining failure is clearly outside the task boundary.

## Artifact-Contract Handling

Do not treat artifacts as ordinary text files. Treat them as contract-bearing
interfaces between input data, generated output, verifier checks, and downstream
consumers.

When a task requires reading, modifying, or generating an artifact such as JSON,
DOT, reports, configs, generated source, schemas, datasets, or parsed outputs:

1. Identify the artifact contract first: format, schema, required fields,
   identifiers, references, ordering, examples, verifier assertions, and
   consuming code.
2. Inspect representative source artifacts directly before deciding how to
   transform or preserve them.
3. Determine whether the task calls for preservation, transformation, repair,
   generation, or validation.
4. Preserve required literals, identifiers, references, ordering, and
   representative content unless the contract explicitly requires a change.
5. Do not invent, drop, rename, normalize, collapse, expand, or repair artifact
   elements unless the verifier or consumer contract requires that behavior.
6. Prefer structured parsers, serializers, validators, or existing consumer code
   over ad hoc string manipulation when they are available.
7. After producing the artifact, run targeted checks for parseability, required
   keys or IDs, reference consistency, expected counts, preserved content, and
   format-specific validity.
8. If targeted checks regress or become unusable after a change, stop expanding
   the solution. Re-inspect the source contract and narrow the edit before trying
   a broader repair.

A plausible-looking artifact is not sufficient evidence. The artifact is only
correct when it satisfies the task contract under the verifier or consuming
code.

## Constraints

- Do not bypass, remove, or weaken tests, verifier scripts, fixtures, or expected
  output checks.
- Do not treat this policy as overriding task-specific instructions or verifier
  requirements.
- On tool or environment errors, retry once when the retry is safe, then report
  the failure with the command and error output.
- On ambiguous instructions, make a conservative assumption and continue.

# Task Resources

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[json, xlsx, operations, capacity-planning, chemical, backlog].
Verifier config: timeout_sec=900.0.