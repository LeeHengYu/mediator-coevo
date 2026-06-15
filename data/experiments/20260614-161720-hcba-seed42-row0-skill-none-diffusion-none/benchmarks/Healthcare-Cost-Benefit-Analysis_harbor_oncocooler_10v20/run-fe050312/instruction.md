# Task Instruction

Execute the following steps to produce the two required output files.

## Step 1 – Inspect input files

Read and display:
- `/root/program_catalog.json`
- `/root/cooler_cost.csv`
- `/root/contract_payment.csv`
- `/root/site_overrides.csv`

## Step 2 – Write and run a Python script

Create `/root/solve.py` with the logic below, then run it with `python3 /root/solve.py`.

```python
import json, csv, math

# ── Load data ──────────────────────────────────────────────
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

cooler_rows = read_csv('/root/cooler_cost.csv')
payment_rows = read_csv('/root/contract_payment.csv')
override_rows = read_csv('/root/site_overrides.csv')

# ── Filter in-scope programs (review_flag == 'review') ────
in_scope = [p for p in catalog if p.get('review_flag') == 'review']

# ── Build lookup: cooler_type -> cooler_cost_usd ──────────
cooler_map = {r['cooler_type']: float(r['cooler_cost_usd']) for r in cooler_rows}

# ── Build lookup: program_label -> payment row ────────────
# Map each program_label in contract_payment.csv to an in-scope program.
# Match by program_name or any entry in known_labels.
label_to_program = {}
for p in in_scope:
    label_to_program[p['program_name']] = p
    for lbl in p.get('known_labels', []):
        label_to_program[lbl] = p

# For each in-scope program, find its payment row
payment_map = {}  # program_code -> payment_per_dispatch_per_site_usd
for row in payment_rows:
    pl = row['program_label']
    if pl in label_to_program:
        prog = label_to_program[pl]
        payment_map[prog['program_code']] = float(row['payment_per_dispatch_per_site_usd'])

# ── Resolve active sites ──────────────────────────────────
# From site_overrides: only approved rows, highest version_no per program_code
approved = [r for r in override_rows if r['approval_state'] == 'approved']
best_override = {}
for r in approved:
    pc = r['program_code']
    vn = int(r['version_no'])
    if pc not in best_override or vn > best_override[pc][1]:
        best_override[pc] = (r, vn)

def get_active_sites(prog):
    pc = prog['program_code']
    if pc in best_override:
        row = best_override[pc][0]
        return int(row['active_sites'])
    return int(prog['default_active_sites'])

# ── Constants ─────────────────────────────────────────────
DPY_10 = 36
DPY_20 = 18
DAYS_10 = 10
DAYS_20 = 20
THRESHOLD = 10000

# ── Compute per-program ───────────────────────────────────
programs_out = []
for prog in in_scope:
    pc = prog['program_code']
    pn = prog['program_name']
    sites = get_active_sites(prog)
    acq = float(prog['acquisition_cost_per_1000_units_usd'])
    upd = float(prog['units_per_day'])
    ct = prog['cooler_type']
    cc = cooler_map[ct]
    ppd = payment_map[pc]

    # Revenue
    rev_10 = ppd * sites * DPY_10
    rev_20 = ppd * sites * DPY_20

    # Drug cost
    drug_10 = acq * sites * upd * DAYS_10 * DPY_10 / 1000
    drug_20 = acq * sites * upd * DAYS_20 * DPY_20 / 1000

    # Cooler cost
    cooler_10 = cc * sites * DPY_10
    cooler_20 = cc * sites * DPY_20

    # Margin
    margin_10 = rev_10 - drug_10 - cooler_10
    margin_20 = rev_20 - drug_20 - cooler_20
    diff = margin_20 - margin_10

    programs_out.append({
        'program_code': pc,
        'program_name': pn,
        'active_sites': sites,
        'acquisition_cost_per_1000_units_usd': round(acq, 2),
        'units_per_day': round(upd, 2),
        'cooler_type': ct,
        'cooler_cost_usd': round(cc, 2),
        'payment_per_dispatch_per_site_usd': round(ppd, 2),
        'annual_drug_cost_10_day_usd': round(drug_10, 2),
        'annual_drug_cost_20_day_usd': round(drug_20, 2),
        'annual_cooler_cost_10_day_usd': round(cooler_10, 2),
        'annual_cooler_cost_20_day_usd': round(cooler_20, 2),
        'annual_revenue_10_day_usd': round(rev_10, 2),
        'annual_revenue_20_day_usd': round(rev_20, 2),
        'annual_margin_10_day_usd': round(margin_10, 2),
        'annual_margin_20_day_usd': round(margin_20, 2),
        'annual_margin_difference_20_minus_10_usd': round(diff, 2)
    })

# Sort by program_code ascending
programs_out.sort(key=lambda x: x['program_code'])

# ── Totals ────────────────────────────────────────────────
tot_10 = round(sum(p['annual_margin_10_day_usd'] for p in programs_out), 2)
tot_20 = round(sum(p['annual_margin_20_day_usd'] for p in programs_out), 2)
tot_diff = round(tot_20 - tot_10, 2)
abs_diff = round(abs(tot_diff), 2)

if abs_diff < 10000:
    decision = 'move_to_20_day'
else:
    decision = 'keep_10_day'

justification = (f'The absolute total margin difference is ${abs_diff:.2f}, '
                 f'which is {"below" if abs_diff < 10000 else "at or above"} '
                 f'the ${THRESHOLD} threshold, so the recommendation is {decision}.')

result = {
    'assumptions': {
        'dispatches_per_year_10_day': DPY_10,
        'dispatches_per_year_20_day': DPY_20,
        'days_per_dispatch_10_day': DAYS_10,
        'days_per_dispatch_20_day': DAYS_20,
        'switch_threshold_usd': THRESHOLD,
        'site_override_rule': 'highest approved version_no per program_code, else default_active_sites'
    },
    'programs': programs_out,
    'totals': {
        'total_annual_margin_10_day_usd': tot_10,
        'total_annual_margin_20_day_usd': tot_20,
        'total_annual_margin_difference_20_minus_10_usd': tot_diff,
        'absolute_total_margin_difference_usd': abs_diff
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/oncocooler_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

print('JSON written.')

# ── Summary markdown ──────────────────────────────────────
# IMPORTANT: Do NOT use comma-separated number formatting.
# Use f'{value:.2f}' not f'{value:,.2f}'
lines = [
    '# OncoCooler 10-Day vs 20-Day Analysis Summary',
    '',
    f'Total 10-day annual margin (USD): {tot_10:.2f}',
    f'Total 20-day annual margin (USD): {tot_20:.2f}',
    f'Absolute margin difference (USD): {abs_diff:.2f}',
    f'Recommendation: {decision}',
    '',
    f'{justification}'
]

with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Summary written.')
print(f'Decision: {decision}, abs_diff: {abs_diff}')
```

## Step 3 – Validate outputs

1. `cat /root/oncocooler_analysis.json` – confirm it parses, has the `assumptions`, `programs` (sorted by program_code), `totals`, and `recommendation` keys.
2. `cat /root/oncocooler_summary.md` – confirm 4-8 non-empty lines, contains total 10-day margin, total 20-day margin, absolute difference, and the exact decision slug (`move_to_20_day` or `keep_10_day`). Confirm NO comma-separated numbers.
3. Verify currency values are rounded to 2 decimal places.
4. Spot-check one program's math manually if possible.

## Key cautions from cross-task feedback
- **Number formatting in markdown**: Use `f'{value:.2f}'` (NO commas). Different verifiers have been inconsistent about expecting commas vs plain numbers. For this specific task, the previous successful run (reward 1.0) did not mention comma issues, so use plain formatting without commas to be safe.
- **Cooler cost formula**: `cooler_cost_usd * active_sites * dispatches_per_year` (confirmed from prior success).
- **Site overrides**: Only `approved` rows; highest `version_no` per `program_code`; fallback to `default_active_sites`.
- **Payment matching**: Match `program_label` against both `program_name` and `known_labels` entries.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[oncology, json, csv, structural-adaptation, decision-analysis].
Verifier config: timeout_sec=900.0.