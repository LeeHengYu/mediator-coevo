# Task Instruction

Execute the following steps in order to produce the two required output files.

## Step 1 – Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

Understand the structure and data before writing any code.

## Step 2 – Write and run the solver script

Create `/root/solve.py` with the following logic (use Python 3 with only stdlib + json + csv):

```python
import json, csv, math
from collections import defaultdict

# ── Load inputs ──────────────────────────────────────────────
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

cooler_rows = read_csv('/root/cooler_cost.csv')
contract_rows = read_csv('/root/contract_payment.csv')
site_rows = read_csv('/root/site_overrides.csv')

# ── Build cooler-cost lookup ─────────────────────────────────
cooler_cost_map = {}
for r in cooler_rows:
    cooler_cost_map[r['cooler_type'].strip()] = float(r['cooler_cost_usd'])

# ── Filter in-scope programs (review_flag == 'review') ───────
in_scope = [p for p in catalog if p.get('review_flag') == 'review']

# ── Build label → program_code mapping ───────────────────────
label_to_code = {}
for p in in_scope:
    label_to_code[p['program_name'].strip().lower()] = p['program_code']
    for lbl in p.get('known_labels', []):
        label_to_code[lbl.strip().lower()] = p['program_code']

# ── Resolve contract payments per program_code ───────────────
payment_map = {}  # program_code -> payment_per_dispatch_per_site_usd
for r in contract_rows:
    key = r['program_label'].strip().lower()
    if key in label_to_code:
        code = label_to_code[key]
        payment_map[code] = float(r['payment_per_dispatch_per_site_usd'])

# ── Resolve active sites per program_code ────────────────────
# Filter approved rows, keep highest version_no per program_code
approved = [r for r in site_rows if r['approval_state'].strip().lower() == 'approved']
best_override = {}
for r in approved:
    pc = r['program_code'].strip()
    vn = int(r['version_no'])
    if pc not in best_override or vn > best_override[pc][1]:
        best_override[pc] = (int(r['active_sites']), vn)

# ── Constants ────────────────────────────────────────────────
DISP_10 = 36
DISP_20 = 18
DAYS_10 = 10
DAYS_20 = 20
THRESHOLD = 10000

# ── Per-program calculations ─────────────────────────────────
programs = []
for p in sorted(in_scope, key=lambda x: x['program_code']):
    pc = p['program_code']
    pn = p['program_name']

    # Active sites
    if pc in best_override:
        active_sites = best_override[pc][0]
    else:
        active_sites = int(p['default_active_sites'])

    acq = float(p['acquisition_cost_per_1000_units_usd'])
    upd = float(p['units_per_day'])
    ct = p['cooler_type'].strip()
    cooler_usd = cooler_cost_map[ct]
    pay = payment_map[pc]

    # ── 10-day model ──
    drug_10 = acq * active_sites * upd * DAYS_10 * DISP_10 / 1000
    cooler_10 = cooler_usd * active_sites * DISP_10          # NOTE: includes active_sites
    rev_10 = pay * active_sites * DISP_10
    margin_10 = rev_10 - drug_10 - cooler_10

    # ── 20-day model ──
    drug_20 = acq * active_sites * upd * DAYS_20 * DISP_20 / 1000
    cooler_20 = cooler_usd * active_sites * DISP_20          # NOTE: includes active_sites
    rev_20 = pay * active_sites * DISP_20
    margin_20 = rev_20 - drug_20 - cooler_20

    diff = margin_20 - margin_10

    programs.append({
        'program_code': pc,
        'program_name': pn,
        'active_sites': active_sites,
        'acquisition_cost_per_1000_units_usd': round(acq, 2),
        'units_per_day': round(upd, 2),
        'cooler_type': ct,
        'cooler_cost_usd': round(cooler_usd, 2),
        'payment_per_dispatch_per_site_usd': round(pay, 2),
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

# ── Totals ───────────────────────────────────────────────────
tot_10 = sum(pr['annual_margin_10_day_usd'] for pr in programs)
tot_20 = sum(pr['annual_margin_20_day_usd'] for pr in programs)
tot_diff = round(tot_20 - tot_10, 2)
abs_diff = round(abs(tot_diff), 2)

if abs_diff < THRESHOLD:
    decision = 'move_to_20_day'
    justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                     f'which is below the ${THRESHOLD:,.2f} threshold. '
                     f'Moving to 20-day dispatches is recommended.')
else:
    decision = 'keep_10_day'
    justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                     f'which exceeds the ${THRESHOLD:,.2f} threshold. '
                     f'Keeping 10-day dispatches is recommended.')

result = {
    'assumptions': {
        'dispatches_per_year_10_day': DISP_10,
        'dispatches_per_year_20_day': DISP_20,
        'days_per_dispatch_10_day': DAYS_10,
        'days_per_dispatch_20_day': DAYS_20,
        'switch_threshold_usd': THRESHOLD,
        'site_override_rule': 'highest approved version_no per program_code, else default_active_sites'
    },
    'programs': programs,
    'totals': {
        'total_annual_margin_10_day_usd': round(tot_10, 2),
        'total_annual_margin_20_day_usd': round(tot_20, 2),
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

# ── Markdown summary ─────────────────────────────────────────
with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('# OncoCooler 10-Day vs 20-Day Dispatch Analysis\n')
    f.write(f'\n')
    f.write(f'Total 10-day annual margin: ${round(tot_10, 2):,.2f} USD\n')
    f.write(f'Total 20-day annual margin: ${round(tot_20, 2):,.2f} USD\n')
    f.write(f'Absolute margin difference: ${abs_diff:,.2f} USD\n')
    f.write(f'\n')
    f.write(f'Decision: {decision}\n')
    f.write(f'\n')
    f.write(f'{justification}\n')

print('Done. Files written.')
```

Run it:
```bash
python3 /root/solve.py
```

## Step 3 – Validate outputs

```bash
cat /root/oncocooler_analysis.json
cat /root/oncocooler_summary.md
```

Confirm:
- JSON is valid and parseable.
- `programs` array is sorted by `program_code` ascending.
- All currency values are rounded to 2 decimal places.
- `annual_cooler_cost_*` includes `active_sites` in the multiplication (this was the bug in the previous attempt).
- Summary has 4–8 non-empty lines and contains the total 10-day margin, total 20-day margin, absolute difference, and the exact decision slug.

## Step 4 – Run the verifier if available

```bash
ls /root/test_output.py 2>/dev/null && python3 -m pytest /root/test_output.py -v
```

If any test fails, read the error message carefully, identify the specific field and expected value, fix the calculation in `solve.py`, and re-run. Pay particular attention to:
- Whether cooler cost formula is `cooler_cost_usd * active_sites * dispatches_per_year` (confirmed by previous feedback).
- Whether drug cost or revenue formulas match expectations.
- Any edge cases in site override resolution or label matching.

## Critical Notes from Previous Failure

The previous iteration failed because `annual_cooler_cost` was computed as `cooler_cost_usd * dispatches_per_year` WITHOUT multiplying by `active_sites`. The correct formula is:

```
annual_cooler_cost = cooler_cost_usd * active_sites * dispatches_per_year
```

This is analogous to the harbor_vaxcrate_6v12 task which had the identical failure pattern (468.0 vs 7956.0). Ensure this multiplication is present in both the 10-day and 20-day cooler cost calculations.

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