# Task Instruction

Execute the following steps to produce the two required output files.

## Step 1: Inspect input files

```bash
cat /root/therapy_catalog.json
cat /root/bag_supply_cost.csv
cat /root/delivery_payment.csv
cat /root/patient_overrides.csv
```

## Step 2: Create and run the solver script

Create `/root/solve.py` with the following logic:

```python
import json, csv, math

# ── Load inputs ──
with open('/root/therapy_catalog.json') as f:
    catalog = json.load(f)

with open('/root/bag_supply_cost.csv') as f:
    bag_costs = list(csv.DictReader(f))

with open('/root/delivery_payment.csv') as f:
    payments = list(csv.DictReader(f))

with open('/root/patient_overrides.csv') as f:
    overrides = list(csv.DictReader(f))

# ── Filter in-scope therapies ──
in_scope = [t for t in catalog if t.get('include_in_review') == True]

# ── Build lookup: therapy_label -> therapy_code (using therapy_name + aliases) ──
label_to_code = {}
for t in in_scope:
    label_to_code[t['therapy_name'].strip().lower()] = t['therapy_code']
    for alias in t.get('aliases', []):
        label_to_code[alias.strip().lower()] = t['therapy_code']

# ── Build therapy dict keyed by therapy_code ──
therapy_map = {t['therapy_code']: t for t in in_scope}

# ── Resolve delivery payments ──
payment_by_code = {}
for row in payments:
    label = row['therapy_label'].strip().lower()
    code = label_to_code.get(label)
    if code is None:
        continue
    payment_by_code[code] = float(row['payment_per_delivery_per_patient_usd'])

# ── Bag supply cost lookup by bag_size_ml ──
bag_cost_map = {}
for row in bag_costs:
    bag_cost_map[int(row['bag_size_ml'])] = float(row['bag_supply_cost_usd'])

# ── Patient overrides: approved, highest revision per therapy_code ──
approved = [r for r in overrides if r['status'].strip().lower() == 'approved']
best_override = {}
for r in approved:
    code = r['therapy_code'].strip()
    if code not in therapy_map:
        continue
    rev = int(r['revision'])
    if code not in best_override or rev > best_override[code]['rev']:
        best_override[code] = {'rev': rev, 'patients': int(r['active_patients'])}

active_patients_map = {code: v['patients'] for code, v in best_override.items()}

# ── Constants ──
DEL7, DEL14 = 52, 26
DAYS7, DAYS14 = 7, 14
THRESHOLD = 15000

# ── Compute per-therapy ──
therapies_out = []
for code in sorted(therapy_map.keys()):
    t = therapy_map[code]
    active = active_patients_map.get(code, 0)
    drug_cost_per_1000 = float(t['drug_cost_per_1000_mg_usd'])
    dose_mg = float(t['dose_mg_per_day'])
    bag_ml = int(t['bag_size_ml'])
    bag_cost = bag_cost_map.get(bag_ml, 0.0)
    pay = payment_by_code.get(code, 0.0)

    # Annual drug cost
    adc7 = drug_cost_per_1000 * active * dose_mg * DAYS7 * DEL7 / 1000
    adc14 = drug_cost_per_1000 * active * dose_mg * DAYS14 * DEL14 / 1000

    # Annual supply cost: bags_per_delivery=1 per patient per delivery
    # Each delivery uses 1 bag per patient (days_per_delivery bags worth)
    # The instructions say bag_supply_cost_usd from bag_supply_cost.csv
    # Supply cost = bag_supply_cost_usd * active_patients * deliveries_per_year
    asc7 = bag_cost * active * DEL7
    asc14 = bag_cost * active * DEL14

    # Annual revenue
    ar7 = pay * active * DEL7
    ar14 = pay * active * DEL14

    # Annual margin
    am7 = round(ar7 - adc7 - asc7, 2)
    am14 = round(ar14 - adc14 - asc14, 2)
    diff = round(am14 - am7, 2)

    therapies_out.append({
        'therapy_code': code,
        'therapy_name': t['therapy_name'],
        'active_patients': active,
        'drug_cost_per_1000_mg_usd': round(drug_cost_per_1000, 2),
        'dose_mg_per_day': round(dose_mg, 2),
        'bag_size_ml': bag_ml,
        'bag_supply_cost_usd': round(bag_cost, 2),
        'payment_per_delivery_per_patient_usd': round(pay, 2),
        'annual_drug_cost_7_day_usd': round(adc7, 2),
        'annual_drug_cost_14_day_usd': round(adc14, 2),
        'annual_supply_cost_7_day_usd': round(asc7, 2),
        'annual_supply_cost_14_day_usd': round(asc14, 2),
        'annual_revenue_7_day_usd': round(ar7, 2),
        'annual_revenue_14_day_usd': round(ar14, 2),
        'annual_margin_7_day_usd': am7,
        'annual_margin_14_day_usd': am14,
        'annual_margin_difference_14_minus_7_usd': diff
    })

total_m7 = round(sum(t['annual_margin_7_day_usd'] for t in therapies_out), 2)
total_m14 = round(sum(t['annual_margin_14_day_usd'] for t in therapies_out), 2)
total_diff = round(total_m14 - total_m7, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < THRESHOLD:
    decision = 'move_to_14_day'
else:
    decision = 'keep_7_day'

justification = (f"The absolute margin difference of ${abs_diff:,.2f} is "
                 f"{'below' if abs_diff < THRESHOLD else 'at or above'} "
                 f"the ${THRESHOLD:,.2f} threshold, so the recommendation is {decision}.")

result = {
    'assumptions': {
        'deliveries_per_year_7_day': DEL7,
        'deliveries_per_year_14_day': DEL14,
        'days_per_delivery_7_day': DAYS7,
        'days_per_delivery_14_day': DAYS14,
        'switch_threshold_usd': THRESHOLD,
        'patient_override_rule': 'highest approved revision per therapy_code'
    },
    'therapies': therapies_out,
    'totals': {
        'total_annual_margin_7_day_usd': total_m7,
        'total_annual_margin_14_day_usd': total_m14,
        'total_annual_margin_difference_14_minus_7_usd': total_diff,
        'absolute_total_margin_difference_usd': abs_diff
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/infusion_batch_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

# ── Markdown summary (use commas as thousands separators!) ──
lines = [
    '# Infusion Batch Analysis Summary',
    '',
    f'- Total 7-day annual margin: ${total_m7:,.2f} USD',
    f'- Total 14-day annual margin: ${total_m14:,.2f} USD',
    f'- Absolute margin difference: ${abs_diff:,.2f} USD',
    f'- Final decision: {decision}',
]

with open('/root/infusion_batch_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done.')
print(f'Total 7-day margin: {total_m7}')
print(f'Total 14-day margin: {total_m14}')
print(f'Difference: {total_diff}')
print(f'Abs diff: {abs_diff}')
print(f'Decision: {decision}')
```

Run:
```bash
python3 /root/solve.py
```

## Step 3: Validate outputs

```bash
cat /root/infusion_batch_analysis.json
cat /root/infusion_batch_summary.md
```

Verify:
1. The JSON is valid and contains all required keys: `assumptions`, `therapies` (sorted by therapy_code), `totals`, `recommendation` (with `decision` and `justification`).
2. All currency values are rounded to 2 decimals.
3. The markdown has 4-8 non-empty lines.
4. Currency values in the markdown use commas as thousands separators (e.g., `-455,619.31` not `-455619.31`).
5. The decision slug is exactly `move_to_14_day` or `keep_7_day`.
6. The `therapies` array is sorted by `therapy_code` ascending.

If any check fails, inspect the input data again and fix the script accordingly before re-running.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[home-infusion, json, csv, alias-resolution, decision-analysis].
Verifier config: timeout_sec=900.0.