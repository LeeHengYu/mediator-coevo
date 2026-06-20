# Task Instruction

Execute the following steps exactly:

1. **Read all input files** and print their contents so you can inspect the data:
```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

2. **Read the test file** to understand the verifier's exact expectations:
```bash
cat /root/test_output.py
```

3. **Write and run a Python script** `/root/solve.py` that does the following:

```python
import json
import csv
import math

# --- Load data ---
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

cooler_rows = read_csv('/root/cooler_cost.csv')
payment_rows = read_csv('/root/contract_payment.csv')
override_rows = read_csv('/root/site_overrides.csv')

# --- Step 1: Filter in-scope programs (review_flag == 'review') ---
in_scope = [p for p in catalog if p.get('review_flag') == 'review']

# --- Step 2: Build cooler cost lookup by cooler_type ---
cooler_lookup = {}
for row in cooler_rows:
    cooler_lookup[row['cooler_type'].strip()] = float(row['cooler_cost_usd'])

# --- Step 3: Build payment lookup ---
# Map program_label -> payment_per_dispatch_per_site_usd
# Match program_label to program_name or any entry in known_labels
payment_lookup = {}  # program_code -> payment
for prog in in_scope:
    code = prog['program_code']
    name = prog['program_name']
    labels = prog.get('known_labels', [])
    all_names = [name] + labels
    for pay_row in payment_rows:
        pl = pay_row['program_label'].strip()
        if pl in [n.strip() for n in all_names]:
            payment_lookup[code] = float(pay_row['payment_per_dispatch_per_site_usd'])
            break

# --- Step 4: Resolve active sites ---
# From site_overrides.csv: only approval_state == 'approved'
# If multiple approved rows for same program_code, keep highest version_no
# If no approved override, use default_active_sites from catalog
approved_overrides = [r for r in override_rows if r['approval_state'].strip().lower() == 'approved']

# Group by program_code, pick highest version_no
best_override = {}
for r in approved_overrides:
    pc = r['program_code'].strip()
    vn = int(r['version_no'])
    if pc not in best_override or vn > best_override[pc]['version_no']:
        best_override[pc] = {'version_no': vn, 'row': r}

def get_active_sites(prog):
    code = prog['program_code']
    if code in best_override:
        return int(best_override[code]['row']['active_sites'])
    return int(prog['default_active_sites'])

# --- Step 5: Compute per-program ---
programs_output = []
for prog in in_scope:
    code = prog['program_code']
    name = prog['program_name']
    active_sites = get_active_sites(prog)
    acq_cost = float(prog['acquisition_cost_per_1000_units_usd'])
    units_per_day = float(prog['units_per_day'])
    cooler_type = prog['cooler_type'].strip()
    cooler_cost = cooler_lookup[cooler_type]
    payment = payment_lookup[code]

    # 10-day model
    disp_10 = 36
    days_10 = 10
    annual_drug_10 = acq_cost * active_sites * units_per_day * days_10 * disp_10 / 1000
    annual_cooler_10 = cooler_cost * disp_10
    annual_rev_10 = payment * active_sites * disp_10
    annual_margin_10 = annual_rev_10 - annual_drug_10 - annual_cooler_10

    # 20-day model
    disp_20 = 18
    days_20 = 20
    annual_drug_20 = acq_cost * active_sites * units_per_day * days_20 * disp_20 / 1000
    annual_cooler_20 = cooler_cost * disp_20
    annual_rev_20 = payment * active_sites * disp_20
    annual_margin_20 = annual_rev_20 - annual_drug_20 - annual_cooler_20

    diff = annual_margin_20 - annual_margin_10

    programs_output.append({
        'program_code': code,
        'program_name': name,
        'active_sites': active_sites,
        'acquisition_cost_per_1000_units_usd': round(acq_cost, 2),
        'units_per_day': round(units_per_day, 2),
        'cooler_type': cooler_type,
        'cooler_cost_usd': round(cooler_cost, 2),
        'payment_per_dispatch_per_site_usd': round(payment, 2),
        'annual_drug_cost_10_day_usd': round(annual_drug_10, 2),
        'annual_drug_cost_20_day_usd': round(annual_drug_20, 2),
        'annual_cooler_cost_10_day_usd': round(annual_cooler_10, 2),
        'annual_cooler_cost_20_day_usd': round(annual_cooler_20, 2),
        'annual_revenue_10_day_usd': round(annual_rev_10, 2),
        'annual_revenue_20_day_usd': round(annual_rev_20, 2),
        'annual_margin_10_day_usd': round(annual_margin_10, 2),
        'annual_margin_20_day_usd': round(annual_margin_20, 2),
        'annual_margin_difference_20_minus_10_usd': round(diff, 2)
    })

# Sort by program_code ascending
programs_output.sort(key=lambda x: x['program_code'])

# --- Step 6: Totals ---
total_margin_10 = sum(p['annual_margin_10_day_usd'] for p in programs_output)
total_margin_20 = sum(p['annual_margin_20_day_usd'] for p in programs_output)
total_diff = total_margin_20 - total_margin_10
abs_diff = abs(total_diff)

# --- Step 7: Decision ---
if abs_diff < 10000:
    decision = 'move_to_20_day'
    justification = f'The absolute total margin difference is ${abs_diff:,.2f}, which is below the $10,000 threshold, so moving to 20-day dispatches is recommended.'
else:
    decision = 'keep_10_day'
    justification = f'The absolute total margin difference is ${abs_diff:,.2f}, which exceeds the $10,000 threshold, so keeping 10-day dispatches is recommended.'

# --- Step 8: Build output JSON ---
output = {
    'assumptions': {
        'dispatches_per_year_10_day': 36,
        'dispatches_per_year_20_day': 18,
        'days_per_dispatch_10_day': 10,
        'days_per_dispatch_20_day': 20,
        'switch_threshold_usd': 10000,
        'site_override_rule': 'highest approved version_no per program_code, else default_active_sites'
    },
    'programs': programs_output,
    'totals': {
        'total_annual_margin_10_day_usd': round(total_margin_10, 2),
        'total_annual_margin_20_day_usd': round(total_margin_20, 2),
        'total_annual_margin_difference_20_minus_10_usd': round(total_diff, 2),
        'absolute_total_margin_difference_usd': round(abs_diff, 2)
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/oncocooler_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

# --- Step 9: Build summary markdown ---
# Format numbers with commas and 2 decimal places
def fmt(v):
    return f'{v:,.2f}'

lines = [
    '# OncoCooler Dispatch Analysis Summary',
    '',
    f'Total 10-day annual margin: ${fmt(round(total_margin_10, 2))}',
    f'Total 20-day annual margin: ${fmt(round(total_margin_20, 2))}',
    f'Absolute margin difference: ${fmt(round(abs_diff, 2))}',
    f'Recommendation: {decision}',
    '',
    justification
]

with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Output files written.')
print(json.dumps(output, indent=2))
```

4. Run the script:
```bash
cd /root && python solve.py
```

5. **Verify outputs exist and look correct:**
```bash
cat /root/oncocooler_analysis.json
cat /root/oncocooler_summary.md
```

6. **Run the test suite** to confirm everything passes:
```bash
cd /root && python -m pytest test_output.py -v
```

7. **If tests fail**, carefully read the error messages. Common fixes:
   - If the test expects additional keys in `assumptions`, read the test file to see exactly what keys it checks and add them.
   - If numeric values are slightly off, check rounding — ensure you round the final computed value, not intermediate values.
   - If the summary format is wrong, check the test's regex or string matching and adjust formatting.
   - If `cooler_cost` is per-dispatch (not annual), re-read the cooler_cost.csv to confirm units and adjust the formula: `annual_cooler_cost = cooler_cost_usd * dispatches_per_year`.
   - Re-read the test file carefully for any additional `assumptions` keys it checks beyond the ones in the schema template.

8. After fixing any issues, re-run `python -m pytest test_output.py -v` until all tests pass.

**Critical reminders from prior failure:**
- Use FLAT keys in each program object (e.g., `annual_margin_10_day_usd`), NOT nested objects.
- The `assumptions` object must match the schema exactly as shown. If the test checks for additional keys, add them.
- In the markdown summary, format all USD values with thousands separators (commas) and 2 decimal places (e.g., `$-79,751.70` not `$-79751.70`).
- The `recommendation` key must exist at the top level of the JSON with `decision` and `justification` sub-keys.
- Sort programs by `program_code` ascending.

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