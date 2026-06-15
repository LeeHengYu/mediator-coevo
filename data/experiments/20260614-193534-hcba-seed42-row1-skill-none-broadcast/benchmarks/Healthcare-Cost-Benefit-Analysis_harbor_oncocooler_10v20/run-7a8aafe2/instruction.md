# Task Instruction

Execute the following steps exactly:

1. Read all four input files:
   - `/root/program_catalog.json`
   - `/root/cooler_cost.csv`
   - `/root/contract_payment.csv`
   - `/root/site_overrides.csv`

   Print their contents so you can inspect them before writing any code.

2. Write a Python script `/root/solve.py` that does the following:

```python
import json, csv, math

# Load program catalog
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

# Load cooler costs
cooler_costs = {}
with open('/root/cooler_cost.csv') as f:
    for row in csv.DictReader(f):
        cooler_costs[row['cooler_type'].strip()] = float(row['cooler_cost_usd'])

# Load contract payments
payments_raw = []
with open('/root/contract_payment.csv') as f:
    for row in csv.DictReader(f):
        payments_raw.append(row)

# Load site overrides
overrides_raw = []
with open('/root/site_overrides.csv') as f:
    for row in csv.DictReader(f):
        overrides_raw.append(row)

# Step 1: Filter in-scope programs (review_flag == 'review')
in_scope = [p for p in catalog['programs'] if p.get('review_flag') == 'review']

# Step 2: Build label-to-program mapping for contract_payment resolution
label_to_program = {}
for p in in_scope:
    label_to_program[p['program_name'].strip().lower()] = p['program_code']
    for lbl in p.get('known_labels', []):
        label_to_program[lbl.strip().lower()] = p['program_code']

# Step 3: Resolve payment_per_dispatch_per_site_usd per program_code
payment_by_code = {}
for row in payments_raw:
    pl = row['program_label'].strip().lower()
    if pl in label_to_program:
        code = label_to_program[pl]
        payment_by_code[code] = float(row['payment_per_dispatch_per_site_usd'])

# Step 4: Resolve active_sites per program_code from site_overrides
# Only approved rows; keep highest version_no per program_code
approved = [r for r in overrides_raw if r['approval_state'].strip().lower() == 'approved']
best_override = {}
for r in approved:
    code = r['program_code'].strip()
    ver = int(r['version_no'])
    if code not in best_override or ver > best_override[code]['version_no']:
        best_override[code] = {'version_no': ver, 'active_sites': int(r['active_sites'])}

# Step 5: Build per-program analysis
programs_out = []
for p in in_scope:
    code = p['program_code']
    name = p['program_name']
    
    # Active sites
    if code in best_override:
        active_sites = best_override[code]['active_sites']
    else:
        active_sites = int(p['default_active_sites'])
    
    acq_cost = float(p['acquisition_cost_per_1000_units_usd'])
    units_per_day = float(p['units_per_day'])
    cooler_type = p['cooler_type'].strip()
    cooler_cost = cooler_costs[cooler_type]
    payment = payment_by_code[code]
    
    # 10-day model
    dispatches_10 = 36
    days_10 = 10
    annual_drug_cost_10 = acq_cost * active_sites * units_per_day * days_10 * dispatches_10 / 1000.0
    annual_cooler_cost_10 = cooler_cost * active_sites * dispatches_10
    annual_revenue_10 = payment * active_sites * dispatches_10
    annual_margin_10 = annual_revenue_10 - annual_drug_cost_10 - annual_cooler_cost_10
    
    # 20-day model
    dispatches_20 = 18
    days_20 = 20
    annual_drug_cost_20 = acq_cost * active_sites * units_per_day * days_20 * dispatches_20 / 1000.0
    annual_cooler_cost_20 = cooler_cost * active_sites * dispatches_20
    annual_revenue_20 = payment * active_sites * dispatches_20
    annual_margin_20 = annual_revenue_20 - annual_drug_cost_20 - annual_cooler_cost_20
    
    diff = annual_margin_20 - annual_margin_10
    
    programs_out.append({
        'program_code': code,
        'program_name': name,
        'active_sites': active_sites,
        'acquisition_cost_per_1000_units_usd': round(acq_cost, 2),
        'units_per_day': round(units_per_day, 2),
        'cooler_type': cooler_type,
        'cooler_cost_usd': round(cooler_cost, 2),
        'payment_per_dispatch_per_site_usd': round(payment, 2),
        'annual_drug_cost_10_day_usd': round(annual_drug_cost_10, 2),
        'annual_drug_cost_20_day_usd': round(annual_drug_cost_20, 2),
        'annual_cooler_cost_10_day_usd': round(annual_cooler_cost_10, 2),
        'annual_cooler_cost_20_day_usd': round(annual_cooler_cost_20, 2),
        'annual_revenue_10_day_usd': round(annual_revenue_10, 2),
        'annual_revenue_20_day_usd': round(annual_revenue_20, 2),
        'annual_margin_10_day_usd': round(annual_margin_10, 2),
        'annual_margin_20_day_usd': round(annual_margin_20, 2),
        'annual_margin_difference_20_minus_10_usd': round(diff, 2)
    })

# Sort by program_code ascending
programs_out.sort(key=lambda x: x['program_code'])

# Step 6: Totals
total_margin_10 = round(sum(p['annual_margin_10_day_usd'] for p in programs_out), 2)
total_margin_20 = round(sum(p['annual_margin_20_day_usd'] for p in programs_out), 2)
total_diff = round(total_margin_20 - total_margin_10, 2)
abs_diff = round(abs(total_diff), 2)

# Step 7: Decision
if abs_diff < 10000:
    decision = 'move_to_20_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} is below the $10,000 threshold, so moving to 20-day dispatches is recommended.'
else:
    decision = 'keep_10_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} exceeds the $10,000 threshold, so keeping 10-day dispatches is recommended.'

result = {
    'assumptions': {
        'dispatches_per_year_10_day': 36,
        'dispatches_per_year_20_day': 18,
        'days_per_dispatch_10_day': 10,
        'days_per_dispatch_20_day': 20,
        'switch_threshold_usd': 10000,
        'site_override_rule': 'highest approved version_no per program_code, else default_active_sites'
    },
    'programs': programs_out,
    'totals': {
        'total_annual_margin_10_day_usd': total_margin_10,
        'total_annual_margin_20_day_usd': total_margin_20,
        'total_annual_margin_difference_20_minus_10_usd': total_diff,
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
print(json.dumps(result, indent=2))

# Step 8: Write summary
lines = [
    '# OncoCooler Dispatch Analysis Summary',
    '',
    f'Total 10-day annual margin: ${total_margin_10:,.2f}',
    f'Total 20-day annual margin: ${total_margin_20:,.2f}',
    f'Absolute margin difference: ${abs_diff:,.2f}',
    f'Recommendation: {decision}',
    '',
    justification
]

with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Summary written.')
```

3. Run `python /root/solve.py` and inspect the output.

4. Verify the outputs:
   - `cat /root/oncocooler_analysis.json` — check it parses, has correct schema keys including `switch_threshold_usd` in assumptions, programs sorted by program_code, all currency values rounded to 2 decimals.
   - `cat /root/oncocooler_summary.md` — check it has 4-8 non-empty lines, includes comma-formatted currency values, and contains the exact decision slug.
   - Verify cooler costs are multiplied by `active_sites * dispatches_per_year` (this was the bug in the previous run).
   - Verify drug costs use the formula: `acq_cost * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000`.

5. If there is a test file at `/root/tests/test_outputs.py` or similar, run `cd /root && python -m pytest tests/ -v` to validate.

6. If any test fails, read the error carefully, fix the issue in solve.py, re-run, and re-check.

Key points from previous failure to avoid:
- Annual cooler cost MUST be `cooler_cost_usd * active_sites * dispatches_per_year` (NOT just cooler_cost * dispatches).
- Use `switch_threshold_usd` as the key name in assumptions (NOT `decision_threshold_usd`).
- Format currency in the summary with commas (e.g., `-42,908.83` not `-42908.83`).

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