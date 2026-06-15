# Task Instruction

Execute the following steps in order:

## 1. Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

## 2. Build and run the analysis script

Create `/root/solve.py` with the following logic:

```python
import json, csv, math
from decimal import Decimal, ROUND_HALF_UP

# ---- Load inputs ----
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

cooler_rows = read_csv('/root/cooler_cost.csv')
payment_rows = read_csv('/root/contract_payment.csv')
override_rows = read_csv('/root/site_overrides.csv')

# ---- Step A: Filter in-scope programs (review_flag == 'review') ----
programs_in_scope = [p for p in catalog if p.get('review_flag') == 'review']

# ---- Step B: Build label-to-program mapping ----
# Map program_name and each known_labels entry to the program dict
label_map = {}
for p in programs_in_scope:
    label_map[p['program_name']] = p
    for lbl in p.get('known_labels', []):
        label_map[lbl] = p

# ---- Step C: Resolve contract payments ----
# For each in-scope program, find its payment_per_dispatch_per_site_usd
payment_by_code = {}
for row in payment_rows:
    pl = row['program_label']
    if pl in label_map:
        prog = label_map[pl]
        pc = prog['program_code']
        payment_by_code[pc] = float(row['payment_per_dispatch_per_site_usd'])

# ---- Step D: Resolve active sites from site_overrides ----
# Only approved rows; keep highest version_no per program_code
approved = [r for r in override_rows if r.get('approval_state') == 'approved']
best_override = {}
for r in approved:
    pc = r['program_code']
    vn = int(r['version_no'])
    if pc not in best_override or vn > best_override[pc]['version_no']:
        best_override[pc] = {'version_no': vn, 'active_sites': int(r['active_sites'])}

# ---- Step E: Resolve cooler costs ----
cooler_cost_map = {}
for r in cooler_rows:
    cooler_cost_map[r['cooler_type']] = float(r['cooler_cost_usd'])

# ---- Step F: Compute per-program figures ----
results = []
for p in programs_in_scope:
    pc = p['program_code']
    pname = p['program_name']
    
    # Active sites
    if pc in best_override:
        active_sites = best_override[pc]['active_sites']
    else:
        active_sites = int(p['default_active_sites'])
    
    acq_cost = float(p['acquisition_cost_per_1000_units_usd'])
    units_per_day = float(p['units_per_day'])
    cooler_type = p['cooler_type']
    cooler_cost = cooler_cost_map[cooler_type]
    payment = payment_by_code[pc]
    
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
    
    results.append({
        'program_code': pc,
        'program_name': pname,
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
results.sort(key=lambda x: x['program_code'])

# ---- Step G: Totals ----
total_margin_10 = round(sum(r['annual_margin_10_day_usd'] for r in results), 2)
total_margin_20 = round(sum(r['annual_margin_20_day_usd'] for r in results), 2)
total_diff = round(total_margin_20 - total_margin_10, 2)
abs_diff = round(abs(total_diff), 2)

# ---- Step H: Decision ----
if abs_diff < 10000:
    decision = 'move_to_20_day'
    justification = f'The absolute total margin difference of ${abs_diff} is below the $10,000 threshold, so switching to 20-day dispatches is recommended.'
else:
    decision = 'keep_10_day'
    justification = f'The absolute total margin difference of ${abs_diff} exceeds the $10,000 threshold, so keeping 10-day dispatches is recommended.'

# ---- Build output JSON ----
output = {
    'assumptions': {
        'dispatches_per_year_10_day': 36,
        'dispatches_per_year_20_day': 18,
        'days_per_dispatch_10_day': 10,
        'days_per_dispatch_20_day': 20,
        'switch_threshold_usd': 10000,
        'site_override_rule': 'highest approved version_no per program_code, else default_active_sites'
    },
    'programs': results,
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
    json.dump(output, f, indent=2)

print('JSON written.')
print(json.dumps(output, indent=2))

# ---- Build summary markdown ----
lines = [
    '# OncoCooler Dispatch Analysis Summary',
    f'Total 10-day annual margin: ${total_margin_10:,.2f} USD',
    f'Total 20-day annual margin: ${total_margin_20:,.2f} USD',
    f'Absolute margin difference: ${abs_diff:,.2f} USD',
    f'Recommendation: {decision}',
    f'{justification}'
]

with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Summary written.')
```

Run it:
```bash
python3 /root/solve.py
```

## 3. Validate outputs

```bash
echo '--- JSON output ---'
cat /root/oncocooler_analysis.json
echo ''
echo '--- Summary output ---'
cat /root/oncocooler_summary.md
```

Verify:
- JSON is valid and parseable (use `python3 -c "import json; json.load(open('/root/oncocooler_analysis.json'))"`).
- `programs` array is sorted by `program_code` ascending.
- All currency values have exactly 2 decimal places.
- Summary has 4-8 non-empty lines.
- Summary contains total 10-day margin, total 20-day margin, absolute difference, and the exact decision slug (`move_to_20_day` or `keep_10_day`).
- The `annual_cooler_cost` is `cooler_cost_usd * dispatches_per_year` (NOT multiplied by active_sites — the cooler cost is per-dispatch for the whole program, not per-site). **IMPORTANT**: Re-read the task instructions carefully. The formula says `annual_cooler_cost` but does NOT specify multiplication by `active_sites`. The drug cost and revenue formulas explicitly include `active_sites`, but cooler cost does not. So cooler cost = `cooler_cost_usd * dispatches_per_year` only.

If any validation fails, fix and re-run.

## 4. Double-check the cooler cost interpretation

After seeing the data, reconsider: look at `cooler_cost.csv` column names and values. The task says:
- "Cooler cost uses `cooler_cost_usd` from `cooler_cost.csv`, matched by `cooler_type`."
- There is NO explicit annual cooler cost formula given (unlike drug cost and revenue which have explicit formulas).
- The annual margin formula is: `annual_revenue - annual_drug_cost - annual_cooler_cost`

Since there's no explicit formula for annual_cooler_cost, it's likely: `cooler_cost_usd * dispatches_per_year` (one cooler per dispatch for the program). But if the data or verifier expects it per-site, it would be `cooler_cost_usd * active_sites * dispatches_per_year`. 

Check the JSON output fields: there are `annual_cooler_cost_10_day_usd` and `annual_cooler_cost_20_day_usd`. The most natural reading is one cooler per dispatch (not per site), since the task doesn't mention sites for cooler cost. Keep it as `cooler_cost_usd * dispatches_per_year` unless evidence suggests otherwise.

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