# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

## Step 2: Build and run the analysis script

Create `/root/solve.py` with the following logic:

```python
import json, csv, math
from decimal import Decimal, ROUND_HALF_UP

# Load program catalog
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

# Identify in-scope programs (review_flag == 'review')
in_scope = [p for p in catalog['programs'] if p.get('review_flag') == 'review']

# Load cooler_cost.csv into dict keyed by cooler_type
with open('/root/cooler_cost.csv') as f:
    reader = csv.DictReader(f)
    cooler_costs = {row['cooler_type']: float(row['cooler_cost_usd']) for row in reader}

# Load contract_payment.csv
with open('/root/contract_payment.csv') as f:
    reader = csv.DictReader(f)
    payment_rows = list(reader)

# Build label-to-program_code mapping
# A payment row's program_label matches if it equals program_name or any entry in known_labels
label_to_program = {}
for p in in_scope:
    label_to_program[p['program_name']] = p['program_code']
    for lbl in p.get('known_labels', []):
        label_to_program[lbl] = p['program_code']

# Resolve payments: map each payment row to a program_code; ignore unmapped
payment_by_code = {}
for row in payment_rows:
    plabel = row['program_label']
    if plabel in label_to_program:
        code = label_to_program[plabel]
        payment_by_code[code] = float(row['payment_per_dispatch_per_site_usd'])

# Load site_overrides.csv
with open('/root/site_overrides.csv') as f:
    reader = csv.DictReader(f)
    override_rows = list(reader)

# For each program_code, find approved rows, pick highest version_no
override_sites = {}
for row in override_rows:
    if row['approval_state'] == 'approved':
        code = row['program_code']
        ver = int(row['version_no'])
        if code not in override_sites or ver > override_sites[code][0]:
            override_sites[code] = (ver, int(row['active_sites']))

# Build program-level dict for easy lookup
program_map = {p['program_code']: p for p in in_scope}

# Compute per-program
programs_output = []
for p in sorted(in_scope, key=lambda x: x['program_code']):
    code = p['program_code']
    name = p['program_name']
    
    # Active sites
    if code in override_sites:
        active_sites = override_sites[code][1]
    else:
        active_sites = p['default_active_sites']
    
    acq_cost = float(p['acquisition_cost_per_1000_units_usd'])
    units_day = float(p['units_per_day'])
    cooler_type = p['cooler_type']
    cooler_cost = cooler_costs[cooler_type]
    payment = payment_by_code[code]
    
    # 10-day model
    disp_10 = 36
    days_10 = 10
    annual_drug_10 = acq_cost * active_sites * units_day * days_10 * disp_10 / 1000
    annual_cooler_10 = cooler_cost * disp_10  # cooler cost per dispatch * dispatches
    annual_rev_10 = payment * active_sites * disp_10
    annual_margin_10 = annual_rev_10 - annual_drug_10 - annual_cooler_10
    
    # 20-day model
    disp_20 = 18
    days_20 = 20
    annual_drug_20 = acq_cost * active_sites * units_day * days_20 * disp_20 / 1000
    annual_cooler_20 = cooler_cost * disp_20
    annual_rev_20 = payment * active_sites * disp_20
    annual_margin_20 = annual_rev_20 - annual_drug_20 - annual_cooler_20
    
    diff = annual_margin_20 - annual_margin_10
    
    programs_output.append({
        'program_code': code,
        'program_name': name,
        'active_sites': active_sites,
        'acquisition_cost_per_1000_units_usd': round(acq_cost, 2),
        'units_per_day': round(units_day, 2),
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

total_margin_10 = sum(p['annual_margin_10_day_usd'] for p in programs_output)
total_margin_20 = sum(p['annual_margin_20_day_usd'] for p in programs_output)
total_diff = round(total_margin_20 - total_margin_10, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 10000:
    decision = 'move_to_20_day'
    justification = f'The absolute total margin difference is ${abs_diff}, which is below the $10,000 threshold, so moving to 20-day dispatches is recommended.'
else:
    decision = 'keep_10_day'
    justification = f'The absolute total margin difference is ${abs_diff}, which meets or exceeds the $10,000 threshold, so keeping 10-day dispatches is recommended.'

result = {
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

# Write summary
lines = [
    '# OncoCooler Dispatch Analysis Summary',
    '',
    f'Total 10-day annual margin: ${round(total_margin_10, 2):,.2f} USD',
    f'Total 20-day annual margin: ${round(total_margin_20, 2):,.2f} USD',
    f'Absolute margin difference: ${abs_diff:,.2f} USD',
    f'Recommendation: {decision}',
    '',
    justification
]

with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Summary written.')
```

Run it:
```bash
python3 /root/solve.py
```

## Step 3: Validate outputs

```bash
# Verify JSON is valid and has required keys
python3 -c "
import json
with open('/root/oncocooler_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'programs' in d and len(d['programs']) > 0
assert 'totals' in d
assert 'recommendation' in d
assert d['recommendation']['decision'] in ('move_to_20_day', 'keep_10_day')
for p in d['programs']:
    for k in ['program_code','active_sites','annual_margin_10_day_usd','annual_margin_20_day_usd','annual_margin_difference_20_minus_10_usd']:
        assert k in p, f'Missing {k} in program {p}'
# Check programs sorted by program_code ascending
codes = [p['program_code'] for p in d['programs']]
assert codes == sorted(codes), f'Programs not sorted: {codes}'
print('JSON validation passed.')
print(f'Programs: {len(d[\"programs\"])}  Decision: {d[\"recommendation\"][\"decision\"]}')
"

# Verify summary
python3 -c "
with open('/root/oncocooler_summary.md') as f:
    lines = [l for l in f.read().strip().split('\n') if l.strip()]
print(f'Non-empty lines: {len(lines)}')
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
content = open('/root/oncocooler_summary.md').read()
assert 'move_to_20_day' in content or 'keep_10_day' in content, 'Missing decision slug'
print('Summary validation passed.')
"

cat /root/oncocooler_analysis.json
cat /root/oncocooler_summary.md
```

## Important Notes

- **Cooler cost**: The `annual_cooler_cost` is `cooler_cost_usd * dispatches_per_year`. It is NOT multiplied by active_sites — cooler cost is per dispatch, not per site. HOWEVER, re-read the instructions carefully: the formulas do not explicitly state annual_cooler_cost. The margin formula is `annual_revenue - annual_drug_cost - annual_cooler_cost`. Since the instructions say cooler cost uses `cooler_cost_usd` from `cooler_cost.csv` matched by `cooler_type`, and no formula multiplies it by sites, treat it as `cooler_cost_usd * dispatches_per_year` (one cooler per dispatch total). If after inspecting the data this produces unreasonable numbers (e.g., cooler cost is trivially small compared to drug/revenue), reconsider whether cooler cost should be per-site too, but start with the literal reading.

- **Payment resolution**: A payment row matches if its `program_label` equals the catalog entry's `program_name` OR any string in `known_labels`. If multiple payment rows map to the same program, check the data — there should typically be one per program. If duplicates exist, flag it.

- **Rounding**: Round all currency outputs to 2 decimal places at the output stage. Use Python floats with `round(..., 2)`.

- After Step 1 (inspecting files), if the data reveals that `cooler_cost_usd` should be multiplied by `active_sites` (e.g., if the schema or comments suggest per-site cooler costs), adjust the formula accordingly before running. The key is to match what the verifier expects.

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