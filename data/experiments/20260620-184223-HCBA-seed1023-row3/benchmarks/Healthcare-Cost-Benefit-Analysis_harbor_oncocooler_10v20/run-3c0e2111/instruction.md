# Task Instruction

Execute the following steps in order:

## 1. Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

## 2. Create and run the solver script

Create `/root/solve.py` with the following logic:

```python
import json
import csv
import math

# Load program catalog
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

# Load cooler costs
cooler_costs = {}
with open('/root/cooler_cost.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cooler_costs[row['cooler_type'].strip()] = float(row['cooler_cost_usd'])

# Load contract payments
payments_raw = []
with open('/root/contract_payment.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        payments_raw.append(row)

# Load site overrides
overrides_raw = []
with open('/root/site_overrides.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        overrides_raw.append(row)

# Filter in-scope programs (review_flag == 'review')
# catalog could be a list or dict; inspect and handle
if isinstance(catalog, dict):
    programs_list = catalog.get('programs', [])
else:
    programs_list = catalog

in_scope = [p for p in programs_list if p.get('review_flag', '').strip().lower() == 'review']

# Build label-to-program mapping
label_to_program = {}
for p in in_scope:
    pname = p['program_name'].strip()
    label_to_program[pname.lower()] = p
    for lbl in p.get('known_labels', []):
        label_to_program[lbl.strip().lower()] = p

# Map contract payments to in-scope programs
payment_map = {}  # program_code -> payment_per_dispatch_per_site_usd
for row in payments_raw:
    pl = row['program_label'].strip().lower()
    if pl in label_to_program:
        prog = label_to_program[pl]
        pc = prog['program_code']
        payment_map[pc] = float(row['payment_per_dispatch_per_site_usd'])

# Resolve active sites from site_overrides
# Filter approved, then highest version_no per program_code
approved = [r for r in overrides_raw if r.get('approval_state', '').strip().lower() == 'approved']
best_override = {}
for r in approved:
    pc = r['program_code'].strip()
    vn = int(r['version_no'])
    if pc not in best_override or vn > best_override[pc]['version_no']:
        best_override[pc] = {'version_no': vn, 'active_sites': int(r['active_sites'])}

# Build results
results = []
for p in in_scope:
    pc = p['program_code'].strip()
    pname = p['program_name'].strip()
    
    # Active sites
    if pc in best_override:
        active_sites = best_override[pc]['active_sites']
    else:
        active_sites = int(p['default_active_sites'])
    
    acq_cost = float(p['acquisition_cost_per_1000_units_usd'])
    units_per_day = float(p['units_per_day'])
    cooler_type = p['cooler_type'].strip()
    cooler_cost = cooler_costs[cooler_type]
    payment = payment_map[pc]
    
    # 10-day model
    days_10 = 10
    disp_10 = 36
    # 20-day model
    days_20 = 20
    disp_20 = 18
    
    # Drug cost = acq_cost_per_1000 * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000
    drug_10 = acq_cost * active_sites * units_per_day * days_10 * disp_10 / 1000.0
    drug_20 = acq_cost * active_sites * units_per_day * days_20 * disp_20 / 1000.0
    
    # Cooler cost = cooler_cost_usd * dispatches_per_year (NOT per site - just per dispatch total)
    # Actually re-read: the formula says annual_cooler_cost but doesn't specify per-site.
    # The cooler cost is per dispatch. Annual cooler cost = cooler_cost * dispatches_per_year
    # But wait - is it per site? The problem says cooler_cost_usd from cooler_cost.csv.
    # Looking at the schema, there's no multiplication by sites mentioned for cooler cost.
    # Let me check: the formulas given are:
    #   annual_revenue = payment_per_dispatch_per_site * active_sites * dispatches_per_year
    #   annual_drug_cost = acq_cost_per_1000 * active_sites * units_per_day * days * dispatches / 1000
    #   annual_margin = revenue - drug_cost - cooler_cost
    # cooler_cost formula is NOT given explicitly. We need to figure it out.
    # Since revenue and drug cost are both multiplied by active_sites, and cooler is a physical
    # cooler sent per dispatch, it likely is: cooler_cost * dispatches_per_year * active_sites
    # But let me think about this more carefully. A cooler dispatch goes to each site.
    # So annual_cooler_cost = cooler_cost_usd * active_sites * dispatches_per_year
    # Actually, the problem says "cooler dispatches" - each site gets a cooler each dispatch.
    cooler_10 = cooler_cost * active_sites * disp_10
    cooler_20 = cooler_cost * active_sites * disp_20
    
    # Revenue
    rev_10 = payment * active_sites * disp_10
    rev_20 = payment * active_sites * disp_20
    
    # Margin
    margin_10 = rev_10 - drug_10 - cooler_10
    margin_20 = rev_20 - drug_20 - cooler_20
    diff = margin_20 - margin_10
    
    results.append({
        'program_code': pc,
        'program_name': pname,
        'active_sites': active_sites,
        'acquisition_cost_per_1000_units_usd': round(acq_cost, 2),
        'units_per_day': round(units_per_day, 2),
        'cooler_type': cooler_type,
        'cooler_cost_usd': round(cooler_cost, 2),
        'payment_per_dispatch_per_site_usd': round(payment, 2),
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
results.sort(key=lambda x: x['program_code'])

# Totals
total_margin_10 = sum(r['annual_margin_10_day_usd'] for r in results)
total_margin_20 = sum(r['annual_margin_20_day_usd'] for r in results)
total_diff = sum(r['annual_margin_difference_20_minus_10_usd'] for r in results)
abs_diff = abs(total_diff)

# Decision
if abs_diff < 10000:
    decision = 'move_to_20_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} is below the $10,000 threshold, so moving to 20-day dispatches is recommended.'
else:
    decision = 'keep_10_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} exceeds the $10,000 threshold, so keeping 10-day dispatches is recommended.'

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

print('JSON written.')
print(f'Total 10-day margin: ${total_margin_10:,.2f}')
print(f'Total 20-day margin: ${total_margin_20:,.2f}')
print(f'Absolute difference: ${abs_diff:,.2f}')
print(f'Decision: {decision}')

# Write summary
with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('# OncoCooler Dispatch Analysis Summary\n')
    f.write(f'Total 10-day annual margin: ${total_margin_10:,.2f}\n')
    f.write(f'Total 20-day annual margin: ${total_margin_20:,.2f}\n')
    f.write(f'Absolute margin difference: ${abs_diff:,.2f}\n')
    f.write(f'Recommendation: {decision}\n')
    f.write(f'{justification}\n')

print('Summary written.')
```

Run:
```bash
python3 /root/solve.py
```

## 3. Validate the outputs

```bash
cat /root/oncocooler_analysis.json
cat /root/oncocooler_summary.md
```

## 4. Cross-check drug cost logic

The previous feedback said drug cost had a factor-of-15 error (302.4 vs 4536.0). Note that 4536 / 302.4 = 15, and 15 = active_sites for that program (likely). This means the previous run was missing the `active_sites` multiplier in the drug cost formula. The formula above includes it:

`drug_cost = acq_cost_per_1000 * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000`

Verify by inspecting the first program's numbers manually. For example, if a program has acq_cost=X, units_per_day=Y, active_sites=Z, then:
- drug_10 = X * Z * Y * 10 * 36 / 1000
- drug_20 = X * Z * Y * 20 * 18 / 1000
- These should be EQUAL (both = X * Z * Y * 360 / 1000)

Confirm that drug_10 == drug_20 for each program in the JSON output.

## 5. If cooler cost logic seems wrong

The cooler cost might NOT be per-site. If the verifier fails, try the alternative: `cooler_cost * dispatches_per_year` (without active_sites). But first try with the per-site multiplication since cooler dispatches logically go to each site.

If the test runner exists, run it:
```bash
ls /root/test_output.py 2>/dev/null && cd /root && python3 -m pytest test_output.py -v
```

## 6. If tests fail, examine the error messages carefully

Look at which specific values are expected vs actual. Adjust the cooler cost formula if needed (per-site vs not-per-site). Re-run solve.py and re-verify.

## Key warnings from cross-task context:
- Do NOT add extra keys to the JSON (no 'warnings', 'model_constants', etc.) - match the schema exactly
- Use comma-formatted currency in the markdown summary (f'{value:,.2f}')
- Ensure JSON programs array is sorted by program_code ascending

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