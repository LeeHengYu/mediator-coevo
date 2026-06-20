# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

## Step 2: Build and run a Python script that produces both output files

Create `/root/solve.py` with the following logic:

```python
import json, csv, math

# Load inputs
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

cooler_rows = read_csv('/root/cooler_cost.csv')
payment_rows = read_csv('/root/contract_payment.csv')
override_rows = read_csv('/root/site_overrides.csv')

# --- 1. Filter in-scope programs (review_flag == 'review') ---
programs_in_scope = [p for p in catalog if p.get('review_flag') == 'review']

# --- 2. Build cooler cost lookup by cooler_type ---
cooler_lookup = {}
for r in cooler_rows:
    cooler_lookup[r['cooler_type'].strip()] = float(r['cooler_cost_usd'])

# --- 3. Build payment lookup: map program_label -> payment row ---
# We need to resolve each payment row's program_label to an in-scope program.
# A payment row matches a program if program_label == program_name OR program_label is in known_labels.

def build_label_to_program(programs):
    mapping = {}
    for p in programs:
        pname = p['program_name'].strip()
        mapping[pname] = p
        for lbl in p.get('known_labels', []):
            mapping[lbl.strip()] = p
    return mapping

label_map = build_label_to_program(programs_in_scope)

# For each in-scope program, find its payment_per_dispatch_per_site_usd
payment_by_code = {}
for r in payment_rows:
    plabel = r['program_label'].strip()
    if plabel in label_map:
        prog = label_map[plabel]
        code = prog['program_code']
        payment_by_code[code] = float(r['payment_per_dispatch_per_site_usd'])

# --- 4. Resolve active sites from site_overrides.csv ---
# Only approved rows; for each program_code keep highest version_no
approved = [r for r in override_rows if r['approval_state'].strip() == 'approved']
best_override = {}
for r in approved:
    pc = r['program_code'].strip()
    vno = int(r['version_no'])
    if pc not in best_override or vno > best_override[pc]['version_no']:
        best_override[pc] = {'version_no': vno, 'row': r}

def get_active_sites(prog):
    pc = prog['program_code'].strip()
    if pc in best_override:
        return int(best_override[pc]['row']['active_sites'])
    return int(prog['default_active_sites'])

# --- 5. Compute per-program metrics ---
results = []
for prog in programs_in_scope:
    code = prog['program_code'].strip()
    name = prog['program_name'].strip()
    active_sites = get_active_sites(prog)
    acq_cost = float(prog['acquisition_cost_per_1000_units_usd'])
    upd = float(prog['units_per_day'])
    ct = prog['cooler_type'].strip()
    cooler_cost = cooler_lookup[ct]
    payment = payment_by_code[code]

    # 10-day model
    disp10 = 36; days10 = 10
    rev10 = payment * active_sites * disp10
    drug10 = acq_cost * active_sites * upd * days10 * disp10 / 1000
    cooler10 = cooler_cost * disp10
    margin10 = rev10 - drug10 - cooler10

    # 20-day model
    disp20 = 18; days20 = 20
    rev20 = payment * active_sites * disp20
    drug20 = acq_cost * active_sites * upd * days20 * disp20 / 1000
    cooler20 = cooler_cost * disp20
    margin20 = rev20 - drug20 - cooler20

    diff = margin20 - margin10

    results.append({
        'program_code': code,
        'program_name': name,
        'active_sites': active_sites,
        'acquisition_cost_per_1000_units_usd': round(acq_cost, 2),
        'units_per_day': round(upd, 2),
        'cooler_type': ct,
        'cooler_cost_usd': round(cooler_cost, 2),
        'payment_per_dispatch_per_site_usd': round(payment, 2),
        'annual_drug_cost_10_day_usd': round(drug10, 2),
        'annual_drug_cost_20_day_usd': round(drug20, 2),
        'annual_cooler_cost_10_day_usd': round(cooler10, 2),
        'annual_cooler_cost_20_day_usd': round(cooler20, 2),
        'annual_revenue_10_day_usd': round(rev10, 2),
        'annual_revenue_20_day_usd': round(rev20, 2),
        'annual_margin_10_day_usd': round(margin10, 2),
        'annual_margin_20_day_usd': round(margin20, 2),
        'annual_margin_difference_20_minus_10_usd': round(diff, 2)
    })

# Sort by program_code ascending
results.sort(key=lambda x: x['program_code'])

# --- 6. Totals ---
total_m10 = round(sum(r['annual_margin_10_day_usd'] for r in results), 2)
total_m20 = round(sum(r['annual_margin_20_day_usd'] for r in results), 2)
total_diff = round(total_m20 - total_m10, 2)
abs_diff = round(abs(total_diff), 2)

# --- 7. Decision ---
if abs_diff < 10000:
    decision = 'move_to_20_day'
    justification = f'Absolute total margin difference ${abs_diff} is below the $10,000 threshold, so switching to 20-day dispatches is recommended.'
else:
    decision = 'keep_10_day'
    justification = f'Absolute total margin difference ${abs_diff} exceeds the $10,000 threshold, so keeping 10-day dispatches is recommended.'

# --- 8. Write JSON ---
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
        'total_annual_margin_10_day_usd': total_m10,
        'total_annual_margin_20_day_usd': total_m20,
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
print(f'Total 10-day margin: {total_m10}')
print(f'Total 20-day margin: {total_m20}')
print(f'Total difference: {total_diff}')
print(f'Absolute difference: {abs_diff}')
print(f'Decision: {decision}')

# --- 9. Write Markdown summary ---
md_lines = [
    '# OncoCooler Dispatch Analysis Summary',
    '',
    f'Total 10-day annual margin: ${total_m10:,.2f} USD',
    f'Total 20-day annual margin: ${total_m20:,.2f} USD',
    f'Absolute margin difference: ${abs_diff:,.2f} USD',
    f'Recommendation: {decision}',
    '',
    justification
]

with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('\n'.join(md_lines) + '\n')

print('Markdown written.')
```

Run the script:
```bash
python3 /root/solve.py
```

## Step 3: Validate outputs

```bash
# Check JSON is valid and has required keys
python3 -c "
import json
with open('/root/oncocooler_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'programs' in d and len(d['programs']) > 0
assert 'totals' in d
assert 'recommendation' in d
assert d['recommendation']['decision'] in ('move_to_20_day', 'keep_10_day')
# Check programs sorted by program_code
codes = [p['program_code'] for p in d['programs']]
assert codes == sorted(codes), f'Not sorted: {codes}'
# Check all required fields in each program
req = ['program_code','program_name','active_sites','acquisition_cost_per_1000_units_usd','units_per_day','cooler_type','cooler_cost_usd','payment_per_dispatch_per_site_usd','annual_drug_cost_10_day_usd','annual_drug_cost_20_day_usd','annual_cooler_cost_10_day_usd','annual_cooler_cost_20_day_usd','annual_revenue_10_day_usd','annual_revenue_20_day_usd','annual_margin_10_day_usd','annual_margin_20_day_usd','annual_margin_difference_20_minus_10_usd']
for p in d['programs']:
    for k in req:
        assert k in p, f'Missing {k} in {p["program_code"]}'
print('JSON validation passed.')
print(json.dumps(d, indent=2))
"

# Check markdown
cat /root/oncocooler_summary.md
python3 -c "
with open('/root/oncocooler_summary.md') as f:
    lines = [l for l in f.read().strip().split('\\n') if l.strip()]
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
print(f'Markdown has {len(lines)} non-empty lines. OK.')
"
```

## Important Notes
- Read all input files first before running the script, so you can debug any field name mismatches.
- The `annual_cooler_cost` is per-program: `cooler_cost_usd * dispatches_per_year`. It is NOT multiplied by active_sites (cooler cost is per dispatch, not per site per dispatch). Verify this matches the formulas: the instructions say "cooler cost uses cooler_cost_usd from cooler_cost.csv" and the margin formula is `revenue - drug_cost - cooler_cost`. The cooler cost formula is not explicitly given per-site, so use just `cooler_cost_usd * dispatches_per_year` unless the data or instructions indicate otherwise. **However**, re-read the instructions carefully—if the cooler cost CSV has a per-dispatch cost and the task doesn't say per-site, then it's just per dispatch times number of dispatches. Check the CSV to understand the unit.
- After reading the CSV files, if any column names differ from what the script expects (e.g., extra whitespace, different casing), adjust the script accordingly before running.
- If the script fails, read the error, fix the issue, and re-run. Do not give up.

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