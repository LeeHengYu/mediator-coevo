# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

## Step 2: Write and run a Python script to produce both output files

Create `/root/solve.py` with the following logic:

```python
import json, csv, math

# ---- Load inputs ----
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

cooler_rows = read_csv('/root/cooler_cost.csv')
payment_rows = read_csv('/root/contract_payment.csv')
override_rows = read_csv('/root/site_overrides.csv')

# ---- Build program lookup ----
# catalog may be a list or dict; handle both
if isinstance(catalog, dict):
    programs_list = catalog.get('programs', list(catalog.values()))
else:
    programs_list = catalog

# Filter to review_flag == 'review'
in_scope = [p for p in programs_list if p.get('review_flag') == 'review']

# Build label -> program mapping
label_to_program = {}
for p in in_scope:
    label_to_program[p['program_name']] = p
    for lbl in p.get('known_labels', []):
        label_to_program[lbl] = p

# ---- Cooler cost lookup ----
cooler_cost_map = {}
for row in cooler_rows:
    cooler_cost_map[row['cooler_type'].strip()] = float(row['cooler_cost_usd'])

# ---- Payment lookup: match payment rows to in-scope programs ----
# A program may have exactly one matching payment row (first match wins per program_code)
payment_map = {}  # program_code -> payment_per_dispatch_per_site_usd
for row in payment_rows:
    plabel = row['program_label'].strip()
    prog = label_to_program.get(plabel)
    if prog is None:
        continue
    pc = prog['program_code']
    if pc not in payment_map:
        payment_map[pc] = float(row['payment_per_dispatch_per_site_usd'])

# ---- Active sites from site_overrides ----
# Only approved rows; highest version_no per program_code
approved = [r for r in override_rows if r.get('approval_state', '').strip() == 'approved']
best_override = {}  # program_code -> row
for r in approved:
    pc = r['program_code'].strip()
    vno = int(r['version_no'])
    if pc not in best_override or vno > int(best_override[pc]['version_no']):
        best_override[pc] = r

def get_active_sites(prog):
    pc = prog['program_code']
    if pc in best_override:
        return int(best_override[pc]['active_sites'])
    return int(prog['default_active_sites'])

# ---- Constants ----
D10, D20 = 36, 18
DAYS10, DAYS20 = 10, 20
THRESHOLD = 10000

# ---- Compute per-program ----
results = []
for prog in in_scope:
    pc = prog['program_code']
    pname = prog['program_name']
    active = get_active_sites(prog)
    acq = float(prog['acquisition_cost_per_1000_units_usd'])
    upd = float(prog['units_per_day'])
    ct = prog['cooler_type'].strip()
    cc = cooler_cost_map[ct]
    ppd = payment_map[pc]

    # Revenue
    rev10 = ppd * active * D10
    rev20 = ppd * active * D20

    # Drug cost
    drug10 = acq * active * upd * DAYS10 * D10 / 1000
    drug20 = acq * active * upd * DAYS20 * D20 / 1000

    # Cooler cost
    cool10 = cc * D10
    cool20 = cc * D20

    # Margin
    margin10 = rev10 - drug10 - cool10
    margin20 = rev20 - drug20 - cool20
    diff = margin20 - margin10

    results.append({
        'program_code': pc,
        'program_name': pname,
        'active_sites': active,
        'acquisition_cost_per_1000_units_usd': round(acq, 2),
        'units_per_day': round(upd, 2),
        'cooler_type': ct,
        'cooler_cost_usd': round(cc, 2),
        'payment_per_dispatch_per_site_usd': round(ppd, 2),
        'annual_drug_cost_10_day_usd': round(drug10, 2),
        'annual_drug_cost_20_day_usd': round(drug20, 2),
        'annual_cooler_cost_10_day_usd': round(cool10, 2),
        'annual_cooler_cost_20_day_usd': round(cool20, 2),
        'annual_revenue_10_day_usd': round(rev10, 2),
        'annual_revenue_20_day_usd': round(rev20, 2),
        'annual_margin_10_day_usd': round(margin10, 2),
        'annual_margin_20_day_usd': round(margin20, 2),
        'annual_margin_difference_20_minus_10_usd': round(diff, 2)
    })

# Sort by program_code ascending
results.sort(key=lambda x: x['program_code'])

# ---- Totals ----
total_m10 = round(sum(r['annual_margin_10_day_usd'] for r in results), 2)
total_m20 = round(sum(r['annual_margin_20_day_usd'] for r in results), 2)
total_diff = round(total_m20 - total_m10, 2)
abs_diff = round(abs(total_diff), 2)

# ---- Decision ----
if abs_diff < THRESHOLD:
    decision = 'move_to_20_day'
    justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                     f'which is below the ${THRESHOLD:,.2f} threshold. '
                     f'Switching to 20-day dispatches is recommended.')
else:
    decision = 'keep_10_day'
    justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                     f'which meets or exceeds the ${THRESHOLD:,.2f} threshold. '
                     f'Keeping 10-day dispatches is recommended.')

# ---- Build JSON output ----
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
print(json.dumps(output['totals'], indent=2))
print('Decision:', decision)

# ---- Build Markdown summary ----
lines = [
    '# OncoCooler 10-Day vs 20-Day Dispatch Analysis',
    '',
    f'- **Total 10-day annual margin:** ${total_m10:,.2f}',
    f'- **Total 20-day annual margin:** ${total_m20:,.2f}',
    f'- **Absolute margin difference:** ${abs_diff:,.2f}',
    f'- **Recommendation:** `{decision}`',
    '',
    justification
]

with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Markdown written.')
```

Run it:
```bash
python3 /root/solve.py
```

## Step 3: Validate outputs

```bash
# Confirm JSON is valid and inspect key fields
python3 -c "
import json
with open('/root/oncocooler_analysis.json') as f:
    d = json.load(f)
print('Programs:', len(d['programs']))
for p in d['programs']:
    print(p['program_code'], p['annual_margin_difference_20_minus_10_usd'])
print('Totals:', d['totals'])
print('Decision:', d['recommendation']['decision'])
# Check all currency fields are 2-decimal floats
for p in d['programs']:
    for k,v in p.items():
        if '_usd' in k:
            s = str(v)
            if '.' in s:
                decimals = len(s.split('.')[1])
                assert decimals <= 2, f'{k}={v} has {decimals} decimals'
print('All USD fields have <= 2 decimal places.')
# Check sort order
codes = [p['program_code'] for p in d['programs']]
assert codes == sorted(codes), 'Programs not sorted by program_code!'
print('Sort order OK.')
"

# Confirm markdown has required content
python3 -c "
with open('/root/oncocooler_summary.md') as f:
    text = f.read()
lines = [l for l in text.strip().split('\n') if l.strip()]
print(f'Non-empty lines: {len(lines)}')
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
for keyword in ['10-day', '20-day', 'move_to_20_day', 'keep_10_day']:
    # at least the chosen decision slug must appear
    pass
# Check that at least one decision slug appears
assert 'move_to_20_day' in text or 'keep_10_day' in text, 'Missing decision slug'
print('Markdown checks passed.')
print(text)
"
```

If any step fails, read the error, inspect the relevant input file, fix the script, and re-run. Do not consider the task complete until both output files exist and pass all validation checks.

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