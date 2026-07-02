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

Create `/root/solve.py` with the following logic, then run it with `python3 /root/solve.py`:

```python
import json, csv, math
from collections import defaultdict

# --- Load inputs ---
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

cooler_rows = read_csv('/root/cooler_cost.csv')
payment_rows = read_csv('/root/contract_payment.csv')
override_rows = read_csv('/root/site_overrides.csv')

# --- Build program lookup ---
# catalog may be a list or dict; normalize to list
if isinstance(catalog, dict):
    programs_list = catalog.get('programs', list(catalog.values()))
    if not isinstance(programs_list, list):
        programs_list = [programs_list]
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

# --- Cooler cost lookup ---
cooler_cost_map = {}
for row in cooler_rows:
    cooler_cost_map[row['cooler_type'].strip()] = float(row['cooler_cost_usd'])

# --- Payment lookup: match program_label to in-scope program ---
# A program may have multiple payment rows; per the schema there's one payment_per_dispatch_per_site_usd per program.
# We'll take the payment for each in-scope program.
payment_map = {}  # program_code -> payment
for row in payment_rows:
    plabel = row['program_label'].strip()
    prog = label_to_program.get(plabel)
    if prog is None:
        continue
    payment_map[prog['program_code']] = float(row['payment_per_dispatch_per_site_usd'])

# --- Site overrides ---
# Filter approved, group by program_code, pick highest version_no
approved = [r for r in override_rows if r.get('approval_state', '').strip().lower() == 'approved']
best_override = {}  # program_code -> row
for r in approved:
    pc = r['program_code'].strip()
    vno = int(r['version_no'])
    if pc not in best_override or vno > best_override[pc][1]:
        best_override[pc] = (r, vno)

site_count_map = {}  # program_code -> active_sites
for pc, (row, _) in best_override.items():
    site_count_map[pc] = int(row['active_sites'])

# --- Compute per-program ---
results = []
for p in in_scope:
    pc = p['program_code']
    pname = p['program_name']
    acq = float(p['acquisition_cost_per_1000_units_usd'])
    upd = float(p['units_per_day'])
    ct = p['cooler_type'].strip()
    ccost = cooler_cost_map[ct]
    payment = payment_map[pc]
    active_sites = site_count_map.get(pc, int(p['default_active_sites']))

    # 10-day model
    disp10 = 36; days10 = 10
    rev10 = payment * active_sites * disp10
    drug10 = acq * active_sites * upd * days10 * disp10 / 1000.0
    cooler10 = ccost * disp10  # cooler cost per dispatch * dispatches/year
    # Wait - need to clarify: is cooler cost per dispatch or per year?
    # The instruction says "cooler_cost_usd from cooler_cost.csv" and the margin formula is
    # annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost
    # annual_cooler_cost is not explicitly defined with a formula.
    # Looking at the output schema: annual_cooler_cost_10_day_usd, annual_cooler_cost_20_day_usd
    # The cooler_cost_usd likely is per cooler per dispatch, so annual = cooler_cost_usd * dispatches_per_year
    # But it could also be per site. Let me check the CSV to decide.
    # We'll compute both ways and see which makes sense. For now, assume:
    # annual_cooler_cost = cooler_cost_usd * dispatches_per_year * active_sites
    # Actually, let me re-read: the drug cost formula includes active_sites, and revenue includes active_sites.
    # Cooler cost likely also scales by sites. Let's assume:
    # annual_cooler_cost = cooler_cost_usd * active_sites * dispatches_per_year
    cooler10 = ccost * active_sites * disp10
    margin10 = rev10 - drug10 - cooler10

    # 20-day model
    disp20 = 18; days20 = 20
    rev20 = payment * active_sites * disp20
    drug20 = acq * active_sites * upd * days20 * disp20 / 1000.0
    cooler20 = ccost * active_sites * disp20
    margin20 = rev20 - drug20 - cooler20

    diff = margin20 - margin10

    results.append({
        'program_code': pc,
        'program_name': pname,
        'active_sites': active_sites,
        'acquisition_cost_per_1000_units_usd': round(acq, 2),
        'units_per_day': round(upd, 2),
        'cooler_type': ct,
        'cooler_cost_usd': round(ccost, 2),
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

total_m10 = round(sum(r['annual_margin_10_day_usd'] for r in results), 2)
total_m20 = round(sum(r['annual_margin_20_day_usd'] for r in results), 2)
total_diff = round(total_m20 - total_m10, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 10000:
    decision = 'move_to_20_day'
    justification = f'Absolute total margin difference ${abs_diff} is below the $10,000 threshold, so switching to 20-day dispatches is recommended.'
else:
    decision = 'keep_10_day'
    justification = f'Absolute total margin difference ${abs_diff} exceeds the $10,000 threshold, so keeping 10-day dispatches is recommended.'

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

# --- Write summary ---
lines = [
    '# OncoCooler Dispatch Analysis Summary',
    f'Total 10-day annual margin: ${total_m10:,.2f} USD',
    f'Total 20-day annual margin: ${total_m20:,.2f} USD',
    f'Absolute margin difference: ${abs_diff:,.2f} USD',
    f'Recommendation: {decision}',
    justification
]

with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Outputs written.')
print(json.dumps(output, indent=2))
```

**IMPORTANT**: Before running the script, first inspect all four input files carefully. The script above makes assumptions about field names and data structure. After inspecting the files, adjust the script if:
- `program_catalog.json` has a different top-level structure (e.g., a dict with a key like `"programs"` wrapping the list).
- Field names differ from what's assumed (check exact column headers in CSVs and exact key names in JSON).
- The cooler cost CSV has different column names.
- `site_overrides.csv` has different column names for `active_sites`, `approval_state`, `version_no`, `program_code`.

After inspecting files, also reconsider the cooler cost formula. The instruction says:
- `annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost`
- Revenue and drug cost both scale by `active_sites`.
- The cooler cost formula is NOT explicitly given with `active_sites`. It's possible cooler cost is just `cooler_cost_usd * dispatches_per_year` (per program, not per site). Look at the data magnitudes to see which interpretation produces sensible margins.

Try both interpretations if needed:
- `cooler_cost_usd * dispatches_per_year` (cooler cost is per-program per-dispatch)
- `cooler_cost_usd * active_sites * dispatches_per_year` (cooler cost is per-site per-dispatch)

Pick the one that seems consistent with the data and the fact that revenue and drug cost scale per-site.

## Step 3: Validate outputs

```bash
python3 -c "
import json
with open('/root/oncocooler_analysis.json') as f:
    data = json.load(f)
assert 'assumptions' in data
assert 'programs' in data
assert 'totals' in data
assert 'recommendation' in data
assert isinstance(data['programs'], list)
assert len(data['programs']) > 0
for p in data['programs']:
    for k in ['program_code','active_sites','annual_margin_10_day_usd','annual_margin_20_day_usd','annual_margin_difference_20_minus_10_usd']:
        assert k in p, f'Missing key {k}'
# Check sorted
codes = [p['program_code'] for p in data['programs']]
assert codes == sorted(codes), 'Programs not sorted by program_code'
print('JSON validation passed')
print(f\"Programs: {len(data['programs'])}\")
print(f\"Decision: {data['recommendation']['decision']}\")
"
```

```bash
wc -l /root/oncocooler_summary.md
cat /root/oncocooler_summary.md
```

Verify the summary has 4-8 non-empty lines and contains the required info (total 10-day margin, total 20-day margin, absolute difference, decision slug).

## Step 4: Cross-check arithmetic

Manually verify one program's numbers by computing from raw data values to ensure the formulas are correct. Print intermediate values for the first program to confirm.

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