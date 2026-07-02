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

Create `/root/solve.py` with the following content and run it with `python3 /root/solve.py`.

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
payment_rows = read_csv('/root/contract_payment.csv')
override_rows = read_csv('/root/site_overrides.csv')

# ── 1. Filter in-scope programs (review_flag == 'review') ────
in_scope = [p for p in catalog if p.get('review_flag') == 'review']

# ── 2. Build label→program_code lookup ───────────────────────
label_to_program = {}
for p in in_scope:
    label_to_program[p['program_name']] = p['program_code']
    for lbl in p.get('known_labels', []):
        label_to_program[lbl] = p['program_code']

# ── 3. Resolve contract_payment rows to in-scope programs ────
payment_map = {}  # program_code → payment_per_dispatch_per_site_usd
for row in payment_rows:
    pl = row['program_label']
    if pl in label_to_program:
        pc = label_to_program[pl]
        payment_map[pc] = float(row['payment_per_dispatch_per_site_usd'])

# ── 4. Resolve active_sites from site_overrides ──────────────
# Keep only approved rows; for each program_code keep highest version_no
approved = [r for r in override_rows if r.get('approval_state','').strip().lower() == 'approved']
best_override = {}  # program_code → row
for r in approved:
    pc = r['program_code']
    vn = int(r['version_no'])
    if pc not in best_override or vn > int(best_override[pc]['version_no']):
        best_override[pc] = r

# ── 5. Build cooler cost lookup ──────────────────────────────
cooler_map = {}
for r in cooler_rows:
    cooler_map[r['cooler_type']] = float(r['cooler_cost_usd'])

# ── 6. Compute per-program metrics ──────────────────────────
programs_out = []
for p in in_scope:
    pc = p['program_code']
    pname = p['program_name']

    # active sites
    if pc in best_override:
        active_sites = int(best_override[pc]['active_sites'])
    else:
        active_sites = int(p['default_active_sites'])

    acq_cost = float(p['acquisition_cost_per_1000_units_usd'])
    upd = float(p['units_per_day'])
    ct = p['cooler_type']
    cooler_cost = cooler_map[ct]
    payment = payment_map[pc]

    # 10-day model
    disp10 = 36; days10 = 10
    rev10 = payment * active_sites * disp10
    drug10 = acq_cost * active_sites * upd * days10 * disp10 / 1000.0
    cooler10 = cooler_cost  # annual cooler cost (one cooler cost per year? Need to check)
    # Re-read: "cooler_cost_usd from cooler_cost.csv" — the instructions don't multiply by dispatches or sites.
    # But logically cooler cost should be per dispatch. Let me re-examine the formulas.
    # The instructions say:
    #   annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost
    # But they don't give an explicit formula for annual_cooler_cost.
    # Given that the schema has annual_cooler_cost_10_day_usd and annual_cooler_cost_20_day_usd,
    # and cooler_cost_usd is per cooler, it's most likely:
    #   annual_cooler_cost = cooler_cost_usd * dispatches_per_year
    # (one cooler per dispatch, cost is per cooler/dispatch)
    cooler10 = cooler_cost * disp10
    margin10 = rev10 - drug10 - cooler10

    # 20-day model
    disp20 = 18; days20 = 20
    rev20 = payment * active_sites * disp20
    drug20 = acq_cost * active_sites * upd * days20 * disp20 / 1000.0
    cooler20 = cooler_cost * disp20
    margin20 = rev20 - drug20 - cooler20

    diff = margin20 - margin10

    programs_out.append({
        'program_code': pc,
        'program_name': pname,
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
        'annual_margin_difference_20_minus_10_usd': round(diff, 2),
    })

# Sort by program_code ascending
programs_out.sort(key=lambda x: x['program_code'])

# ── 7. Totals ────────────────────────────────────────────────
total_m10 = round(sum(p['annual_margin_10_day_usd'] for p in programs_out), 2)
total_m20 = round(sum(p['annual_margin_20_day_usd'] for p in programs_out), 2)
total_diff = round(total_m20 - total_m10, 2)
abs_diff = round(abs(total_diff), 2)

# ── 8. Decision ──────────────────────────────────────────────
if abs_diff < 10000:
    decision = 'move_to_20_day'
    justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                     f'which is below the $10,000 threshold. '
                     f'Recommend moving to 20-day dispatches.')
else:
    decision = 'keep_10_day'
    justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                     f'which exceeds the $10,000 threshold. '
                     f'Recommend keeping 10-day dispatches.')

# ── 9. Build output JSON ─────────────────────────────────────
output = {
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

print('=== JSON written ===')
print(json.dumps(output, indent=2))

# ── 10. Build summary markdown ───────────────────────────────
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

print('\n=== Summary written ===')
with open('/root/oncocooler_summary.md') as f:
    print(f.read())
```

## Step 3: Validate outputs

After the script runs, verify:

1. **JSON validity:**
```bash
python3 -c "import json; d=json.load(open('/root/oncocooler_analysis.json')); print('programs:', len(d['programs'])); print('totals:', d['totals']); print('decision:', d['recommendation']['decision'])"
```

2. **Markdown line count and required content:**
```bash
wc -l /root/oncocooler_summary.md
grep -c 'move_to_20_day\|keep_10_day' /root/oncocooler_summary.md
```

3. **Programs sorted by program_code:**
```bash
python3 -c "import json; d=json.load(open('/root/oncocooler_analysis.json')); codes=[p['program_code'] for p in d['programs']]; assert codes==sorted(codes), 'NOT SORTED'; print('Sorted OK:', codes)"
```

4. **All currency values rounded to 2 decimals:**
```bash
python3 -c "
import json
d=json.load(open('/root/oncocooler_analysis.json'))
for p in d['programs']:
    for k,v in p.items():
        if isinstance(v, float):
            assert round(v,2)==v, f'{k}={v} not rounded'
for k,v in d['totals'].items():
    assert round(v,2)==v, f'{k}={v} not rounded'
print('All values properly rounded.')
"
```

5. **Summary has 4-8 non-empty lines:**
```bash
python3 -c "
lines = [l for l in open('/root/oncocooler_summary.md').read().strip().split('\n') if l.strip()]
print(f'Non-empty lines: {len(lines)}')
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
print('Line count OK')
"
```

If the initial script fails (e.g., missing keys, KeyError), inspect the actual data files to understand column names and structure, then fix the script accordingly. The most likely issues are:
- Column names in CSVs may have whitespace or different casing
- `program_catalog.json` may be a list or a dict
- `known_labels` might not exist for all programs
- The cooler_cost formula may need adjustment

If any validation fails, debug and fix before considering the task complete.

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