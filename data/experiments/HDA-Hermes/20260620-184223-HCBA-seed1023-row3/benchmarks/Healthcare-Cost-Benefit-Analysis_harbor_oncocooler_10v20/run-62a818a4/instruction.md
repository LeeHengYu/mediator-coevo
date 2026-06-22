# Task Instruction

Execute the following steps in order to produce `/root/oncocooler_analysis.json` and `/root/oncocooler_summary.md`.

## Step 0 — Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

Read every file carefully before writing any code.

## Step 1 — Write and run the analysis script

Create `/root/solve.py` with the following logic. Follow the formulas and rules **exactly**.

```python
import json, csv, math

# ── Load inputs ──────────────────────────────────────────────────────
with open('/root/program_catalog.json') as f:
    catalog = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

cooler_rows = read_csv('/root/cooler_cost.csv')
payment_rows = read_csv('/root/contract_payment.csv')
override_rows = read_csv('/root/site_overrides.csv')

# ── Constants ────────────────────────────────────────────────────────
D10_DAYS = 10
D20_DAYS = 20
D10_DISPATCHES = 36
D20_DISPATCHES = 18
THRESHOLD = 10000

# ── Filter in-scope programs (review_flag == 'review') ───────────────
in_scope = [p for p in catalog if p.get('review_flag') == 'review']

# ── Build cooler cost lookup ─────────────────────────────────────────
cooler_lookup = {}
for r in cooler_rows:
    cooler_lookup[r['cooler_type'].strip()] = float(r['cooler_cost_usd'])

# ── Build payment lookup: program_label -> payment_per_dispatch_per_site_usd
#    Map each payment row's program_label to a catalog program via
#    program_name or known_labels.
payment_by_label = {}
for r in payment_rows:
    payment_by_label[r['program_label'].strip()] = float(r['payment_per_dispatch_per_site_usd'])

def find_payment(prog):
    name = prog['program_name'].strip()
    if name in payment_by_label:
        return payment_by_label[name]
    for lbl in prog.get('known_labels', []):
        if lbl.strip() in payment_by_label:
            return payment_by_label[lbl.strip()]
    return None

# ── Build active-sites lookup from site_overrides ────────────────────
#    Only approved rows; keep highest version_no per program_code.
approved = [r for r in override_rows if r['approval_state'].strip().lower() == 'approved']
best_override = {}
for r in approved:
    pc = r['program_code'].strip()
    vn = int(r['version_no'])
    if pc not in best_override or vn > best_override[pc][1]:
        best_override[pc] = (int(r['active_sites']), vn)

def get_active_sites(prog):
    pc = prog['program_code'].strip()
    if pc in best_override:
        return best_override[pc][0]
    return int(prog['default_active_sites'])

# ── Compute per-program financials ───────────────────────────────────
programs_out = []
for prog in in_scope:
    pc = prog['program_code'].strip()
    pname = prog['program_name'].strip()
    active_sites = get_active_sites(prog)
    acq_cost = float(prog['acquisition_cost_per_1000_units_usd'])
    upd = float(prog['units_per_day'])
    ct = prog['cooler_type'].strip()
    cooler_cost = cooler_lookup[ct]
    payment = find_payment(prog)
    if payment is None:
        continue  # no matching payment row → skip (should not happen for in-scope)

    # Drug cost = acq_cost * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000
    drug_10 = acq_cost * active_sites * upd * D10_DAYS * D10_DISPATCHES / 1000
    drug_20 = acq_cost * active_sites * upd * D20_DAYS * D20_DISPATCHES / 1000

    # Cooler cost  (per dispatch, so multiply by dispatches_per_year)
    # NOTE: cooler_cost_usd is per-cooler-per-dispatch cost.
    # Annual cooler cost = cooler_cost_usd * dispatches_per_year
    cooler_10 = cooler_cost * D10_DISPATCHES
    cooler_20 = cooler_cost * D20_DISPATCHES

    # Revenue = payment_per_dispatch_per_site * active_sites * dispatches_per_year
    rev_10 = payment * active_sites * D10_DISPATCHES
    rev_20 = payment * active_sites * D20_DISPATCHES

    margin_10 = rev_10 - drug_10 - cooler_10
    margin_20 = rev_20 - drug_20 - cooler_20
    diff = margin_20 - margin_10

    programs_out.append({
        'program_code': pc,
        'program_name': pname,
        'active_sites': active_sites,
        'acquisition_cost_per_1000_units_usd': round(acq_cost, 2),
        'units_per_day': round(upd, 2),
        'cooler_type': ct,
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
        'annual_margin_difference_20_minus_10_usd': round(diff, 2),
    })

# Sort by program_code ascending
programs_out.sort(key=lambda x: x['program_code'])

# ── Totals ───────────────────────────────────────────────────────────
total_10 = round(sum(p['annual_margin_10_day_usd'] for p in programs_out), 2)
total_20 = round(sum(p['annual_margin_20_day_usd'] for p in programs_out), 2)
total_diff = round(total_20 - total_10, 2)
abs_diff = round(abs(total_diff), 2)

# ── Decision ─────────────────────────────────────────────────────────
if abs_diff < THRESHOLD:
    decision = 'move_to_20_day'
    justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                     f'which is below the ${THRESHOLD:,.2f} threshold. '
                     f'Moving to 20-day dispatches is recommended.')
else:
    decision = 'keep_10_day'
    justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                     f'which exceeds the ${THRESHOLD:,.2f} threshold. '
                     f'Keeping 10-day dispatches is recommended.')

# ── Build output JSON ────────────────────────────────────────────────
output = {
    'assumptions': {
        'dispatches_per_year_10_day': D10_DISPATCHES,
        'dispatches_per_year_20_day': D20_DISPATCHES,
        'days_per_dispatch_10_day': D10_DAYS,
        'days_per_dispatch_20_day': D20_DAYS,
        'switch_threshold_usd': THRESHOLD,
        'site_override_rule': 'highest approved version_no per program_code, else default_active_sites',
    },
    'programs': programs_out,
    'totals': {
        'total_annual_margin_10_day_usd': total_10,
        'total_annual_margin_20_day_usd': total_20,
        'total_annual_margin_difference_20_minus_10_usd': total_diff,
        'absolute_total_margin_difference_usd': abs_diff,
    },
    'recommendation': {
        'decision': decision,
        'justification': justification,
    },
}

with open('/root/oncocooler_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print('JSON written.')
print(f'Total 10-day margin: {total_10}')
print(f'Total 20-day margin: {total_20}')
print(f'Total diff: {total_diff}  abs: {abs_diff}')
print(f'Decision: {decision}')

# ── Build summary markdown ───────────────────────────────────────────
lines = [
    '# OncoCooler Dispatch Analysis Summary',
    '',
    f'- Total 10-day annual margin: ${total_10:,.2f}',
    f'- Total 20-day annual margin: ${total_20:,.2f}',
    f'- Absolute margin difference: ${abs_diff:,.2f}',
    f'- Recommendation: **{decision}**',
    '',
    justification,
]

with open('/root/oncocooler_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Summary written.')
```

Run:
```bash
python3 /root/solve.py
```

## Step 2 — Inspect outputs and spot-check drug cost formula

After the script runs, print the first program entry to verify the drug-cost magnitude matches the formula `acq_cost * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000`.

```bash
python3 -c "
import json
with open('/root/oncocooler_analysis.json') as f:
    d = json.load(f)
for p in d['programs'][:2]:
    print(json.dumps(p, indent=2))
print('Totals:', json.dumps(d['totals'], indent=2))
print('Decision:', d['recommendation']['decision'])
"
```

Also verify the summary:
```bash
cat /root/oncocooler_summary.md
```

## Step 3 — Validate schema conformance

Check that the JSON has exactly the required top-level keys (`assumptions`, `programs`, `totals`, `recommendation`) and no extra keys. Check that each program dict has exactly the 17 specified keys and no extras.

```bash
python3 -c "
import json
with open('/root/oncocooler_analysis.json') as f:
    d = json.load(f)

expected_top = {'assumptions','programs','totals','recommendation'}
assert set(d.keys()) == expected_top, f'Extra/missing top keys: {set(d.keys()) ^ expected_top}'

prog_keys = {'program_code','program_name','active_sites',
  'acquisition_cost_per_1000_units_usd','units_per_day','cooler_type',
  'cooler_cost_usd','payment_per_dispatch_per_site_usd',
  'annual_drug_cost_10_day_usd','annual_drug_cost_20_day_usd',
  'annual_cooler_cost_10_day_usd','annual_cooler_cost_20_day_usd',
  'annual_revenue_10_day_usd','annual_revenue_20_day_usd',
  'annual_margin_10_day_usd','annual_margin_20_day_usd',
  'annual_margin_difference_20_minus_10_usd'}
for i,p in enumerate(d['programs']):
    assert set(p.keys()) == prog_keys, f'Program {i} key mismatch: {set(p.keys()) ^ prog_keys}'

print('Schema OK')
"
```

## Step 4 — Validate summary requirements

Check the summary has 4-8 non-empty lines, includes USD margin figures with commas, and contains the decision slug.

```bash
python3 -c "
with open('/root/oncocooler_summary.md') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
print(f'Non-empty lines: {len(lines)}')
assert 4 <= len(lines) <= 8, f'Line count {len(lines)} out of range'
text = ' '.join(lines)
assert 'move_to_20_day' in text or 'keep_10_day' in text, 'Missing decision slug'
print('Summary OK')
"
```

## Important notes on the previous failure

The prior run had a **factor-of-15 error** in drug costs. The drug cost formula must multiply by **days_per_dispatch × dispatches_per_year** (i.e., 10×36=360 for 10-day, 20×18=360 for 20-day). If you get identical drug costs for both models, that is correct — the total annual days are the same (360). The margin difference then comes entirely from revenue and cooler cost differences. Double-check this: 10×36 = 360 annual days = 20×18 = 360 annual days, so drug costs are equal for both models. The difference is driven by:
- Revenue: payment × sites × 36 vs payment × sites × 18 (halved)
- Cooler cost: cooler × 36 vs cooler × 18 (halved)
- Drug cost: identical

Verify this reasoning against the printed output in Step 2.

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