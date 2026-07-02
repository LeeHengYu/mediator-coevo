# Task Instruction

Execute the following steps in order to produce `/root/vaxcrate_analysis.json` and `/root/vaxcrate_summary.md`.

## Step 1 – Inspect all input files and the test suite

```bash
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

Also inspect the verifier/test file:
```bash
find /root -name '*.py' -path '*/test*' | head -20
cat /root/tests/test_output.py 2>/dev/null || cat /root/tests/test_outputs.py 2>/dev/null || find /root -name 'test_*' -exec cat {} \;
```

Read and understand the exact assertions the test makes (key names, formatting expectations, schema structure).

## Step 2 – Write and run a Python script

Create `/root/solve.py` with the complete logic below. Adapt field names if the input files use slightly different headers (confirm by inspecting the files first).

```python
import json, csv, math, os

# ── Load inputs ──────────────────────────────────────────────
with open('/root/campaign_manifest.json') as f:
    manifest = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

crate_cost_rows = read_csv('/root/crate_cost.csv')
billing_rows = read_csv('/root/billing.csv')
override_rows = read_csv('/root/location_overrides.csv')
suspension_rows = read_csv('/root/suspensions.csv')

# ── Build lookup maps ────────────────────────────────────────
crate_cost_map = {r['crate_tier'].strip(): float(r['crate_cost_usd'].strip()) for r in crate_cost_rows}

# Suspended campaign_ids (hold)
suspended_ids = set()
for r in suspension_rows:
    if r.get('suspension_status','').strip().lower() == 'hold':
        suspended_ids.add(r['campaign_id'].strip())

# ── Filter campaigns: analysis_flag == review, not suspended ─
campaigns = []
for c in manifest.get('campaigns', manifest) if isinstance(manifest, list) else manifest.get('campaigns', []):
    if str(c.get('analysis_flag','')).strip().lower() != 'review':
        continue
    cid = str(c['campaign_id']).strip()
    if cid in suspended_ids:
        continue
    campaigns.append(c)

# ── Resolve billing rows ─────────────────────────────────────
# Build campaign_id -> campaign object map and label -> campaign_id map
cid_map = {}
label_to_cid = {}
for c in campaigns:
    cid = str(c['campaign_id']).strip()
    cid_map[cid] = c
    label_to_cid[str(c.get('campaign_name','')).strip().lower()] = cid
    for alias in c.get('alias_labels', []):
        label_to_cid[str(alias).strip().lower()] = cid

# Filter active billing rows, resolve to campaign_id
billing_by_cid = {}  # cid -> list of rows
for r in billing_rows:
    if r.get('status','').strip().lower() != 'active':
        continue
    label = r.get('campaign_label','').strip().lower()
    cid = label_to_cid.get(label)
    if cid is None:
        continue
    billing_by_cid.setdefault(cid, []).append(r)

# Keep latest cycle_tag per campaign
retained_billing = {}
for cid, rows in billing_by_cid.items():
    best = max(rows, key=lambda r: r.get('cycle_tag','').strip())
    retained_billing[cid] = best

# ── Resolve active_clinics from location_overrides ───────────
overrides_by_cid = {}
for r in override_rows:
    if r.get('state','').strip().lower() != 'approved':
        continue
    rev = r.get('revision','').strip()
    ac = r.get('active_clinics','').strip()
    if rev == '' or ac == '':
        continue
    cid = r.get('campaign_id','').strip()
    try:
        rev_num = float(rev)
        ac_num = float(ac)
    except ValueError:
        continue
    if cid not in overrides_by_cid or rev_num > overrides_by_cid[cid][0]:
        overrides_by_cid[cid] = (rev_num, ac_num)

def get_active_clinics(c):
    cid = str(c['campaign_id']).strip()
    if cid in overrides_by_cid:
        return int(overrides_by_cid[cid][1])
    return int(c.get('default_active_clinics', 0))

# ── Compute per-campaign metrics ─────────────────────────────
results = []
for c in campaigns:
    cid = str(c['campaign_id']).strip()
    if cid not in retained_billing:
        continue  # no active billing row
    bill = retained_billing[cid]
    ac = get_active_clinics(c)
    drug_cost_per_1000 = float(c['drug_cost_per_1000_doses_usd'])
    doses_per_day = float(c['doses_per_day'])
    crate_tier = str(c['crate_tier']).strip()
    crate_cost = crate_cost_map[crate_tier]
    payment = float(bill['payment_per_dispatch_per_clinic_usd'])

    # 6-day
    rev_6 = payment * ac * 60
    drug_6 = drug_cost_per_1000 * ac * doses_per_day * 6 * 60 / 1000
    crate_6 = crate_cost * 60  # crate_cost per dispatch * dispatches
    # Wait – re-read: annual_crate_cost. The instruction says "crate cost uses crate_cost_usd from crate_cost.csv"
    # It doesn't specify a per-clinic or per-dispatch formula beyond listing crate_cost_usd.
    # Let me check the test to see what's expected. For now, assume crate_cost is per dispatch.
    # Actually the instruction only says crate_cost_usd matched by crate_tier.
    # Annual crate cost = crate_cost_usd * dispatches_per_year (no per-clinic multiplier unless test says otherwise)
    margin_6 = rev_6 - drug_6 - crate_6

    # 12-day
    rev_12 = payment * ac * 30
    drug_12 = drug_cost_per_1000 * ac * doses_per_day * 12 * 30 / 1000
    crate_12 = crate_cost * 30
    margin_12 = rev_12 - drug_12 - crate_12

    diff = margin_12 - margin_6

    results.append({
        'campaign_id': cid,
        'campaign_name': str(c.get('campaign_name','')).strip(),
        'active_clinics': ac,
        'drug_cost_per_1000_doses_usd': round(drug_cost_per_1000, 2),
        'doses_per_day': round(doses_per_day, 2),
        'crate_tier': crate_tier,
        'crate_cost_usd': round(crate_cost, 2),
        'payment_per_dispatch_per_clinic_usd': round(payment, 2),
        'annual_drug_cost_6_day_usd': round(drug_6, 2),
        'annual_drug_cost_12_day_usd': round(drug_12, 2),
        'annual_crate_cost_6_day_usd': round(crate_6, 2),
        'annual_crate_cost_12_day_usd': round(crate_12, 2),
        'annual_revenue_6_day_usd': round(rev_6, 2),
        'annual_revenue_12_day_usd': round(rev_12, 2),
        'annual_margin_6_day_usd': round(margin_6, 2),
        'annual_margin_12_day_usd': round(margin_12, 2),
        'annual_margin_difference_12_minus_6_usd': round(diff, 2)
    })

# Sort by campaign_id ascending
results.sort(key=lambda x: x['campaign_id'])

total_margin_6 = round(sum(r['annual_margin_6_day_usd'] for r in results), 2)
total_margin_12 = round(sum(r['annual_margin_12_day_usd'] for r in results), 2)
total_diff = round(total_margin_12 - total_margin_6, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 11000:
    decision = 'move_to_12_day'
else:
    decision = 'keep_6_day'

justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                 f'which is {"below" if abs_diff < 11000 else "at or above"} '
                 f'the $11,000.00 threshold, so the recommendation is {decision}.')

output = {
    'assumptions': {
        'dispatches_per_year_6_day': 60,
        'dispatches_per_year_12_day': 30,
        'days_per_dispatch_6_day': 6,
        'days_per_dispatch_12_day': 12,
        'switch_threshold_usd': 11000,
        'override_rule': 'highest numeric approved revision with non-empty active_clinics, else default_active_clinics',
        'suspension_rule': 'exclude hold campaigns'
    },
    'campaigns': results,
    'totals': {
        'total_annual_margin_6_day_usd': total_margin_6,
        'total_annual_margin_12_day_usd': total_margin_12,
        'total_annual_margin_difference_12_minus_6_usd': total_diff,
        'absolute_total_margin_difference_usd': abs_diff
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/vaxcrate_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

# ── Summary markdown ─────────────────────────────────────────
lines = [
    '# VaxCrate 6-Day vs 12-Day Dispatch Analysis Summary',
    '',
    f'Total 6-day annual margin: ${total_margin_6:,.2f}',
    f'Total 12-day annual margin: ${total_margin_12:,.2f}',
    f'Absolute margin difference: ${abs_diff:,.2f}',
    f'Recommendation: {decision}',
    '',
    justification
]

with open('/root/vaxcrate_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Output files written.')
```

## Step 3 – IMPORTANT: Before running, inspect the test file

Read the test file carefully. Look for:
- Expected key names in the JSON (especially in totals, campaigns, assumptions)
- Whether `annual_crate_cost` is expected to be `crate_cost_usd * dispatches_per_year` or `crate_cost_usd * active_clinics * dispatches_per_year`
- Currency formatting expectations in the summary (commas, dollar signs)
- Any other schema expectations

Adjust the script accordingly before running it.

## Step 4 – Also inspect the manifest structure

The manifest might be `{"campaigns": [...]}` or just `[...]`. Adjust the parsing. Also check if `alias_labels` is a list or comma-separated string.

## Step 5 – Run the script

```bash
python3 /root/solve.py
```

## Step 6 – Validate outputs

```bash
cat /root/vaxcrate_analysis.json
cat /root/vaxcrate_summary.md
```

Verify:
- JSON is valid and has all required keys: `assumptions`, `campaigns`, `totals`, `recommendation`
- Each campaign has ALL fields from the schema including `drug_cost_per_1000_doses_usd`, `doses_per_day`, `annual_margin_difference_12_minus_6_usd`
- Summary has 4-8 non-empty lines with comma-formatted currency values
- Campaigns sorted by campaign_id ascending

## Step 7 – Run the test suite

```bash
cd /root && python3 -m pytest tests/ -v 2>&1 | head -80
```

If any test fails, read the error carefully, fix the script, and re-run. Common pitfalls from prior feedback:
- Missing `assumptions` key
- Wrong field names (must be exact match)
- Currency values in summary must use comma formatting (e.g., `$-83,406.84` not `$-83406.84`)
- The crate cost formula might need `active_clinics` multiplier – check the test
- `annual_margin_difference_12_minus_6_usd` not `margin_difference_12_minus_6_usd`

Iterate until all tests pass.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[vaccination, json, csv, distractor-handling, decision-analysis].
Verifier config: timeout_sec=900.0.