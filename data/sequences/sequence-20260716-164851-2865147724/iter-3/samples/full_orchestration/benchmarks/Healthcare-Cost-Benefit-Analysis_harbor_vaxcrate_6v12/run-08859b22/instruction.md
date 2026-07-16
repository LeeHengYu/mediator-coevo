# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure and data types:
```bash
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

2. **Create and run `/root/solve.py`** with the following logic:

```python
import json
import csv
import math

# Load campaign manifest
with open('/root/campaign_manifest.json', 'r') as f:
    manifest = json.load(f)

# Load crate costs
crate_costs = {}
with open('/root/crate_cost.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        crate_costs[row['crate_tier'].strip()] = float(row['crate_cost_usd'].strip())

# Load billing
billing_rows = []
with open('/root/billing.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        billing_rows.append({k.strip(): v.strip() for k, v in row.items()})

# Load location overrides
overrides = []
with open('/root/location_overrides.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        overrides.append({k.strip(): v.strip() for k, v in row.items()})

# Load suspensions
suspended_ids = set()
with open('/root/suspensions.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row = {k.strip(): v.strip() for k, v in row.items()}
        if row.get('suspension_status', '').lower() == 'hold':
            suspended_ids.add(row['campaign_id'].strip())

# Get campaigns list from manifest
# manifest could be a list or dict with a key
if isinstance(manifest, list):
    campaigns_list = manifest
elif isinstance(manifest, dict):
    # Could be dict with 'campaigns' key or just campaign_id keys
    if 'campaigns' in manifest:
        campaigns_list = manifest['campaigns']
    else:
        # Try to see if it's keyed by campaign_id
        campaigns_list = list(manifest.values()) if all(isinstance(v, dict) for v in manifest.values()) else [manifest]
else:
    campaigns_list = [manifest]

# Filter: analysis_flag == 'review'
def is_review(c):
    flag = c.get('analysis_flag', '')
    if isinstance(flag, bool):
        return flag
    return str(flag).strip().lower() == 'review'

review_campaigns = [c for c in campaigns_list if is_review(c)]

# Exclude suspended campaigns
retained = [c for c in review_campaigns if c['campaign_id'].strip() not in suspended_ids]

# Build alias mapping: campaign_label -> campaign
# For each campaign, collect campaign_name and alias_labels
def get_names(c):
    names = set()
    names.add(c['campaign_name'].strip().lower())
    aliases = c.get('alias_labels', [])
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(',')]
    for a in aliases:
        if a.strip():
            names.add(a.strip().lower())
    return names

# For each retained campaign, find the best billing row
def find_billing(campaign):
    names = get_names(campaign)
    cid = campaign['campaign_id'].strip()
    matching = []
    for br in billing_rows:
        status = br.get('status', '').lower()
        if status != 'active':
            continue
        label = br.get('campaign_label', '').strip().lower()
        if label in names:
            matching.append(br)
    if not matching:
        return None
    # Keep the one with the latest cycle_tag
    matching.sort(key=lambda x: x.get('cycle_tag', ''), reverse=True)
    return matching[0]

# For each retained campaign, find active_clinics from overrides
def find_active_clinics(campaign):
    cid = campaign['campaign_id'].strip()
    valid = []
    for o in overrides:
        if o.get('campaign_id', '').strip() != cid:
            continue
        if o.get('state', '').strip().lower() != 'approved':
            continue
        rev = o.get('revision', '').strip()
        ac = o.get('active_clinics', '').strip()
        if rev == '' or ac == '':
            continue
        try:
            rev_num = float(rev)
            ac_num = float(ac)
        except ValueError:
            continue
        valid.append((rev_num, ac_num, o))
    if valid:
        valid.sort(key=lambda x: x[0], reverse=True)
        return int(valid[0][1]) if valid[0][1] == int(valid[0][1]) else valid[0][1]
    # Use default
    default = campaign.get('default_active_clinics')
    if default is not None:
        return int(float(str(default).strip())) if float(str(default).strip()) == int(float(str(default).strip())) else float(str(default).strip())
    return 0

# Constants
disp_6 = 60
disp_12 = 30
days_6 = 6
days_12 = 12
threshold = 11000

results = []
for c in retained:
    cid = c['campaign_id'].strip()
    cname = c['campaign_name'].strip()
    
    billing = find_billing(c)
    if billing is None:
        continue  # skip if no billing found
    
    active_clinics = find_active_clinics(c)
    # Ensure active_clinics is a number
    active_clinics_val = active_clinics
    if isinstance(active_clinics_val, float) and active_clinics_val == int(active_clinics_val):
        active_clinics_val = int(active_clinics_val)
    
    drug_cost_per_1000 = float(str(c.get('drug_cost_per_1000_doses_usd', 0)).strip())
    doses_per_day = float(str(c.get('doses_per_day', 0)).strip())
    crate_tier = str(c.get('crate_tier', '')).strip()
    crate_cost = crate_costs.get(crate_tier, 0.0)
    payment = float(str(billing.get('payment_per_dispatch_per_clinic_usd', 0)).strip())
    
    # 6-day
    rev_6 = round(payment * active_clinics_val * disp_6, 2)
    drug_6 = round(drug_cost_per_1000 * active_clinics_val * doses_per_day * days_6 * disp_6 / 1000, 2)
    crate_6 = round(crate_cost * disp_6, 2)
    margin_6 = round(rev_6 - drug_6 - crate_6, 2)
    
    # 12-day
    rev_12 = round(payment * active_clinics_val * disp_12, 2)
    drug_12 = round(drug_cost_per_1000 * active_clinics_val * doses_per_day * days_12 * disp_12 / 1000, 2)
    crate_12 = round(crate_cost * disp_12, 2)
    margin_12 = round(rev_12 - drug_12 - crate_12, 2)
    
    diff = round(margin_12 - margin_6, 2)
    
    results.append({
        'campaign_id': cid,
        'campaign_name': cname,
        'active_clinics': active_clinics_val,
        'drug_cost_per_1000_doses_usd': drug_cost_per_1000,
        'doses_per_day': doses_per_day,
        'crate_tier': crate_tier,
        'crate_cost_usd': crate_cost,
        'payment_per_dispatch_per_clinic_usd': payment,
        'annual_drug_cost_6_day_usd': drug_6,
        'annual_drug_cost_12_day_usd': drug_12,
        'annual_crate_cost_6_day_usd': crate_6,
        'annual_crate_cost_12_day_usd': crate_12,
        'annual_revenue_6_day_usd': rev_6,
        'annual_revenue_12_day_usd': rev_12,
        'annual_margin_6_day_usd': margin_6,
        'annual_margin_12_day_usd': margin_12,
        'annual_margin_difference_12_minus_6_usd': diff
    })

# Sort by campaign_id ascending
results.sort(key=lambda x: x['campaign_id'])

# Totals
total_margin_6 = round(sum(r['annual_margin_6_day_usd'] for r in results), 2)
total_margin_12 = round(sum(r['annual_margin_12_day_usd'] for r in results), 2)
total_diff = round(total_margin_12 - total_margin_6, 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 11000:
    decision = 'move_to_12_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} is below the ${threshold:,} threshold, so consolidating to 12-day dispatches is recommended.'
else:
    decision = 'keep_6_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} exceeds the ${threshold:,} threshold, so keeping 6-day dispatches is recommended.'

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

print('JSON written successfully.')
print(f'Total 6-day margin: ${total_margin_6:,.2f}')
print(f'Total 12-day margin: ${total_margin_12:,.2f}')
print(f'Absolute difference: ${abs_diff:,.2f}')
print(f'Decision: {decision}')

# Write markdown summary
with open('/root/vaxcrate_summary.md', 'w') as f:
    f.write('# VaxCrate Dispatch Policy Analysis Summary\n')
    f.write(f'\n')
    f.write(f'Total 6-day annual margin: ${total_margin_6:,.2f}\n')
    f.write(f'Total 12-day annual margin: ${total_margin_12:,.2f}\n')
    f.write(f'Absolute total margin difference: ${abs_diff:,.2f}\n')
    f.write(f'\n')
    f.write(f'Recommendation: **{decision}**\n')
    f.write(f'\n')
    f.write(f'{justification}\n')

print('Markdown written successfully.')
```

3. **Run the script:**
```bash
python3 /root/solve.py
```

4. **Validate the outputs:**
```bash
cat /root/vaxcrate_analysis.json
cat /root/vaxcrate_summary.md
```

5. **Check the JSON is valid and the markdown has the required content:**
   - Verify JSON parses correctly with `python3 -c "import json; d=json.load(open('/root/vaxcrate_analysis.json')); print('campaigns:', len(d['campaigns'])); print('decision:', d['recommendation']['decision'])"`
   - Verify markdown has at least 4 non-empty lines: `grep -c '.' /root/vaxcrate_summary.md`
   - Verify markdown contains the exact decision slug: `grep -E 'move_to_12_day|keep_6_day' /root/vaxcrate_summary.md`
   - Verify markdown contains comma-formatted currency values: `grep -oP '\$[\d,]+\.\d{2}' /root/vaxcrate_summary.md`

6. **Important edge cases to watch for after inspecting the data:**
   - If `campaign_manifest.json` has boolean `analysis_flag` values (true/false) instead of strings, the script handles both.
   - If `active_clinics` or `revision` in overrides are non-numeric strings, they are skipped.
   - The crate cost formula is `crate_cost_usd * dispatches_per_year` (per-dispatch cost, NOT per-clinic). Inspect the data to verify this interpretation. If the numbers look off or the crate_cost values are very small (suggesting per-crate-per-clinic), then adjust to `crate_cost_usd * active_clinics * dispatches_per_year`. Check the schema carefully — the output schema has `annual_crate_cost_6_day_usd` as a single number, not per-clinic, so consider which interpretation matches.
   - Currency values in the markdown MUST use thousands separators (e.g., `$42,908.83` not `$42908.83`). The script uses Python's `:,.2f` format which handles this.
   - `alias_labels` could be a list or comma-separated string — handled in the script.

7. **If the crate cost interpretation seems wrong after inspecting data**, re-examine whether annual_crate_cost should be `crate_cost_usd * dispatches_per_year` or `crate_cost_usd * active_clinics * dispatches_per_year`. The task says "Crate cost uses crate_cost_usd from crate_cost.csv" but doesn't explicitly state the annual formula. Look at the output schema and reason about what makes sense. If there's no explicit annual_crate_cost formula in the instructions, check if there's a pattern from similar tasks (the cross-task artifact mentions `carrier_cost_usd * active_labs * runs_per_year`). Adjust accordingly and re-run.

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