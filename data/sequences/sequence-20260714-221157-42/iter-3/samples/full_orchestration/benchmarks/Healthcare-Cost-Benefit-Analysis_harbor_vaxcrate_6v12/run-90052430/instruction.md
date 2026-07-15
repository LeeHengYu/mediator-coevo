# Task Instruction

Execute the following steps in order:

## 1. Inspect all input files

Read and display the full contents of:
- `/root/campaign_manifest.json`
- `/root/crate_cost.csv`
- `/root/billing.csv`
- `/root/location_overrides.csv`
- `/root/suspensions.csv`

## 2. Build a Python script at `/root/solve.py` that does the following:

```python
import json, csv, math

# Load all input files
with open('/root/campaign_manifest.json') as f:
    manifest = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

crate_cost_rows = read_csv('/root/crate_cost.csv')
billing_rows = read_csv('/root/billing.csv')
override_rows = read_csv('/root/location_overrides.csv')
suspension_rows = read_csv('/root/suspensions.csv')

# Build crate cost lookup: crate_tier -> crate_cost_usd
crate_cost_map = {}
for row in crate_cost_rows:
    crate_cost_map[row['crate_tier'].strip()] = float(row['crate_cost_usd'].strip())

# Build suspension set: campaign_ids on hold
hold_set = set()
for row in suspension_rows:
    if row['suspension_status'].strip().lower() == 'hold':
        hold_set.add(row['campaign_id'].strip())

# Process campaigns from manifest
# manifest could be a dict with a 'campaigns' key or could be a list
if isinstance(manifest, dict) and 'campaigns' in manifest:
    campaigns_list = manifest['campaigns']
elif isinstance(manifest, list):
    campaigns_list = manifest
else:
    # Try to figure out the structure
    campaigns_list = manifest

# Filter: analysis_flag == 'review' and not in hold_set
retained = []
for c in campaigns_list:
    if c.get('analysis_flag', '').strip().lower() != 'review':
        continue
    if c['campaign_id'].strip() in hold_set:
        continue
    retained.append(c)

# Build alias lookup: label -> campaign_id
# campaign_name -> campaign_id, alias_labels entries -> campaign_id
label_to_cid = {}
for c in retained:
    cid = c['campaign_id'].strip()
    label_to_cid[c['campaign_name'].strip()] = cid
    aliases = c.get('alias_labels', [])
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(',')]
    for a in aliases:
        label_to_cid[a.strip()] = cid

# Resolve billing rows
# Only active status, match campaign_label to label_to_cid
# Group by campaign_id, keep latest cycle_tag
billing_by_cid = {}
for row in billing_rows:
    if row['status'].strip().lower() != 'active':
        continue
    label = row['campaign_label'].strip()
    cid = label_to_cid.get(label)
    if cid is None:
        continue
    cycle_tag = row['cycle_tag'].strip()
    if cid not in billing_by_cid or cycle_tag > billing_by_cid[cid]['cycle_tag'].strip():
        billing_by_cid[cid] = row

# Resolve location overrides
# approved state, non-empty revision and active_clinics
# Group by campaign_id, keep highest numeric revision
override_by_cid = {}
for row in override_rows:
    if row['state'].strip().lower() != 'approved':
        continue
    rev = row.get('revision', '').strip()
    ac = row.get('active_clinics', '').strip()
    if rev == '' or ac == '':
        continue
    cid = row['campaign_id'].strip()
    rev_num = float(rev)
    if cid not in override_by_cid or rev_num > override_by_cid[cid][0]:
        override_by_cid[cid] = (rev_num, int(float(ac)))

# Build output campaigns
output_campaigns = []
for c in retained:
    cid = c['campaign_id'].strip()
    cname = c['campaign_name'].strip()
    
    # Active clinics
    if cid in override_by_cid:
        active_clinics = override_by_cid[cid][1]
    else:
        active_clinics = int(float(c['default_active_clinics']))
    
    drug_cost_per_1000 = float(c['drug_cost_per_1000_doses_usd'])
    doses_per_day = float(c['doses_per_day'])
    crate_tier = c['crate_tier'].strip()
    crate_cost_usd = crate_cost_map[crate_tier]
    
    billing_row = billing_by_cid[cid]
    payment_per_dispatch = float(billing_row['payment_per_dispatch_per_clinic_usd'].strip())
    
    # 6-day model
    days6 = 6; disp6 = 60
    # 12-day model
    days12 = 12; disp12 = 30
    
    rev_6 = payment_per_dispatch * active_clinics * disp6
    rev_12 = payment_per_dispatch * active_clinics * disp12
    
    drug_6 = drug_cost_per_1000 * active_clinics * doses_per_day * days6 * disp6 / 1000
    drug_12 = drug_cost_per_1000 * active_clinics * doses_per_day * days12 * disp12 / 1000
    
    crate_6 = crate_cost_usd * disp6
    crate_12 = crate_cost_usd * disp12
    
    margin_6 = rev_6 - drug_6 - crate_6
    margin_12 = rev_12 - drug_12 - crate_12
    diff = margin_12 - margin_6
    
    output_campaigns.append({
        'campaign_id': cid,
        'campaign_name': cname,
        'active_clinics': active_clinics,
        'drug_cost_per_1000_doses_usd': round(drug_cost_per_1000, 2),
        'doses_per_day': round(doses_per_day, 2),
        'crate_tier': crate_tier,
        'crate_cost_usd': round(crate_cost_usd, 2),
        'payment_per_dispatch_per_clinic_usd': round(payment_per_dispatch, 2),
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
output_campaigns.sort(key=lambda x: x['campaign_id'])

# Totals
total_margin_6 = sum(c['annual_margin_6_day_usd'] for c in output_campaigns)
total_margin_12 = sum(c['annual_margin_12_day_usd'] for c in output_campaigns)
total_diff = sum(c['annual_margin_difference_12_minus_6_usd'] for c in output_campaigns)
abs_diff = abs(total_diff)

if abs_diff < 11000:
    decision = 'move_to_12_day'
    justification = f'Absolute total margin difference ${round(abs_diff,2)} is below the $11,000 threshold, so consolidating to 12-day dispatches is recommended.'
else:
    decision = 'keep_6_day'
    justification = f'Absolute total margin difference ${round(abs_diff,2)} exceeds the $11,000 threshold, so keeping 6-day dispatches is recommended.'

result = {
    'assumptions': {
        'dispatches_per_year_6_day': 60,
        'dispatches_per_year_12_day': 30,
        'days_per_dispatch_6_day': 6,
        'days_per_dispatch_12_day': 12,
        'switch_threshold_usd': 11000,
        'override_rule': 'highest numeric approved revision with non-empty active_clinics, else default_active_clinics',
        'suspension_rule': 'exclude hold campaigns'
    },
    'campaigns': output_campaigns,
    'totals': {
        'total_annual_margin_6_day_usd': round(total_margin_6, 2),
        'total_annual_margin_12_day_usd': round(total_margin_12, 2),
        'total_annual_margin_difference_12_minus_6_usd': round(total_diff, 2),
        'absolute_total_margin_difference_usd': round(abs_diff, 2)
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/vaxcrate_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

# Write summary markdown
lines = [
    '# VaxCrate Dispatch Analysis Summary',
    f'Total 6-day annual margin: ${round(total_margin_6, 2)}',
    f'Total 12-day annual margin: ${round(total_margin_12, 2)}',
    f'Absolute margin difference: ${round(abs_diff, 2)}',
    f'Decision: {decision}',
    justification
]
with open('/root/vaxcrate_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Outputs written.')
print(json.dumps(result, indent=2))
```

IMPORTANT: Before writing the script, first inspect all five input files carefully. Adapt the script if column names differ from what is assumed above (e.g., check exact CSV header names, JSON key names). The script above is a template — adjust field names to match the actual data.

## 3. Run the script

```bash
python3 /root/solve.py
```

## 4. Validate outputs

- Read `/root/vaxcrate_analysis.json` and verify:
  - The `assumptions` block matches exactly the required schema.
  - The `campaigns` array is sorted by `campaign_id` ascending.
  - All currency values are rounded to 2 decimal places.
  - The `recommendation.decision` is exactly `move_to_12_day` or `keep_6_day` (no other string).
  - The schema keys match exactly what's specified (no extra keys, no missing keys).
  - The `annual_crate_cost` fields use `crate_cost_usd * dispatches_per_year` (NOT multiplied by active_clinics).

- Read `/root/vaxcrate_summary.md` and verify:
  - It has 4-8 non-empty lines.
  - Contains total 6-day margin, total 12-day margin, absolute difference, and the exact decision slug.

## 5. Cross-check one campaign manually

Pick the first campaign in the output and manually verify:
- The active_clinics value (from override or default)
- The billing row selection (latest cycle_tag among active rows)
- The drug cost, crate cost, revenue, and margin calculations for both 6-day and 12-day
- The margin difference

If any discrepancy is found, fix and re-run.

## Key warnings from prior failures:
- Do NOT nest `decision` and `justification` outside of `recommendation` — they must be inside the `recommendation` object.
- Do NOT add extra keys to campaign objects beyond what the schema specifies.
- Double-check that crate cost is per-dispatch (not per-clinic-per-dispatch). The formula is: `crate_cost_usd * dispatches_per_year`. There is no multiplication by active_clinics for crate cost.
- Ensure `active_clinics` is an integer in the output.
- Ensure `doses_per_day` and `drug_cost_per_1000_doses_usd` are output as floats rounded to 2 decimals.

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