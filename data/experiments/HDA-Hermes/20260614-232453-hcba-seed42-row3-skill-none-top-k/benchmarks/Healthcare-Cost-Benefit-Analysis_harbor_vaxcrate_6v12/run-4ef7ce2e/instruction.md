# Task Instruction

Execute the following steps carefully and in order.

## 1. Inspect all input files

```bash
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

Read and understand the structure and content of each file before writing any code.

## 2. Write and run a Python script

Create `/root/solve.py` with the following logic. Follow every rule precisely.

### Step-by-step logic

1. **Load data:**
   - `campaign_manifest.json` — list of campaign objects.
   - `crate_cost.csv` — columns include `crate_tier` and `crate_cost_usd`.
   - `billing.csv` — columns include `campaign_label`, `status`, `cycle_tag`, `payment_per_dispatch_per_clinic_usd`.
   - `location_overrides.csv` — columns include `campaign_id`, `state`, `revision`, `active_clinics`.
   - `suspensions.csv` — columns include `campaign_id`, `suspension_status`.

2. **Filter campaigns:**
   - From the manifest, keep only campaigns where `analysis_flag == "review"`.
   - From `suspensions.csv`, collect all `campaign_id` values where `suspension_status == "hold"`. Exclude any campaign whose `campaign_id` is in that set.

3. **Resolve billing rows:**
   - For each retained campaign, find billing rows where `campaign_label` matches either the campaign's `campaign_name` OR any entry in its `alias_labels` list.
   - Keep only billing rows where `status == "active"`.
   - If multiple active rows match, keep the one with the latest (lexicographically largest) `cycle_tag`.
   - Extract `payment_per_dispatch_per_clinic_usd` from the retained billing row.

4. **Resolve active clinics:**
   - From `location_overrides.csv`, filter rows where `state == "approved"`.
   - Among those, discard rows where `revision` is blank/empty/NaN OR `active_clinics` is blank/empty/NaN.
   - For each `campaign_id`, if multiple valid approved rows exist, keep the one with the highest numeric `revision`.
   - For each retained campaign: if a valid override row exists, use its `active_clinics`. Otherwise, use `default_active_clinics` from the manifest.

5. **Resolve crate cost:**
   - Match each campaign's `crate_tier` (from manifest) to `crate_cost.csv` to get `crate_cost_usd`.

6. **Compute per-campaign figures (all rounded to 2 decimals at the end):**

   For 6-day model: `days_per_dispatch=6`, `dispatches_per_year=60`
   For 12-day model: `days_per_dispatch=12`, `dispatches_per_year=30`

   - `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
   - `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
   
   **CRITICAL — Crate cost formula:** The crate cost is per crate per dispatch. Each clinic needs a crate per dispatch. Therefore:
   - `annual_crate_cost = crate_cost_usd * active_clinics * dispatches_per_year`
   
   This is the key fix from the previous failed run. The prior attempt used `crate_cost_usd * dispatches_per_year` without multiplying by `active_clinics`. The expected values (e.g., 7956 vs 468) confirm the crate cost scales with clinics. Verify: if crate_cost_usd is about 2.21 and active_clinics is 60 and dispatches is 60, then 2.21*60*60 = 7956, which matches the expected value.

   - `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
   - `margin_difference = annual_margin_12_day - annual_margin_6_day`

   Round all USD values to 2 decimal places.

7. **Compute totals:**
   - `total_annual_margin_6_day_usd` = sum of all campaigns' `annual_margin_6_day_usd`
   - `total_annual_margin_12_day_usd` = sum of all campaigns' `annual_margin_12_day_usd`
   - `total_annual_margin_difference_12_minus_6_usd` = sum of all per-campaign differences
   - `absolute_total_margin_difference_usd` = abs(total_difference)
   Round all to 2 decimals.

8. **Decision:**
   - If `abs(total_difference) < 11000`: decision = `"move_to_12_day"`
   - Otherwise: decision = `"keep_6_day"`
   - Write a brief justification string.

9. **Sort campaigns array** by `campaign_id` ascending.

10. **Write `/root/vaxcrate_analysis.json`** with the exact schema specified. Use `json.dump` with `indent=2`. Ensure all numeric currency fields are floats rounded to 2 decimals.

11. **Write `/root/vaxcrate_summary.md`** with 4-8 non-empty lines containing:
    - Total 6-day margin (USD) — use the number without comma formatting
    - Total 12-day margin (USD)
    - Absolute difference (USD)
    - The exact decision slug (`move_to_12_day` or `keep_6_day`)

## 3. Run the script

```bash
python3 /root/solve.py
```

## 4. Validate outputs

```bash
cat /root/vaxcrate_analysis.json
cat /root/vaxcrate_summary.md
python3 -c "import json; d=json.load(open('/root/vaxcrate_analysis.json')); print('Campaigns:', len(d['campaigns'])); print('IDs:', [c['campaign_id'] for c in d['campaigns']]); print('Totals:', d['totals']); print('Decision:', d['recommendation']['decision'])"
```

## 5. Spot-check crate cost calculation

For the first campaign in the output, manually verify:
- `annual_crate_cost_6_day = crate_cost_usd * active_clinics * 60`
- `annual_crate_cost_12_day = crate_cost_usd * active_clinics * 30`

Print these values and confirm they match the JSON output. If they don't match, debug and fix.

## 6. Check for the verifier test file and run it if present

```bash
ls /root/test_output.py 2>/dev/null && python3 -m pytest /root/test_output.py -v || echo 'No test file found'
```

If tests fail, read the error messages carefully, fix the issue in solve.py, re-run, and re-verify.

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