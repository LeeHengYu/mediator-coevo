# Task Instruction

Execute the following steps in order to produce `/root/vaxcrate_analysis.json` and `/root/vaxcrate_summary.md`.

## Step 0 — Inspect all input files

```bash
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

Also inspect the test file if it exists:
```bash
cat /root/tests/test_output.py 2>/dev/null || cat /root/test_output.py 2>/dev/null || echo 'no test file found'
```

## Step 1 — Write and run the Python script

Create `/root/solve.py` with the following logic. Read every input file, then:

### 1a. Load data
- `campaign_manifest.json` — list of campaign objects.
- `crate_cost.csv` — map `crate_tier` → `crate_cost_usd`.
- `billing.csv` — rows with `campaign_label`, `status`, `cycle_tag`, `payment_per_dispatch_per_clinic_usd`.
- `location_overrides.csv` — rows with `campaign_id`, `state`, `revision`, `active_clinics`.
- `suspensions.csv` — rows with `campaign_id`, `suspension_status`.

### 1b. Filter campaigns
- Keep only campaigns where `analysis_flag == 'review'`.
- Exclude any campaign whose `campaign_id` appears in `suspensions.csv` with `suspension_status == 'hold'`.

### 1c. Resolve billing rows
For each retained campaign:
- Match billing rows where `campaign_label` equals `campaign_name` OR `campaign_label` is in the campaign's `alias_labels` list.
- Keep only rows with `status == 'active'`.
- If multiple active rows match, keep the one with the **latest** `cycle_tag` (lexicographic or date comparison — inspect the data to decide).
- Extract `payment_per_dispatch_per_clinic_usd` from the kept row.

### 1d. Resolve active clinics
For each retained campaign:
- From `location_overrides.csv`, select rows matching `campaign_id` where `state == 'approved'`.
- **Discard** rows where `revision` is blank/empty or `active_clinics` is blank/empty.
- Among remaining rows, keep the one with the **highest numeric** `revision`.
- Use its `active_clinics` (convert to int/float).
- If no valid override row exists, fall back to `default_active_clinics` from the manifest.

### 1e. Compute per-campaign values
For each retained campaign, using:
- `drug_cost_per_1000_doses_usd` and `doses_per_day` from the manifest.
- `crate_cost_usd` from `crate_cost.csv` matched by the campaign's `crate_tier`.
- `payment_per_dispatch_per_clinic_usd` from the billing row.
- `active_clinics` from step 1d.

**6-day model** (days_per_dispatch=6, dispatches_per_year=60):
```
annual_revenue_6 = payment_per_dispatch_per_clinic_usd * active_clinics * 60
annual_drug_cost_6 = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * 6 * 60 / 1000
annual_crate_cost_6 = crate_cost_usd * 60
```

**12-day model** (days_per_dispatch=12, dispatches_per_year=30):
```
annual_revenue_12 = payment_per_dispatch_per_clinic_usd * active_clinics * 30
annual_drug_cost_12 = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * 12 * 30 / 1000
annual_crate_cost_12 = crate_cost_usd * 30
```

⚠️ **CRITICAL**: `annual_crate_cost` is `crate_cost_usd * dispatches_per_year` — do **NOT** multiply by `active_clinics`. This was the bug in the previous run.

```
annual_margin_6 = annual_revenue_6 - annual_drug_cost_6 - annual_crate_cost_6
annual_margin_12 = annual_revenue_12 - annual_drug_cost_12 - annual_crate_cost_12
difference = annual_margin_12 - annual_margin_6
```

Round all currency values to 2 decimal places.

### 1f. Totals and decision
```
total_margin_6 = sum of all annual_margin_6
total_margin_12 = sum of all annual_margin_12
total_difference = sum of all per-campaign differences
absolute_difference = abs(total_difference)
```

Decision rule:
- If `abs(total_difference) < 11000` → `move_to_12_day`
- Otherwise → `keep_6_day`

### 1g. Output JSON
Write `/root/vaxcrate_analysis.json` with the exact schema from the task. Sort the `campaigns` array by `campaign_id` ascending. Include the `assumptions` block exactly as specified. Include a `justification` string that mentions the absolute difference and threshold.

### 1h. Output summary
Write `/root/vaxcrate_summary.md` with 4–8 non-empty lines including:
- Total 6-day margin (USD)
- Total 12-day margin (USD)
- Absolute difference (USD)
- Final decision using the exact slug `move_to_12_day` or `keep_6_day`

## Step 2 — Run the script
```bash
cd /root && python solve.py
```

## Step 3 — Validate outputs
```bash
cat /root/vaxcrate_analysis.json
cat /root/vaxcrate_summary.md
```

Check:
- JSON is valid and parseable.
- `campaigns` array is sorted by `campaign_id`.
- All required keys exist with correct names (including `_usd` suffixes).
- `annual_crate_cost` values are NOT multiplied by `active_clinics`.
- Verify one campaign's numbers by hand if feasible.

## Step 4 — Run tests
```bash
cd /root && python -m pytest tests/test_output.py -v 2>/dev/null || python -m pytest test_output.py -v 2>/dev/null || echo 'no standard test path found'
```

If tests fail, read the error messages carefully, fix the script, re-run, and re-validate. Pay special attention to:
- Value mismatches (check if `active_clinics` override logic is correct)
- Key errors (check field name spelling)
- Ordering issues (ensure sort by `campaign_id`)

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