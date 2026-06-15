# Task Instruction

Execute the following steps in order:

1. **Read all input files** to understand the data:
 ```
 cat /root/campaign_manifest.json
 cat /root/crate_cost.csv
 cat /root/billing.csv
 cat /root/location_overrides.csv
 cat /root/suspensions.csv
 ```

2. **Read the test file** to understand the exact verifier expectations:
 ```
 cat /root/test_output.py
 ```

3. **Write a Python script** `/root/solve.py` that does the following:

 a. Load all input files (JSON for manifest, CSV for the rest).

 b. Filter campaigns: keep only those with `analysis_flag == "review"`.

 c. Exclude campaigns whose `campaign_id` appears in `suspensions.csv` with `suspension_status == "hold"`.

 d. For each retained campaign, resolve its billing row:
 - Match `billing.csv` rows by checking if `campaign_label` equals `campaign_name` OR is found in the `alias_labels` list from the manifest.
 - Keep only rows where `status == "active"`.
 - If multiple active rows map to the same campaign, keep the one with the latest `cycle_tag` (string sort is fine if tags are like "2024-Q3"; otherwise parse appropriately — inspect the data first).

 e. For each retained campaign, resolve active clinics from `location_overrides.csv`:
 - Use only rows where `state == "approved"`.
 - Ignore rows where `revision` is blank/empty or `active_clinics` is blank/empty.
 - If multiple valid approved rows exist for the same `campaign_id`, keep the one with the highest numeric `revision`.
 - If no valid override row exists, use `default_active_clinics` from the manifest.

 f. Look up `crate_cost_usd` from `crate_cost.csv` by matching `crate_tier`.

 g. Compute for each campaign (using the exact formulas):
 - **6-day model**: days_per_dispatch=6, dispatches_per_year=60
 - **12-day model**: days_per_dispatch=12, dispatches_per_year=30
 - `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
 - `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
 - `annual_crate_cost = crate_cost_usd * dispatches_per_year` (NOTE: inspect test file to confirm whether crate cost is per dispatch or per dispatch per clinic — check the test expectations carefully; if the test multiplies by active_clinics, do so)
 - `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
 - `difference = annual_margin_12_day - annual_margin_6_day`

   **IMPORTANT about crate cost**: Re-read the task instruction — it says `crate_cost_usd` from `crate_cost.csv`. The formula is not explicitly stated for annual_crate_cost. Look at the test file for hints. A common pattern in these tasks is `annual_crate_cost = crate_cost_usd * dispatches_per_year` (no clinic multiplier). But verify by checking the test.

 h. Round all currency values to 2 decimal places.

 i. Sort campaigns by `campaign_id` ascending.

 j. Compute totals:
 - `total_annual_margin_6_day_usd`: sum of all campaign `annual_margin_6_day_usd`
 - `total_annual_margin_12_day_usd`: sum of all campaign `annual_margin_12_day_usd`
 - `total_annual_margin_difference_12_minus_6_usd`: sum of all per-campaign differences
 - `absolute_total_margin_difference_usd`: abs of total difference

 k. Decision:
 - If `abs(total_difference) < 11000`, decision = `"move_to_12_day"`
 - Otherwise, decision = `"keep_6_day"`

 l. Build the output JSON with **exactly** these keys and structure (no extra keys, no nested models):

   ```python
   output = {
     "assumptions": {
       "dispatches_per_year_6_day": 60,
       "dispatches_per_year_12_day": 30,
       "days_per_dispatch_6_day": 6,
       "days_per_dispatch_12_day": 12,
       "switch_threshold_usd": 11000,
       "override_rule": "highest numeric approved revision with non-empty active_clinics, else default_active_clinics",
       "suspension_rule": "exclude hold campaigns"
     },
     "campaigns": [ ... ],  # each with EXACTLY the flat keys from the schema
     "totals": {
       "total_annual_margin_6_day_usd": ...,
       "total_annual_margin_12_day_usd": ...,
       "total_annual_margin_difference_12_minus_6_usd": ...,
       "absolute_total_margin_difference_usd": ...
     },
     "recommendation": {
       "decision": "move_to_12_day" or "keep_6_day",
       "justification": "..."
     }
   }
   ```

   Each campaign object must have **exactly** these keys (no extras like `six_day_model`, `twelve_day_model`, `billing_cycle_tag`, `active_clinics_revision`, `active_clinics_source`, `region`):
   - `campaign_id`, `campaign_name`, `active_clinics`
   - `drug_cost_per_1000_doses_usd`, `doses_per_day`
   - `crate_tier`, `crate_cost_usd`
   - `payment_per_dispatch_per_clinic_usd`
   - `annual_drug_cost_6_day_usd`, `annual_drug_cost_12_day_usd`
   - `annual_crate_cost_6_day_usd`, `annual_crate_cost_12_day_usd`
   - `annual_revenue_6_day_usd`, `annual_revenue_12_day_usd`
   - `annual_margin_6_day_usd`, `annual_margin_12_day_usd`
   - `annual_margin_difference_12_minus_6_usd`

 m. Write `/root/vaxcrate_analysis.json` with `json.dump(..., indent=2)`.

 n. Write `/root/vaxcrate_summary.md` with 4-8 non-empty lines including:
   - Total 6-day margin (USD)
   - Total 12-day margin (USD)
   - Absolute difference (USD)
   - Final decision using the exact slug `move_to_12_day` or `keep_6_day`

4. **Run the script**:
 ```
 cd /root && python solve.py
 ```

5. **Validate the output** by inspecting the JSON:
 ```
 cat /root/vaxcrate_analysis.json
 cat /root/vaxcrate_summary.md
 ```

6. **Run the verifier tests**:
 ```
 cd /root && python -m pytest test_output.py -v
 ```

7. If any test fails, read the error carefully, fix the issue in `solve.py`, re-run, and re-test. Pay special attention to:
   - Extra or missing keys in any object
   - Key naming (must match exactly)
   - The `assumptions` object must have exactly the keys shown in the schema
   - The `totals` object must use `total_annual_margin_6_day_usd` etc.
   - Whether `annual_crate_cost` involves `active_clinics` or not (check the test)
   - Rounding to 2 decimals
   - Sort order of campaigns
   - The decision threshold is strict less-than (`< 11000`), not less-than-or-equal

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