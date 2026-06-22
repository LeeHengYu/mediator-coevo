# Task Instruction

Execute the following steps in order:

1. **Read all input files** to understand their structure:
   - `cat /root/campaign_manifest.json`
   - `cat /root/crate_cost.csv`
   - `cat /root/billing.csv`
   - `cat /root/location_overrides.csv`
   - `cat /root/suspensions.csv`

2. **Read the test file** to understand exact validation expectations:
   - `cat /root/tests/test_output.py` (or find it with `find /root/tests -name '*.py'`)

3. **Write a Python script** `/root/solve.py` that does the following:

   a. Load `campaign_manifest.json`. It contains an array of campaign objects. Each has at minimum: `campaign_id`, `campaign_name`, `analysis_flag`, `alias_labels` (list of strings), `default_active_clinics`, `drug_cost_per_1000_doses_usd`, `doses_per_day`, `crate_tier`.

   b. Filter to only campaigns where `analysis_flag == "review"`.

   c. Load `suspensions.csv`. Exclude any campaign whose `campaign_id` appears in suspensions with `suspension_status == "hold"`.

   d. Load `billing.csv`. For each retained campaign, find billing rows where `campaign_label` matches either the campaign's `campaign_name` or any value in its `alias_labels`. Keep only rows with `status == "active"`. If multiple active rows match the same campaign, keep the one with the latest `cycle_tag` (compare as strings — they are likely formatted like "2024-Q3" or similar; sort lexicographically and take the max).

   e. Load `location_overrides.csv`. For each retained campaign, find rows matching by `campaign_id` where `state == "approved"`. Discard rows where `revision` is blank/empty or `active_clinics` is blank/empty. Among remaining valid rows, keep the one with the highest numeric `revision`. Use its `active_clinics` (as integer). If no valid override row exists, use `default_active_clinics` from the manifest.

   f. Load `crate_cost.csv`. Build a lookup from `crate_tier` to `crate_cost_usd`.

   g. For each retained campaign, compute:
      - `crate_cost_usd` = lookup by `crate_tier`
      - `payment_per_dispatch_per_clinic_usd` = from the retained billing row
      - **6-day model** (days_per_dispatch=6, dispatches_per_year=60):
        - `annual_revenue_6_day_usd = payment_per_dispatch_per_clinic_usd * active_clinics * 60`
        - `annual_drug_cost_6_day_usd = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * 6 * 60 / 1000`
        - `annual_crate_cost_6_day_usd = crate_cost_usd * 60`
        - `annual_margin_6_day_usd = annual_revenue_6_day_usd - annual_drug_cost_6_day_usd - annual_crate_cost_6_day_usd`
      - **12-day model** (days_per_dispatch=12, dispatches_per_year=30):
        - `annual_revenue_12_day_usd = payment_per_dispatch_per_clinic_usd * active_clinics * 30`
        - `annual_drug_cost_12_day_usd = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * 12 * 30 / 1000`
        - `annual_crate_cost_12_day_usd = crate_cost_usd * 30`
        - `annual_margin_12_day_usd = annual_revenue_12_day_usd - annual_drug_cost_12_day_usd - annual_crate_cost_12_day_usd`
      - `annual_margin_difference_12_minus_6_usd = annual_margin_12_day_usd - annual_margin_6_day_usd`
      - Round ALL currency values to 2 decimal places.

   h. Sort campaigns by `campaign_id` ascending.

   i. Compute totals:
      - `total_annual_margin_6_day_usd` = sum of all `annual_margin_6_day_usd`
      - `total_annual_margin_12_day_usd` = sum of all `annual_margin_12_day_usd`
      - `total_annual_margin_difference_12_minus_6_usd` = sum of all `annual_margin_difference_12_minus_6_usd`
      - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_12_minus_6_usd)
      - Round all to 2 decimals.

   j. Decision:
      - If `absolute_total_margin_difference_usd < 11000`, decision = `"move_to_12_day"`
      - Otherwise, decision = `"keep_6_day"`

   k. Build the output JSON with this **exact** structure (match field names precisely):
   ```json
   {
     "assumptions": {
       "dispatches_per_year_6_day": 60,
       "dispatches_per_year_12_day": 30,
       "days_per_dispatch_6_day": 6,
       "days_per_dispatch_12_day": 12,
       "switch_threshold_usd": 11000,
       "override_rule": "highest numeric approved revision with non-empty active_clinics, else default_active_clinics",
       "suspension_rule": "exclude hold campaigns"
     },
     "campaigns": [ ... ],
     "totals": {
       "total_annual_margin_6_day_usd": ...,
       "total_annual_margin_12_day_usd": ...,
       "total_annual_margin_difference_12_minus_6_usd": ...,
       "absolute_total_margin_difference_usd": ...
     },
     "recommendation": {
       "decision": "move_to_12_day" or "keep_6_day",
       "justification": "<a brief sentence explaining the decision>"
     }
   }
   ```
   Each campaign object must have exactly these fields (no extras, no missing):
   `campaign_id`, `campaign_name`, `active_clinics`, `drug_cost_per_1000_doses_usd`, `doses_per_day`, `crate_tier`, `crate_cost_usd`, `payment_per_dispatch_per_clinic_usd`, `annual_drug_cost_6_day_usd`, `annual_drug_cost_12_day_usd`, `annual_crate_cost_6_day_usd`, `annual_crate_cost_12_day_usd`, `annual_revenue_6_day_usd`, `annual_revenue_12_day_usd`, `annual_margin_6_day_usd`, `annual_margin_12_day_usd`, `annual_margin_difference_12_minus_6_usd`

   l. Write the JSON to `/root/vaxcrate_analysis.json` with `indent=2`.

   m. Write `/root/vaxcrate_summary.md` with 4-8 non-empty lines including:
      - Total 6-day margin in USD
      - Total 12-day margin in USD
      - Absolute difference in USD
      - The exact decision slug (`move_to_12_day` or `keep_6_day`)

4. **Run the script**: `python /root/solve.py`

5. **Validate the output**:
   - `cat /root/vaxcrate_analysis.json` — verify the structure matches the schema exactly
   - `cat /root/vaxcrate_summary.md` — verify 4-8 non-empty lines with required content

6. **Run the tests**:
   - `cd /root && python -m pytest tests/ -v` (or wherever the test file is located)
   - If any tests fail, read the error messages carefully, fix the script, re-run, and re-test.

**Critical reminders from previous failure:**
- The `assumptions` key MUST be present at the root level of the JSON.
- Campaign objects must use the EXACT field names from the schema (e.g., `annual_drug_cost_6_day_usd`, NOT `six_day` sub-objects).
- Totals must use exact keys: `total_annual_margin_6_day_usd`, `total_annual_margin_12_day_usd`, `total_annual_margin_difference_12_minus_6_usd`, `absolute_total_margin_difference_usd`.
- Do NOT include extra fields like `billing_cycle_tag`, `matched_billing_label`, `active_labs_source`, etc. in campaign objects.
- All numeric currency values must be rounded to exactly 2 decimal places (use `round(value, 2)`).
- Ensure `doses_per_day` and `drug_cost_per_1000_doses_usd` are treated as floats.

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