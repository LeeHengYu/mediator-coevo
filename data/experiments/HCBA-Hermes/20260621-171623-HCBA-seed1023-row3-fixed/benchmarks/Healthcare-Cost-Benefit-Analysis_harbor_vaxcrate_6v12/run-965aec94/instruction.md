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

2. **Write and run a Python script** at `/root/solve.py` that performs the full analysis. The script must:

 a. **Load inputs:**
 - `campaign_manifest.json` (JSON)
 - `crate_cost.csv` (CSV)
 - `billing.csv` (CSV)
 - `location_overrides.csv` (CSV)
 - `suspensions.csv` (CSV)

 b. **Filter campaigns:**
 - From the manifest, keep only campaigns where `analysis_flag == "review"`.
 - Exclude any campaign whose `campaign_id` appears in `suspensions.csv` with `suspension_status == "hold"`.

 c. **Resolve billing rows:**
 - For each retained campaign, find rows in `billing.csv` where `campaign_label` matches either the campaign's `campaign_name` OR any entry in its `alias_labels` list.
 - Keep only billing rows with `status == "active"`.
 - If multiple active rows match the same campaign, keep the one with the latest (lexicographically greatest) `cycle_tag`.
 - Extract `payment_per_dispatch_per_clinic_usd` from the retained billing row.

 d. **Resolve active clinics from location_overrides.csv:**
 - Filter to rows where `state == "approved"`.
 - Ignore rows where `revision` is blank/empty or `active_clinics` is blank/empty.
 - Among valid approved rows for the same `campaign_id`, keep the one with the highest numeric `revision`.
 - If no valid override row exists for a campaign, use `default_active_clinics` from the manifest.

 e. **Look up crate cost:**
 - Match each campaign's `crate_tier` (from manifest) to `crate_cost.csv` to get `crate_cost_usd`.

 f. **Compute per-campaign metrics (all rounded to 2 decimals):**
 - 6-day model: 6 days/dispatch, 60 dispatches/year
 - 12-day model: 12 days/dispatch, 30 dispatches/year
 - `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
 - `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
 - `annual_crate_cost = crate_cost_usd * dispatches_per_year` (NOTE: this is the crate cost per dispatch times number of dispatches — verify from the data whether crate_cost_usd is per-dispatch or annual; if the CSV has a single cost value, assume it is per dispatch)
 - `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
 - `annual_margin_difference_12_minus_6 = annual_margin_12_day - annual_margin_6_day`
 - Round each of these to 2 decimal places.

 g. **Compute totals:**
 - `total_annual_margin_6_day_usd` = sum of all campaign `annual_margin_6_day_usd`
 - `total_annual_margin_12_day_usd` = sum of all campaign `annual_margin_12_day_usd`
 - `total_annual_margin_difference_12_minus_6_usd` = sum of all per-campaign differences
 - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_12_minus_6_usd)
 - Round all to 2 decimals.

 h. **Decision rule:**
 - If `absolute_total_margin_difference_usd < 11000`, decision = `"move_to_12_day"`
 - Otherwise, decision = `"keep_6_day"`
 - Write a justification string that mentions the absolute difference and threshold.

 i. **Sort campaigns** by `campaign_id` ascending.

 j. **Write `/root/vaxcrate_analysis.json`** with the exact schema from the task (including the `assumptions` block with all specified keys and values). Use `json.dump` with `indent=2`.

 k. **Write `/root/vaxcrate_summary.md`** with 4-8 non-empty lines containing:
 - Total 6-day margin in USD (comma-formatted, e.g., `1,234.56`)
 - Total 12-day margin in USD (comma-formatted)
 - Absolute difference in USD (comma-formatted)
 - The exact decision slug (`move_to_12_day` or `keep_6_day`)

 **IMPORTANT formatting notes for the summary:**
 - Use comma-formatted numbers (e.g., `12,345.67` not `12345.67`). Use Python's `"{:,.2f}".format(value)` for this.
 - Include the exact slug string in the text.

3. **Run the script:**
 ```
 cd /root && python solve.py
 ```

4. **Validate outputs:**
 ```
 cat /root/vaxcrate_analysis.json
 cat /root/vaxcrate_summary.md
 python -c "import json; d=json.load(open('/root/vaxcrate_analysis.json')); print('Keys in assumptions:', list(d['assumptions'].keys())); print('Num campaigns:', len(d['campaigns'])); print('Totals:', d['totals']); print('Recommendation:', d['recommendation'])"
 ```

5. **Run the verifier if available:**
 ```
 ls /root/test_output.py && cd /root && python -m pytest test_output.py -v
 ```

**Key pitfalls to avoid (from cross-task feedback):**
- Ensure the `assumptions` block has exactly the keys specified in the schema — no extra keys, no missing keys.
- Ensure `recommendation` contains a `justification` key (string).
- Use comma-formatted currency strings in the markdown summary (e.g., `-7,106.39` not `-7106.39`).
- Make sure every field in the campaign objects matches the schema exactly (no extra fields, no missing fields).
- The `annual_crate_cost` formula: carefully check what `crate_cost_usd` represents. Read the CSV header and values. If it's a per-crate or per-dispatch cost, multiply by dispatches_per_year. If it's already annual, don't multiply. Inspect the data first.
- When matching `campaign_label` from billing to campaigns, handle `alias_labels` which may be a list in the JSON manifest. Ensure you check membership correctly.
- For `cycle_tag` comparison to find the latest, treat it as a string and use lexicographic ordering (or parse if it has a date-like format — inspect the data first).

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