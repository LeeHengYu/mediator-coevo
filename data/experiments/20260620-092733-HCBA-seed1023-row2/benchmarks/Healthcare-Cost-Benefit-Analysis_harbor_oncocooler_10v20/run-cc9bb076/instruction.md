# Task Instruction

Execute the following steps exactly:

1. **Read all input files** before any computation:
   ```
   cat /root/program_catalog.json
   cat /root/cooler_cost.csv
   cat /root/contract_payment.csv
   cat /root/site_overrides.csv
   ```

2. **Read the test file** to understand the exact verifier contract:
   ```
   cat /root/tests/test_output.py
   ```

3. **Write a Python script** `/root/solve.py` that does the following:

   a. Load all four input files (JSON via `json.load`, CSVs via `csv.DictReader`).

   b. Filter `program_catalog.json` to only programs where `review_flag == "review"`. These are the in-scope programs.

   c. For each in-scope program, resolve `active_sites`:
      - Look in `site_overrides.csv` for rows matching the program's `program_code` where `approval_state == "approved"`.
      - If multiple approved rows exist for the same `program_code`, keep the one with the highest `version_no` (compare as integers).
      - Use that row's `active_sites` (as int).
      - If no approved override row exists, use `default_active_sites` from the catalog.

   d. For each in-scope program, resolve `payment_per_dispatch_per_site_usd`:
      - In `contract_payment.csv`, match each row's `program_label` against the program's `program_name` OR any entry in the program's `known_labels` list.
      - Use the matched row's `payment_per_dispatch_per_site_usd` (as float).
      - Ignore payment rows that don't map to any in-scope program.

   e. For each in-scope program, get `cooler_cost_usd` by matching the program's `cooler_type` against `cooler_cost.csv`'s `cooler_type` column.

   f. Compute per-program values (all floats, rounded to 2 decimals at the end):
      - `annual_drug_cost_10_day = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * 10 * 36 / 1000`
      - `annual_drug_cost_20_day = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * 20 * 18 / 1000`
      - `annual_cooler_cost_10_day = cooler_cost_usd * 36`
      - `annual_cooler_cost_20_day = cooler_cost_usd * 18`
      - `annual_revenue_10_day = payment_per_dispatch_per_site_usd * active_sites * 36`
      - `annual_revenue_20_day = payment_per_dispatch_per_site_usd * active_sites * 18`
      - `annual_margin_10_day = annual_revenue_10_day - annual_drug_cost_10_day - annual_cooler_cost_10_day`
      - `annual_margin_20_day = annual_revenue_20_day - annual_drug_cost_20_day - annual_cooler_cost_20_day`
      - `annual_margin_difference_20_minus_10 = annual_margin_20_day - annual_margin_10_day`

   g. Build each program object with EXACTLY these keys (no extras, no missing):
      ```
      program_code, program_name, active_sites,
      acquisition_cost_per_1000_units_usd, units_per_day,
      cooler_type, cooler_cost_usd,
      payment_per_dispatch_per_site_usd,
      annual_drug_cost_10_day_usd, annual_drug_cost_20_day_usd,
      annual_cooler_cost_10_day_usd, annual_cooler_cost_20_day_usd,
      annual_revenue_10_day_usd, annual_revenue_20_day_usd,
      annual_margin_10_day_usd, annual_margin_20_day_usd,
      annual_margin_difference_20_minus_10_usd
      ```
      All USD values rounded to 2 decimals. Sort the programs list by `program_code` ascending.

   h. Compute totals with EXACTLY these keys:
      ```
      total_annual_margin_10_day_usd
      total_annual_margin_20_day_usd
      total_annual_margin_difference_20_minus_10_usd
      absolute_total_margin_difference_usd
      ```
      Where `absolute_total_margin_difference_usd = abs(total_annual_margin_difference_20_minus_10_usd)`. All rounded to 2 decimals.

   i. Decision rule:
      - If `absolute_total_margin_difference_usd < 10000`, decision = `"move_to_20_day"`
      - Otherwise, decision = `"keep_10_day"`
      - Include a `justification` string that mentions the absolute difference and the threshold.

   j. Build the `assumptions` object with EXACTLY these keys (no extras):
      ```
      dispatches_per_year_10_day: 36
      dispatches_per_year_20_day: 18
      days_per_dispatch_10_day: 10
      days_per_dispatch_20_day: 20
      switch_threshold_usd: 10000
      site_override_rule: "highest approved version_no per program_code, else default_active_sites"
      ```

   k. Write the complete JSON to `/root/oncocooler_analysis.json` with `json.dump(..., indent=2)`.

   l. Write `/root/oncocooler_summary.md` with 4-8 non-empty lines including:
      - Total 10-day margin (USD) with the numeric value
      - Total 20-day margin (USD) with the numeric value
      - Absolute difference (USD) with the numeric value
      - The exact decision slug (`move_to_20_day` or `keep_10_day`)

4. **Run the script:**
   ```
   cd /root && python solve.py
   ```

5. **Validate the outputs:**
   ```
   cat /root/oncocooler_analysis.json
   cat /root/oncocooler_summary.md
   ```
   Verify:
   - The JSON `assumptions` has exactly 6 keys as listed above.
   - Each program object has exactly 17 keys as listed above.
   - The `totals` object has exactly 4 keys as listed above.
   - The `recommendation` object has `decision` and `justification` keys.
   - All USD values are rounded to 2 decimal places.
   - Programs are sorted by `program_code`.
   - The summary has 4-8 non-empty lines with the required content.

6. **Run the verifier:**
   ```
   cd /root && python -m pytest tests/test_output.py -v
   ```

7. If any test fails, read the error carefully, fix the issue in `solve.py`, re-run, and re-verify. Pay special attention to:
   - Key naming (must match exactly, including suffixes like `_usd`)
   - No extra keys in any object
   - No missing keys
   - Correct rounding
   - Correct data type (int for active_sites, float for USD values)

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