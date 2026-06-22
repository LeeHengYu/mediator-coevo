# Task Instruction

Execute the following steps in order:

1. **Read all input files** to understand the data:
   ```
   cat /root/program_catalog.json
   cat /root/cooler_cost.csv
   cat /root/contract_payment.csv
   cat /root/site_overrides.csv
   ```

2. **Read the test file** to understand exact verifier expectations:
   ```
   cat /root/test_output.py
   ```

3. **Write a Python script** `/root/solve.py` that does the following:

   a. Load all four input files (JSON for catalog, CSV for the rest).

   b. Filter `program_catalog.json` to only programs where `review_flag == "review"`. These are the in-scope programs.

   c. For each in-scope program, resolve `active_sites`:
      - Look in `site_overrides.csv` for rows matching the program's `program_code` where `approval_state == "approved"`.
      - If multiple approved rows exist for the same `program_code`, keep only the one with the highest `version_no`.
      - Use that row's `active_sites` value (check the actual column name in the CSV).
      - If no approved override row exists, use `default_active_sites` from `program_catalog.json`.

   d. For each in-scope program, resolve `payment_per_dispatch_per_site_usd`:
      - Match `contract_payment.csv` rows by checking if the row's `program_label` equals the program's `program_name` OR is contained in the program's `known_labels` list.
      - Ignore payment rows that don't map to any in-scope program.

   e. For each in-scope program, get `cooler_cost_usd` by matching the program's `cooler_type` to `cooler_cost.csv`.

   f. Compute per-program values (use the program's `acquisition_cost_per_1000_units_usd`, `units_per_day` from catalog):
      - `annual_drug_cost_10_day = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * 10 * 36 / 1000`
      - `annual_drug_cost_20_day = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * 20 * 18 / 1000`
      - `annual_cooler_cost_10_day = cooler_cost_usd * 36` (per year, NOT multiplied by active_sites — but CHECK the test expectations; the formula says `annual_cooler_cost` with no mention of sites, so try `cooler_cost_usd * dispatches_per_year` first; if test fails, try `cooler_cost_usd * active_sites * dispatches_per_year`)
      - `annual_cooler_cost_20_day = cooler_cost_usd * 18`
      - `annual_revenue_10_day = payment_per_dispatch_per_site_usd * active_sites * 36`
      - `annual_revenue_20_day = payment_per_dispatch_per_site_usd * active_sites * 18`
      - `annual_margin_10_day = annual_revenue_10_day - annual_drug_cost_10_day - annual_cooler_cost_10_day`
      - `annual_margin_20_day = annual_revenue_20_day - annual_drug_cost_20_day - annual_cooler_cost_20_day`
      - `annual_margin_difference_20_minus_10 = annual_margin_20_day - annual_margin_10_day`

   g. Round ALL currency values to 2 decimal places.

   h. Sort programs by `program_code` ascending.

   i. Compute totals:
      - `total_annual_margin_10_day_usd` = sum of all program `annual_margin_10_day_usd`
      - `total_annual_margin_20_day_usd` = sum of all program `annual_margin_20_day_usd`
      - `total_annual_margin_difference_20_minus_10_usd` = sum of all program `annual_margin_difference_20_minus_10_usd`
      - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_20_minus_10_usd)
      - Round all totals to 2 decimals.

   j. Decision:
      - If `absolute_total_margin_difference_usd < 10000`, decision = `"move_to_20_day"`
      - Otherwise, decision = `"keep_10_day"`

   k. Build the JSON output with **exactly** these keys and structure (no extra keys, no missing keys):

      The `assumptions` block must have **exactly 6 keys**:
      ```json
      {
        "dispatches_per_year_10_day": 36,
        "dispatches_per_year_20_day": 18,
        "days_per_dispatch_10_day": 10,
        "days_per_dispatch_20_day": 20,
        "switch_threshold_usd": 10000,
        "site_override_rule": "highest approved version_no per program_code, else default_active_sites"
      }
      ```

      Each program object must have **exactly these 18 keys** (no more, no less):
      - `program_code`, `program_name`, `active_sites`, `acquisition_cost_per_1000_units_usd`, `units_per_day`, `cooler_type`, `cooler_cost_usd`, `payment_per_dispatch_per_site_usd`, `annual_drug_cost_10_day_usd`, `annual_drug_cost_20_day_usd`, `annual_cooler_cost_10_day_usd`, `annual_cooler_cost_20_day_usd`, `annual_revenue_10_day_usd`, `annual_revenue_20_day_usd`, `annual_margin_10_day_usd`, `annual_margin_20_day_usd`, `annual_margin_difference_20_minus_10_usd`

      The `totals` block must have exactly 4 keys as specified.

      The `recommendation` block must have `decision` and `justification`.

   l. Write `/root/oncocooler_analysis.json` with `json.dump(..., indent=2)`.

   m. Write `/root/oncocooler_summary.md` with 4-8 non-empty lines including:
      - Total 10-day margin formatted as USD (e.g., `$1,234.56`)
      - Total 20-day margin formatted as USD
      - Absolute difference formatted as USD
      - The exact decision slug (`move_to_20_day` or `keep_10_day`)
      - Format currency with commas and 2 decimal places for the summary.

4. **Run the script**:
   ```
   cd /root && python solve.py
   ```

5. **Validate the output** by inspecting the JSON:
   ```
   cat /root/oncocooler_analysis.json
   cat /root/oncocooler_summary.md
   ```
   - Verify the `assumptions` block has exactly 6 keys.
   - Verify each program object has exactly 18 keys (including `units_per_day` and `acquisition_cost_per_1000_units_usd`, and NO `active_sites_source` or other extra keys).
   - Verify programs are sorted by `program_code`.

6. **Run the test suite**:
   ```
   cd /root && python -m pytest test_output.py -v
   ```

7. **If tests fail**, read the error messages carefully. Common issues to check:
   - If cooler cost values are wrong, the formula might need `cooler_cost_usd * active_sites * dispatches_per_year` instead of just `cooler_cost_usd * dispatches_per_year`. Check the test's expected values and adjust.
   - If schema tests fail, re-check that no extra keys snuck in and no required keys are missing.
   - Fix and re-run until all tests pass.

**CRITICAL SCHEMA RULES (from previous failure feedback):**
- Do NOT add extra keys to `assumptions` (no `active_sites_resolution`, `contract_payment_match`, `cooler_cost_formula`, etc.)
- Do NOT add extra keys to program objects (no `active_sites_source`)
- DO include `units_per_day` and `acquisition_cost_per_1000_units_usd` in every program object
- Match the schema exactly as specified

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