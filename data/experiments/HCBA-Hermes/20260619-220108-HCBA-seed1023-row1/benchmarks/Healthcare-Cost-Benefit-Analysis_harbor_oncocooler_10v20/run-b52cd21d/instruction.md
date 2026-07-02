# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure:
   ```
   cat /root/program_catalog.json
   cat /root/cooler_cost.csv
   cat /root/contract_payment.csv
   cat /root/site_overrides.csv
   ```

2. **Create `/root/solve.py`** with the following logic:

   a. Load `program_catalog.json`. Filter to only programs where `review_flag == "review"`. Build a lookup by `program_code`.

   b. Load `cooler_cost.csv`. Build a dict mapping `cooler_type` → `cooler_cost_usd`.

   c. Load `contract_payment.csv`. For each row, match its `program_label` to a catalog entry by checking if `program_label` equals `program_name` OR if `program_label` is in the `known_labels` list. Skip rows that don't match any in-scope program. Store `payment_per_dispatch_per_site_usd` keyed by `program_code`.

   d. Load `site_overrides.csv`. Filter to rows where `approval_state == "approved"`. Group by `program_code`, keep only the row with the highest `version_no`. Store active site counts by `program_code`.

   e. For each in-scope program, determine `active_sites`: use the site_overrides value if present, otherwise use `default_active_sites` from the catalog.

   f. For each in-scope program, compute (all values rounded to 2 decimals):
      - **annual_drug_cost** = `acquisition_cost_per_1000_units_usd * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000`
      - **annual_cooler_cost** = `cooler_cost_usd * active_sites * dispatches_per_year`  ← CRITICAL: cooler cost is per-site-per-dispatch
      - **annual_revenue** = `payment_per_dispatch_per_site_usd * active_sites * dispatches_per_year`
      - **annual_margin** = `annual_revenue - annual_drug_cost - annual_cooler_cost`
      - **annual_margin_difference_20_minus_10** = `margin_20 - margin_10`

      Compute these for both 10-day (10 days/dispatch, 36 dispatches/year) and 20-day (20 days/dispatch, 18 dispatches/year) models.

   g. Sort programs by `program_code` ascending.

   h. Compute totals:
      - `total_annual_margin_10_day_usd` = sum of all program 10-day margins
      - `total_annual_margin_20_day_usd` = sum of all program 20-day margins
      - `total_annual_margin_difference_20_minus_10_usd` = sum of all per-program differences
      - `absolute_total_margin_difference_usd` = abs(total_difference)

   i. Decision rule:
      - If `abs(total_difference) < 10000` → `"move_to_20_day"`
      - Otherwise → `"keep_10_day"`

   j. Write `/root/oncocooler_analysis.json` with the exact schema specified (including `assumptions`, `programs`, `totals`, `recommendation` with `decision` and `justification` keys). Round all currency values to 2 decimal places.

   k. Write `/root/oncocooler_summary.md` with 4-8 non-empty lines including:
      - Total 10-day margin (USD)
      - Total 20-day margin (USD)
      - Absolute difference (USD)
      - The exact decision slug (`move_to_20_day` or `keep_10_day`)

3. **Run the script:**
   ```
   cd /root && python solve.py
   ```

4. **Validate outputs:**
   - `cat /root/oncocooler_analysis.json` — verify JSON is valid, has `assumptions`, `programs` (sorted by program_code), `totals`, and `recommendation` keys.
   - `cat /root/oncocooler_summary.md` — verify 4-8 non-empty lines with required content.
   - Spot-check one program's cooler cost calculation: confirm it equals `cooler_cost_usd * active_sites * dispatches_per_year` (NOT just `cooler_cost_usd * dispatches_per_year`).

5. If any verifier or test script exists (e.g., `test*.py` or `verify*.py` in /root), run it and fix any failures before finishing.

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