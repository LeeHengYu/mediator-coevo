# Task Instruction

Execute the following steps in order:

1. **Read all input files** to understand the data:
   - `cat /root/program_catalog.json`
   - `cat /root/cooler_cost.csv`
   - `cat /root/contract_payment.csv`
   - `cat /root/site_overrides.csv`

2. **Write and run a Python script** `/root/solve.py` that performs the full analysis. The script must:

   a. Load all four input files (JSON for catalog, CSV for the rest).

   b. **Filter programs**: Only include programs from `program_catalog.json` where `review_flag == "review"`.

   c. **Resolve contract payments**: For each row in `contract_payment.csv`, match its `program_label` against either `program_name` or any entry in the `known_labels` list from the catalog. Ignore payment rows that don't map to any in-scope program. Each in-scope program should get exactly one matching payment row's `payment_per_dispatch_per_site_usd`.

   d. **Resolve active sites**: From `site_overrides.csv`, keep only rows where `approval_state == "approved"`. If multiple approved rows exist for the same `program_code`, keep the one with the highest `version_no`. Use that row's active site count. If no approved override row exists for a program, fall back to `default_active_sites` from the catalog.

   e. **Resolve cooler cost**: Match each program's `cooler_type` to `cooler_cost.csv` to get `cooler_cost_usd`.

   f. **Compute per-program financials** using these exact formulas:
      - `annual_drug_cost = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000`
      - `annual_cooler_cost = cooler_cost_usd * dispatches_per_year` (NOTE: cooler cost is per dispatch, NOT per site per dispatch — but re-check the data; if cooler_cost.csv has a single cost per cooler type, the annual cooler cost is `cooler_cost_usd * dispatches_per_year` — however, verify whether it should be multiplied by active_sites too by checking what makes the numbers consistent. The most natural reading given the formulas listed is: annual_cooler_cost = cooler_cost_usd * dispatches_per_year. But if test failures indicate otherwise, adjust.)
      - Actually, re-reading the instructions: the formulas only explicitly define annual_revenue and annual_drug_cost. For annual_cooler_cost, only `cooler_cost_usd` from `cooler_cost.csv` is mentioned. The annual margin formula is `annual_revenue - annual_drug_cost - annual_cooler_cost`. Since cooler_cost_usd is a per-dispatch cost and there are multiple dispatches per year, compute: `annual_cooler_cost = cooler_cost_usd * dispatches_per_year`. Do NOT multiply by active_sites unless the data/tests require it.
      - `annual_revenue = payment_per_dispatch_per_site_usd * active_sites * dispatches_per_year`
      - `annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost`
      - `annual_margin_difference = annual_margin_20_day - annual_margin_10_day`
      - 10-day model: days_per_dispatch=10, dispatches_per_year=36
      - 20-day model: days_per_dispatch=20, dispatches_per_year=18

   g. **Round all currency values to 2 decimal places.**

   h. **Sort programs by `program_code` ascending.**

   i. **Compute totals**:
      - `total_annual_margin_10_day_usd` = sum of all programs' `annual_margin_10_day_usd`
      - `total_annual_margin_20_day_usd` = sum of all programs' `annual_margin_20_day_usd`
      - `total_annual_margin_difference_20_minus_10_usd` = sum of all programs' `annual_margin_difference_20_minus_10_usd`
      - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_20_minus_10_usd)

   j. **Decision rule**:
      - If `absolute_total_margin_difference_usd < 10000`, decision = `"move_to_20_day"`
      - Otherwise, decision = `"keep_10_day"`
      - Include a brief justification string.

   k. **Write `/root/oncocooler_analysis.json`** with the exact schema specified (including the `assumptions` block with the exact keys and values shown, the `programs` array, `totals` object, and `recommendation` object with `decision` and `justification` keys).

   l. **Write `/root/oncocooler_summary.md`** with 4-8 non-empty lines containing:
      - Total 10-day margin in USD
      - Total 20-day margin in USD
      - Absolute difference in USD
      - The exact decision slug (`move_to_20_day` or `keep_10_day`)

3. **Run the script**: `python3 /root/solve.py`

4. **Validate outputs**:
   - `cat /root/oncocooler_analysis.json` and verify it parses as valid JSON with the correct structure.
   - `cat /root/oncocooler_summary.md` and verify it has 4-8 non-empty lines with required content.
   - Check if a test file exists: `ls /root/test_output.py` or similar, and if so run `cd /root && python3 -m pytest test_output.py -v` to verify.

5. **If tests fail**, read the error messages carefully, fix the script, re-run, and re-validate. Pay special attention to:
   - The `recommendation` key must exist at the top level of the JSON (not nested differently).
   - All field names must match the schema exactly.
   - Currency rounding to 2 decimals.
   - The cooler cost formula — if tests fail on margin values, try the alternative: `annual_cooler_cost = cooler_cost_usd * active_sites * dispatches_per_year` and see if that fixes it.
   - Sort order must be by `program_code` ascending.
   - The `assumptions` block must have the exact keys and values shown in the schema.

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