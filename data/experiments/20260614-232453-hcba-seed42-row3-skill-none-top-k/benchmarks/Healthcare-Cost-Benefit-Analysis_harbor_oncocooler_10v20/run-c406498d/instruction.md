# Task Instruction

Execute the following steps in order to produce `/root/oncocooler_analysis.json` and `/root/oncocooler_summary.md`.

## Step 1 – Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

Read every file carefully before writing any code.

## Step 2 – Inspect the verifier

```bash
find /root -name 'test_output.py' -o -name 'test_*.py' | head -20
```
Then `cat` the test file(s) found. Understand every assertion the verifier makes, especially:
- The exact set of top-level keys expected in the JSON (`assumptions`, `programs`, `totals`, `recommendation` — nothing more, nothing less).
- The exact set of keys inside `assumptions` (must match the schema exactly; do NOT add extra keys like `catalog_structure`).
- Field names and types inside each program object.
- Sorting, rounding, decision logic.

## Step 3 – Write and run the Python solution

Create `/root/solve.py` with the logic below. Pay very close attention to the constants — this was the source of the previous failure:

```
DAYS_PER_DISPATCH_10 = 10
DISPATCHES_PER_YEAR_10 = 36
DAYS_PER_DISPATCH_20 = 20
DISPATCHES_PER_YEAR_20 = 18
SWITCH_THRESHOLD = 10000
```

Detailed logic:

1. Load `program_catalog.json`. Filter to programs where `review_flag == "review"`. Build a dict keyed by `program_code`.

2. For each in-scope program, also index by `program_name` and every entry in `known_labels` (if present) so contract_payment rows can be resolved.

3. Load `contract_payment.csv`. For each row, look up `program_label` against the name/label index built above. Skip rows that don't match any in-scope program. Store the `payment_per_dispatch_per_site_usd` value keyed by `program_code`.

4. Load `cooler_cost.csv`. Build a dict from `cooler_type` → `cooler_cost_usd`.

5. Load `site_overrides.csv`. Keep only rows with `approval_state == "approved"`. Group by `program_code`, pick the row with the highest `version_no`. Store `active_sites` (the column that represents the site count — inspect the CSV header to find the correct column name) keyed by `program_code`.

6. For each in-scope program:
   - `active_sites`: from site_overrides if available, else `default_active_sites` from catalog.
   - `cooler_cost_usd`: from cooler_cost lookup by `cooler_type`.
   - `payment_per_dispatch_per_site_usd`: from contract_payment lookup.
   - Compute for EACH model (10-day and 20-day):
     - `annual_drug_cost = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000`
     - `annual_cooler_cost = cooler_cost_usd * dispatches_per_year`  (NOTE: cooler cost is per dispatch, NOT per site — but re-check the test expectations; if the verifier multiplies by sites, adjust accordingly. Inspect the test file to confirm.)
     - `annual_revenue = payment_per_dispatch_per_site_usd * active_sites * dispatches_per_year`
     - `annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost`
   - `annual_margin_difference_20_minus_10 = margin_20 - margin_10`
   - Round ALL currency values to 2 decimal places.

7. Sort programs by `program_code` ascending.

8. Compute totals:
   - `total_annual_margin_10_day_usd` = sum of all program 10-day margins
   - `total_annual_margin_20_day_usd` = sum of all program 20-day margins
   - `total_annual_margin_difference_20_minus_10_usd` = sum of all per-program differences
   - `absolute_total_margin_difference_usd` = abs(total_difference)
   - Round all to 2 decimals.

9. Decision:
   - If `abs(total_difference) < 10000` → `"move_to_20_day"`
   - Otherwise → `"keep_10_day"`

10. Build the JSON with EXACTLY these top-level keys: `assumptions`, `programs`, `totals`, `recommendation`. No extra keys.
    - `assumptions` must contain EXACTLY these keys:
      ```
      dispatches_per_year_10_day, dispatches_per_year_20_day,
      days_per_dispatch_10_day, days_per_dispatch_20_day,
      switch_threshold_usd, site_override_rule
      ```
      No extra keys.

11. Write `/root/oncocooler_analysis.json` with `json.dump(..., indent=2)`.

12. Write `/root/oncocooler_summary.md` with 4–8 non-empty lines including:
    - Total 10-day margin (USD)
    - Total 20-day margin (USD)
    - Absolute difference (USD)
    - Decision slug exactly as `move_to_20_day` or `keep_10_day`

## Step 4 – Run the solution

```bash
cd /root && python solve.py
```

## Step 5 – Validate

```bash
cat /root/oncocooler_analysis.json
cat /root/oncocooler_summary.md
```

Verify:
- JSON parses cleanly.
- Top-level keys are exactly `{"assumptions", "programs", "totals", "recommendation"}`.
- `assumptions` keys match the schema exactly (no extras).
- Programs are sorted by `program_code`.
- All currency values have at most 2 decimal places.
- Summary has 4–8 non-empty lines with required content.

## Step 6 – Run the verifier

```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```

If any test fails:
- Read the assertion error carefully.
- Check whether cooler cost should be per-dispatch or per-dispatch-per-site (the most common ambiguity).
- Check whether the dispatch/day constants are correctly assigned (DAYS_20=20, DISP_20=18, not swapped).
- Fix and re-run until all tests pass.

**IMPORTANT WARNINGS from previous failure:**
- Do NOT swap the 20-day constants. `days_per_dispatch_20_day = 20` and `dispatches_per_year_20_day = 18`.
- Do NOT add extra keys to the `assumptions` dict or the top-level JSON.
- Inspect the test file BEFORE writing code to understand the exact cooler cost formula (per-dispatch vs per-dispatch-per-site).

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