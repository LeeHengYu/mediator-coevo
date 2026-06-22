# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the full contents of:
- `/root/program_catalog.json`
- `/root/cooler_cost.csv`
- `/root/contract_payment.csv`
- `/root/site_overrides.csv`

## Step 2: Write and run a Python script

Create `/root/solve.py` with the following logic, then run it with `python3 /root/solve.py`.

### Logic:

1. **Load data:**
   - `program_catalog.json` → list of program objects
   - `cooler_cost.csv` → DataFrame
   - `contract_payment.csv` → DataFrame
   - `site_overrides.csv` → DataFrame

2. **Filter in-scope programs:** Only programs where `review_flag == "review"`.

3. **Resolve contract payments:** For each row in `contract_payment.csv`, check if `program_label` matches either `program_name` or any entry in `known_labels` (list) from any in-scope program in the catalog. Ignore payment rows that don't map. Use the matched program's data.

4. **Determine active sites per program:**
   - From `site_overrides.csv`, keep only rows where `approval_state == "approved"`.
   - Among approved rows for the same `program_code`, keep the one with the highest `version_no`.
   - Use the `active_sites` (or equivalent column) from that row.
   - If no approved override row exists for a program, use `default_active_sites` from `program_catalog.json`.

5. **Look up cooler cost:** Match each program's `cooler_type` to `cooler_cost.csv` to get `cooler_cost_usd`.

6. **Compute per-program values (for both 10-day and 20-day models):**
   - `annual_revenue = payment_per_dispatch_per_site_usd * active_sites * dispatches_per_year`
   - `annual_drug_cost = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000`
   - `annual_cooler_cost = cooler_cost_usd * dispatches_per_year` (cooler cost is per dispatch, so multiply by dispatches per year — but WAIT: re-read the task. The task says "Cooler cost uses cooler_cost_usd from cooler_cost.csv" but doesn't give an explicit annual cooler cost formula. The annual_cooler_cost fields exist in the output. Likely: `annual_cooler_cost = cooler_cost_usd * dispatches_per_year * active_sites` — but this needs verification. Actually, look at the margin formula: `annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost`. The cooler cost is likely per cooler per dispatch, and each site gets a cooler each dispatch. So: `annual_cooler_cost = cooler_cost_usd * active_sites * dispatches_per_year`. Use this formula.
   - `annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost`
   - `annual_margin_difference_20_minus_10 = annual_margin_20_day - annual_margin_10_day`

   10-day: dispatches_per_year=36, days_per_dispatch=10
   20-day: dispatches_per_year=18, days_per_dispatch=20

7. **Totals:**
   - Sum all per-program `annual_margin_10_day_usd` → `total_annual_margin_10_day_usd`
   - Sum all per-program `annual_margin_20_day_usd` → `total_annual_margin_20_day_usd`
   - `total_annual_margin_difference_20_minus_10_usd = total_20 - total_10`
   - `absolute_total_margin_difference_usd = abs(total_difference)`

8. **Decision:**
   - If `abs(total_difference) < 10000` → `move_to_20_day`
   - Otherwise → `keep_10_day`

9. **Round** all currency values to 2 decimal places.

10. **Sort** the programs array by `program_code` ascending (string sort).

11. **Write `/root/oncocooler_analysis.json`** with the exact schema specified, using `json.dump` with `indent=2`.

12. **Write `/root/oncocooler_summary.md`** with 4-8 non-empty lines including:
    - Total 10-day margin (USD)
    - Total 20-day margin (USD)
    - Absolute difference (USD)
    - Final decision using exact slug (`move_to_20_day` or `keep_10_day`)

### Important implementation details:
- When matching `program_label` from contract_payment.csv to programs, do case-sensitive matching first. If the data has inconsistencies, try case-insensitive. But start with exact matching.
- `known_labels` in the catalog is a list; check if `program_label` is in that list OR equals `program_name`.
- For site_overrides, inspect the actual column names carefully (they might be `active_sites`, `site_count`, `active_site_count`, etc.). Use whatever column contains the site count.
- Handle the possibility that cooler_cost.csv might have the cooler type column named differently.
- Print intermediate results for debugging: list of in-scope programs, matched payments, active sites, computed values.

## Step 3: Verify outputs

After the script runs:
1. `cat /root/oncocooler_analysis.json` and verify:
   - The `assumptions` block matches exactly as specified
   - All currency values are rounded to 2 decimals
   - Programs are sorted by `program_code` ascending
   - All required fields are present
   - The decision logic is correct
2. `cat /root/oncocooler_summary.md` and verify:
   - 4-8 non-empty lines
   - Contains total 10-day margin, total 20-day margin, absolute difference, and the exact decision slug

If anything looks wrong, fix the script and re-run.

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