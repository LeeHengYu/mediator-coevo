# Task Instruction

Perform the following steps in order:

## Step 1: Inspect all input files

Read and display the full contents of:
- `/root/program_catalog.json`
- `/root/cooler_cost.csv`
- `/root/contract_payment.csv`
- `/root/site_overrides.csv`

## Step 2: Write a Python script `/root/solve.py` that does the full analysis

The script must:

### 2a. Load data
- Load `program_catalog.json` as a list of program objects.
- Load `cooler_cost.csv`, `contract_payment.csv`, `site_overrides.csv` as CSV.

### 2b. Filter in-scope programs
- Keep only programs where `review_flag == "review"` (case-sensitive match).

### 2c. Resolve contract payments
- For each row in `contract_payment.csv`, check if its `program_label` matches either `program_name` or any entry in the `known_labels` list from any in-scope program in `program_catalog.json`.
- Ignore payment rows that don't map to any in-scope program.
- Each in-scope program should get exactly one matching payment row's `payment_per_dispatch_per_site_usd`.

### 2d. Resolve active sites
- From `site_overrides.csv`, keep only rows where `approval_state == "approved"`.
- If multiple approved rows exist for the same `program_code`, keep the one with the highest `version_no`.
- For each in-scope program, look up its `program_code` in the filtered overrides. If found, use that row's active site count. Otherwise, use `default_active_sites` from the catalog.

### 2e. Resolve cooler cost
- Match each program's `cooler_type` to `cooler_cost.csv` to get `cooler_cost_usd`.

### 2f. Compute per-program financials
For each in-scope program, compute:
- `annual_drug_cost_10_day = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * 10 * 36 / 1000`
- `annual_drug_cost_20_day = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * 20 * 18 / 1000`
- `annual_cooler_cost_10_day = cooler_cost_usd * 36`
- `annual_cooler_cost_20_day = cooler_cost_usd * 18`
- `annual_revenue_10_day = payment_per_dispatch_per_site_usd * active_sites * 36`
- `annual_revenue_20_day = payment_per_dispatch_per_site_usd * active_sites * 18`
- `annual_margin_10_day = annual_revenue_10_day - annual_drug_cost_10_day - annual_cooler_cost_10_day`
- `annual_margin_20_day = annual_revenue_20_day - annual_drug_cost_20_day - annual_cooler_cost_20_day`
- `annual_margin_difference_20_minus_10 = annual_margin_20_day - annual_margin_10_day`

All currency values rounded to 2 decimal places.

### 2g. Compute totals
- `total_annual_margin_10_day_usd` = sum of all per-program `annual_margin_10_day`
- `total_annual_margin_20_day_usd` = sum of all per-program `annual_margin_20_day`
- `total_annual_margin_difference_20_minus_10_usd` = sum of all per-program differences
- `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_20_minus_10_usd)

Round totals to 2 decimals.

### 2h. Decision
- If `absolute_total_margin_difference_usd < 10000`, decision = `"move_to_20_day"`
- Otherwise, decision = `"keep_10_day"`

### 2i. Build output JSON
Build the JSON object exactly matching the schema from the task. The `programs` array must be sorted by `program_code` ascending. The `assumptions` block must be exactly as specified. The `justification` string should briefly explain the numbers and decision.

Write to `/root/oncocooler_analysis.json` with `json.dump(..., indent=2)`.

### 2j. Build summary markdown
Write `/root/oncocooler_summary.md` with 4-8 non-empty lines including:
- Total 10-day margin (USD)
- Total 20-day margin (USD)
- Absolute difference (USD)
- The exact decision slug (`move_to_20_day` or `keep_10_day`)

## Step 3: Run the script
```bash
python3 /root/solve.py
```

## Step 4: Validate outputs
- Read and display `/root/oncocooler_analysis.json` fully.
- Read and display `/root/oncocooler_summary.md` fully.
- Verify: JSON is valid, all required keys present, programs sorted by program_code, all currency values have at most 2 decimal places, summary has 4-8 non-empty lines and contains the required info.
- Cross-check: the decision matches the threshold rule, totals match sum of per-program values, drug cost formulas are consistent between 10-day and 20-day (note: `10*36 == 20*18 == 360`, so annual drug costs should be equal for both models).

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