# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

## Step 2: Build and run the analysis script

Create `/root/solve.py` with the following logic, then run it with `python3 /root/solve.py`.

The script must:

### 2a. Load data
- Load `program_catalog.json` as JSON (it may be an array or object with a key holding an array).
- Load `cooler_cost.csv`, `contract_payment.csv`, `site_overrides.csv` as CSV.

### 2b. Filter in-scope programs
- Keep only catalog entries where `review_flag` equals exactly `"review"` (case-sensitive).

### 2c. Resolve contract payments
- For each row in `contract_payment.csv`, match its `program_label` to either `program_name` or any entry in the `known_labels` list from the catalog. Only keep rows that map to an in-scope program. Extract `payment_per_dispatch_per_site_usd` for each program.
- If multiple payment rows match the same program, note this and investigate; the task likely expects one match per program.

### 2d. Resolve active sites
- From `site_overrides.csv`, keep only rows where `approval_state` is `"approved"` (case-sensitive match; inspect actual values first).
- For each `program_code`, if multiple approved rows exist, keep the one with the highest `version_no`.
- Use the `active_sites` (or similarly named column) from that row.
- If an in-scope program has no approved override row, use `default_active_sites` from the catalog.

### 2e. Resolve cooler cost
- Match each program's `cooler_type` to `cooler_cost.csv` to get `cooler_cost_usd`.

### 2f. Compute per-program values
For each in-scope program, for both 10-day (days=10, dispatches=36) and 20-day (days=20, dispatches=18) models:

- `annual_drug_cost = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000`
- `annual_cooler_cost = cooler_cost_usd * dispatches_per_year` (NOTE: the formula says cooler_cost from cooler_cost.csv; inspect whether it should be multiplied by active_sites too — the task says `cooler_cost_usd` from cooler_cost.csv and the cooler is dispatched, so annual_cooler_cost = cooler_cost_usd * dispatches_per_year. But re-read the task: it says "cooler dispatches" so it likely means one cooler per dispatch per site, i.e., `cooler_cost_usd * active_sites * dispatches_per_year`. HOWEVER, the task does NOT explicitly state per-site for cooler cost. Look at the data magnitudes to decide. Actually, the task says "cooler dispatches" in context of sites, and the revenue formula explicitly includes active_sites. For cooler cost, the task only says "Cooler cost uses cooler_cost_usd from cooler_cost.csv, matched by cooler_type" without giving a formula with active_sites. Since the task gives explicit formulas for revenue and drug cost (both include active_sites) but does NOT give an explicit annual_cooler_cost formula, we must infer. The most natural reading for a dispatch-based cooler cost in an oncology program with multiple sites is: `annual_cooler_cost = cooler_cost_usd * active_sites * dispatches_per_year`. But if the numbers don't make sense, try without active_sites. Actually, let me re-read: the task says the output schema has `annual_cooler_cost_10_day_usd` etc. Since the drug cost and revenue both scale by active_sites and dispatches, and cooler dispatches go to sites, use: `annual_cooler_cost = cooler_cost_usd * active_sites * dispatches_per_year`.)
- `annual_revenue = payment_per_dispatch_per_site_usd * active_sites * dispatches_per_year`
- `annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost`
- `annual_margin_difference_20_minus_10 = annual_margin_20_day - annual_margin_10_day`

Round ALL currency values to 2 decimal places.

### 2g. Compute totals
- `total_annual_margin_10_day_usd` = sum of all programs' `annual_margin_10_day_usd`
- `total_annual_margin_20_day_usd` = sum of all programs' `annual_margin_20_day_usd`
- `total_annual_margin_difference_20_minus_10_usd` = sum of all programs' `annual_margin_difference_20_minus_10_usd`
- `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_20_minus_10_usd)

Round all to 2 decimals.

### 2h. Decision
- If `absolute_total_margin_difference_usd < 10000`, decision = `"move_to_20_day"`
- Otherwise, decision = `"keep_10_day"`

### 2i. Write `/root/oncocooler_analysis.json`
Write the JSON file with the exact schema from the task. The `programs` array must be sorted by `program_code` ascending (alphabetical). Include the `assumptions` block exactly as specified. The `justification` string should briefly explain the decision referencing the absolute difference and threshold.

### 2j. Write `/root/oncocooler_summary.md`
Write a markdown file with 4-8 non-empty lines that includes:
- Total 10-day margin in USD
- Total 20-day margin in USD
- Absolute difference in USD
- The exact decision slug (`move_to_20_day` or `keep_10_day`)

## Step 3: Validate outputs

```bash
python3 -c "
import json
data = json.load(open('/root/oncocooler_analysis.json'))
print('Programs:', len(data['programs']))
for p in data['programs']:
    print(p['program_code'], p['annual_margin_10_day_usd'], p['annual_margin_20_day_usd'], p['annual_margin_difference_20_minus_10_usd'])
print('Totals:', data['totals'])
print('Decision:', data['recommendation']['decision'])
"
```

```bash
cat /root/oncocooler_summary.md
```

Verify:
- JSON is valid and parseable
- All currency values have at most 2 decimal places
- Programs are sorted by program_code
- Summary has 4-8 non-empty lines and contains the required info
- The decision slug in the summary matches the JSON decision exactly

If the cooler cost formula `cooler_cost_usd * active_sites * dispatches_per_year` produces results where the absolute total difference >= 10000 and the decision is `keep_10_day`, also try WITHOUT active_sites in cooler cost (i.e., `cooler_cost_usd * dispatches_per_year`) and see which interpretation gives a more reasonable result. Pick the interpretation that makes the numbers internally consistent. Document which interpretation you used.

## Important Notes
- Read all input files carefully before writing the script. Column names may differ slightly from what's described (e.g., underscores vs spaces, different capitalization). Adapt accordingly.
- `known_labels` in the catalog may be a list/array — check each element.
- Be precise with rounding: use `round(value, 2)` for all currency outputs.
- The `programs` array sort must be ascending by `program_code` string value.

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