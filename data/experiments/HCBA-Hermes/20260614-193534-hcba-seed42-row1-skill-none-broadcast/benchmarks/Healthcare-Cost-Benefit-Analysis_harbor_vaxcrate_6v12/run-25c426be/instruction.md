# Task Instruction

Execute the following steps exactly:

1. **Read all input files** before writing any code:
```bash
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

2. **Read the test file** to understand exact verifier expectations:
```bash
find /root -name '*.py' -path '*/test*' | head -20
cat /tests/test_outputs.py 2>/dev/null || cat /root/tests/test_outputs.py 2>/dev/null || find / -name 'test_output*' -o -name 'test_outputs*' 2>/dev/null | head -5
```
Read whatever test file you find.

3. **Write a Python script** `/root/solve.py` that implements the full analysis. Follow these rules meticulously:

**Data Loading:**
- Load `campaign_manifest.json` — it contains a list/dict of campaigns with fields including `campaign_id`, `campaign_name`, `alias_labels`, `analysis_flag`, `default_active_clinics`, `drug_cost_per_1000_doses_usd`, `doses_per_day`, `crate_tier`.
- Load `crate_cost.csv`, `billing.csv`, `location_overrides.csv`, `suspensions.csv` as CSVs.

**Filtering:**
- Keep only campaigns where `analysis_flag == "review"`.
- Exclude any campaign whose `campaign_id` appears in `suspensions.csv` with `suspension_status == "hold"`.

**Billing Resolution:**
- For each retained campaign, find billing rows where `campaign_label` matches either `campaign_name` or any entry in `alias_labels`.
- Keep only billing rows with `status == "active"`.
- If multiple active rows map to the same campaign, keep the one with the latest `cycle_tag` (sort lexicographically/chronologically).
- Extract `payment_per_dispatch_per_clinic_usd` from the retained billing row.

**Active Clinics (Location Overrides):**
- Filter `location_overrides.csv` to rows where `state == "approved"`.
- Discard rows where `revision` is blank/empty or `active_clinics` is blank/empty.
- Among remaining rows for the same `campaign_id`, keep the one with the highest numeric `revision`.
- If no valid override row exists for a campaign, use `default_active_clinics` from the manifest.

**Cost & Revenue Calculations (per campaign):**
- 6-day model: days_per_dispatch=6, dispatches_per_year=60
- 12-day model: days_per_dispatch=12, dispatches_per_year=30
- `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
- `annual_crate_cost = crate_cost_usd * dispatches_per_year` (crate_cost_usd from crate_cost.csv matched by crate_tier)
- `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
- `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
- `annual_margin_difference_12_minus_6 = margin_12 - margin_6`

**Totals:**
- Sum all per-campaign margins for 6-day and 12-day.
- `total_annual_margin_difference_12_minus_6_usd = total_12 - total_6`
- `absolute_total_margin_difference_usd = abs(total_difference)`

**Decision:**
- If `abs(total_difference) < 11000`: decision = `"move_to_12_day"`
- Otherwise: decision = `"keep_6_day"`

**Round ALL currency values to 2 decimal places.**

**Sort campaigns array by `campaign_id` ascending.**

**Output JSON** `/root/vaxcrate_analysis.json` with this EXACT structure (pay close attention to every key name and the `_usd` suffixes):
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
  "campaigns": [...],
  "totals": {
    "total_annual_margin_6_day_usd": ...,
    "total_annual_margin_12_day_usd": ...,
    "total_annual_margin_difference_12_minus_6_usd": ...,
    "absolute_total_margin_difference_usd": ...
  },
  "recommendation": {
    "decision": "move_to_12_day" or "keep_6_day",
    "justification": "..."
  }
}
```

Each campaign object keys MUST be exactly:
- `campaign_id`, `campaign_name`, `active_clinics`, `drug_cost_per_1000_doses_usd`, `doses_per_day`, `crate_tier`, `crate_cost_usd`, `payment_per_dispatch_per_clinic_usd`
- `annual_drug_cost_6_day_usd`, `annual_drug_cost_12_day_usd`
- `annual_crate_cost_6_day_usd`, `annual_crate_cost_12_day_usd`
- `annual_revenue_6_day_usd`, `annual_revenue_12_day_usd`
- `annual_margin_6_day_usd`, `annual_margin_12_day_usd`
- `annual_margin_difference_12_minus_6_usd`

No extra keys, no missing keys.

**Output Markdown** `/root/vaxcrate_summary.md`:
- 4 to 8 non-empty lines.
- Must include total 6-day margin, total 12-day margin, absolute difference, and the exact decision slug.
- **Format currency values with commas** as thousands separators (e.g., `$1,234,567.89` or `1,234.56`). Use Python's `"{:,.2f}".format(value)` for this.

4. **Run the script:**
```bash
python3 /root/solve.py
```

5. **Validate outputs:**
```bash
cat /root/vaxcrate_analysis.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('Keys:', list(d.keys())); print('Assumptions:', d.get('assumptions')); print('Num campaigns:', len(d.get('campaigns',[]))); print('Totals:', d.get('totals')); print('Recommendation:', d.get('recommendation')); print('First campaign keys:', list(d['campaigns'][0].keys()) if d.get('campaigns') else 'NONE')"
cat /root/vaxcrate_summary.md
```

6. **Run the test suite** if found:
```bash
cd / && python3 -m pytest tests/ -v 2>&1 | head -80
```
Or wherever the tests are located.

7. If tests fail, read the error messages carefully, fix the script, re-run, and re-validate. Pay special attention to:
   - Exact key names (especially `_usd` suffixes)
   - The `assumptions` block must be present with exact keys including `switch_threshold_usd` (NOT `decision_threshold_usd`)
   - Currency comma formatting in the markdown summary
   - No extra keys in campaign objects
   - Correct rounding to 2 decimals

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