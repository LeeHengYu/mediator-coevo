# Task Instruction

You need to complete a project proposal HWPX document by filling in placeholders from a JSON file and saving the result.

## Steps

1. **Inspect the workspace**: List files in the task directory to find `project_proposal_template.hwpx` and `project_proposal.json`. Also check for any test/verifier files.

2. **Read the JSON data**: `cat project_proposal.json` to see all the key-value pairs that will replace `{{...}}` placeholders.

3. **Examine the HWPX template**: HWPX files are ZIP archives containing XML files. Do:
   ```bash
   cp project_proposal_template.hwpx /tmp/template.zip
   cd /tmp && mkdir hwpx_work && cd hwpx_work && unzip /tmp/template.zip
   ```
   Then `find . -type f` to see the structure. The content XML files are typically under `Contents/` (e.g., `section0.xml`, `section1.xml`, etc.).

4. **Inspect all section XML files**: Read each `section*.xml` file. Identify:
   - All `{{...}}` placeholders and which JSON keys they correspond to
   - Phase lines (단계1, 단계2, 단계3) that contain date ranges — you'll need to append month spans
   - The budget placeholder — the JSON value's commas must be removed (keep the currency symbol like ₩)
   - Any `<hp:lineSegArray>` elements inside `<hp:p>` blocks

5. **Write a Python script** to perform the transformation. The script should:

   a. **Copy the template** to `/root/project_proposal_ready.hwpx`.
   
   b. **Open it as a ZIP** (read), create a new ZIP (write), and for each entry:
      - If it's a section XML file, apply transformations; otherwise copy as-is.
   
   c. **Placeholder replacement**: For each `{{key}}` placeholder in the XML text, replace it with the corresponding value from the JSON. For the budget field, remove commas from the numeric part while keeping the currency symbol (e.g., `₩1,000,000` → `₩1000000` or if the JSON has `1,000,000` with a separate symbol, handle accordingly).
   
   d. **Month span calculation**: For each phase line (단계1, 단계2, 단계3), find the date range in that line (e.g., `2025.01 ~ 2025.03`), calculate the inclusive month span, and append ` (N개월)` after the phase description. The calculation: if dates are `YYYY.MM ~ YYYY.MM`, months = (end_year - start_year) * 12 + (end_month - start_month) + 1.
   
   e. **Remove stale layout caches**: For any `<hp:p>` paragraph element whose text content was modified, remove the entire `<hp:lineSegArray>...</hp:lineSegArray>` element within it. This prevents rendering artifacts. Use regex to find `<hp:p ...>` blocks, check if they were modified, and strip `<hp:lineSegArray>` elements from them.
   
   f. **Verification**: After writing, re-open the output ZIP and scan all XML content to confirm no `{{` remains anywhere.

   Here's the approach for the layout cache removal — process the XML as a string:
   - Before replacement, record which `<hp:p>` blocks exist.
   - After replacement, for any `<hp:p>` block whose content changed, use regex to remove `<hp:lineSegArray>.*?</hp:lineSegArray>` (with `re.DOTALL`).

6. **Run the script** and verify:
   ```bash
   python3 /tmp/fill_hwpx.py
   ```
   Then verify the output:
   ```bash
   cd /tmp && mkdir verify && cd verify && unzip /root/project_proposal_ready.hwpx
   grep -r '{{' Contents/
   ```
   This grep should return nothing (no remaining placeholders).

7. **Run the verifier**: Check for test files in the task directory and run:
   ```bash
   cd /path/to/task && python -m pytest test_output.py -v
   ```

**Key details to remember:**
- Budget normalization: remove commas from the numeric value, keep currency symbol.
- Month spans are inclusive (Jan to Mar = 3 months).
- Remove `<hp:lineSegArray>` only from paragraphs you actually modified.
- Keep all Korean labels and static note lines unchanged.
- The output must be a valid ZIP (HWPX package) at `/root/project_proposal_ready.hwpx`.

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
Task metadata: author_email=catpaw@example.com, author_name=CatPaw Task Engineer, category=document-editing, difficulty=medium, tags=[hwpx, xml-editing, document-processing, latent-method-reuse].
Verifier config: timeout_sec=600.0.