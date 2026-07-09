# Task Instruction

## Task: Prepare warehouse safety audit brief HWPX document

### Goal
Fill the template `safety_audit_template.hwpx` with data from `audit_overview.json` and `corrective_actions.json`, then save the result to `/root/safety_audit_brief_final.hwpx`.

### Step 0 — Explore the workspace
1. `ls /root/` and find all task-related files.
2. `cat /root/audit_overview.json` — note every field name and value.
3. `cat /root/corrective_actions.json` — note the three corrective-action entries and their order.
4. The template is a `.hwpx` file (ZIP-based). List its contents:
   ```
   cd /root && python3 -c "import zipfile; z=zipfile.ZipFile('safety_audit_template.hwpx'); print('\n'.join(z.namelist()))"
   ```
5. Extract and print every XML file inside the HWPX, especially files under `Contents/` (e.g., `section0.xml`, `section1.xml`, etc.). Read them carefully to understand:
   - Where `{{...}}` placeholders are (exact placeholder names and surrounding XML tags).
   - Which section file contains the overview/summary fields.
   - Which section file contains the audit table and corrective-action lines.
   - How the risk tier placeholder appears (it may appear multiple times across files).
   - How the inspection date placeholder appears.
   - Any `<hp:linesegarray>` or similar layout-cache elements attached to paragraphs.

### Step 1 — Build the replacement plan
From the JSON data and the template XML, create a complete mapping of every `{{placeholder}}` to its replacement value. Apply these rules:

1. **Overview fields**: Replace each overview placeholder with the corresponding value from `audit_overview.json`.
2. **Audit table value cells**: Replace each table-value placeholder with the corresponding value.
3. **Corrective-action lines**: Fill the three corrective-action rows in the same order they appear in `corrective_actions.json`.
4. **Risk tier + severity note**: Wherever the risk tier value appears (after substitution), format it as `RiskTier (SeverityNote)` using this mapping:
   - `High` → `High (즉시조치)`
   - `Medium` → `Medium (계획보완)`
   - `Low` → `Low (모니터링)`
   
   **CRITICAL**: The format MUST use parentheses with a space before the opening parenthesis. E.g., `High (즉시조치)` — NOT `High 즉시조치`.
5. **Inspection date**: Rewrite every occurrence of the inspection date from `YYYY-MM-DD` format to `YYYY.MM.DD` format (replace hyphens with dots).
6. **No leftover placeholders**: After all substitutions, verify that NO `{{...}}` text remains anywhere in any XML file.

### Step 2 — Perform the substitutions in Python
Write a Python script that:
1. Copies `safety_audit_template.hwpx` to `safety_audit_brief_final.hwpx`.
2. Opens the copy as a ZIP, reads each member.
3. For XML files (especially section files), performs all text replacements.
4. **Layout-cache removal**: For any paragraph (`<hp:p>` or similar) whose text content was modified, remove any `<hp:linesegarray>...</hp:linesegarray>` elements (and any similar layout-cache elements like `<hp:lineseg .../>`) from that paragraph. This prevents overlapping-character rendering issues. Be careful to only strip these from paragraphs you actually modified.
5. Writes all members back into a new valid ZIP (the final `.hwpx`), preserving the original ZIP structure and compression.

### Step 3 — Validate the output
1. Confirm `/root/safety_audit_brief_final.hwpx` exists and is a valid ZIP:
   ```python
   import zipfile
   z = zipfile.ZipFile('/root/safety_audit_brief_final.hwpx')
   print(z.namelist())
   ```
2. Read and print the full content of every section XML file from the final HWPX.
3. Verify:
   - The exact string `High (즉시조치)` (or whichever risk tier applies) appears in the XML. Print a confirmation.
   - The date is in `YYYY.MM.DD` format everywhere.
   - No `{{` or `}}` substrings remain anywhere in any XML file.
   - Section titles and row labels from the original template are preserved.
   - The three corrective actions appear in the correct order.
4. If any check fails, fix and re-validate before finishing.

### Important Notes
- Read every file BEFORE editing. Do not assume file contents.
- The severity note format with parentheses is **verified by the test**: `'High (즉시조치)' in xml_content` must be True.
- Keep all existing section titles and row labels unchanged.
- The final file must be at exactly `/root/safety_audit_brief_final.hwpx`.

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