# Task Instruction

## Task: Prepare Warehouse Safety Audit Brief (HWPX)

### Goal
Fill in the template `safety_audit_template.hwpx` using data from `audit_overview.json` and `corrective_actions.json`, then save the result to `/root/safety_audit_brief_final.hwpx`.

### Step-by-step Plan

#### Phase 1 — Reconnaissance
1. List all files in the task directory to locate the template and JSON files.
2. Read `audit_overview.json` and `corrective_actions.json` in full. Note every field name and value.
3. Unzip `safety_audit_template.hwpx` into a temporary working directory (e.g., `/tmp/hwpx_work/`).
4. List the contents of the unzipped archive. Identify all XML content files (typically under `Contents/` — look for `section0.xml`, `section1.xml`, etc.).
5. Read every section XML file in full. Identify:
   - All `{{...}}` placeholders and their exact text (they may be split across multiple `<hp:t>` nodes within a single `<hp:p>`).
   - The structure of the summary/overview section and the audit table.
   - The corrective-action lines.
   - Every occurrence of the risk tier placeholder and the inspection date placeholder.
6. Also check `header.xml`, `masterpage.xml`, or any other XML files for additional placeholder occurrences.

#### Phase 2 — Build the Replacement Logic (Python script)
Write a Python script (`/tmp/fill_hwpx.py`) that does the following:

1. **Parse each section XML** using `xml.etree.ElementTree` with proper namespace handling. Register all namespaces found in the file to preserve them on output.

2. **Merge fragmented placeholders**: For each `<hp:p>` element, collect all `<hp:t>` child nodes (at any depth within `<hp:run>` children). Concatenate their `.text` values. If the concatenated text contains any `{{...}}` pattern, perform replacements on the merged string, then write the entire replaced string into the first `<hp:t>` node's `.text` and set all subsequent `<hp:t>` nodes' `.text` to empty string `""`.

3. **Perform substitutions** using the JSON data:
   - Fill all overview/summary fields from `audit_overview.json`.
   - Fill the audit table value cells from `audit_overview.json` (match placeholder names to JSON keys).
   - Fill the three corrective-action lines in the order they appear in `corrective_actions.json`.
   - For the **risk tier**: replace every occurrence of the risk tier placeholder with the actual risk tier value from the JSON.
   - **Severity note**: Immediately after each risk tier value, append a severity note using this mapping: `High` → ` 즉시조치`, `Medium` → ` 계획보완`, `Low` → ` 모니터링`. The note should be appended with a space separator to the same text node that contains the risk tier.
   - **Date rewriting**: Find every occurrence of the inspection date. The JSON will have it in `YYYY-MM-DD` format. Replace every occurrence (both placeholder-substituted and any that might already be literal) with the `YYYY.MM.DD` format (replace hyphens with dots).

4. **Clear layout caches**: After modifying any `<hp:p>` element's text content, find and remove all `<hp:lineSegArray>` child elements (and `<hp:lineSeg>` if present) from that `<hp:p>`. This prevents stale layout data from causing overlapping characters.

5. **Verify no remaining placeholders**: After all substitutions, scan the entire XML text for any remaining `{{` or `}}` patterns. If found, raise an error with details.

6. **Write back the modified XML** to the same file path within the working directory, preserving the XML declaration and encoding.

#### Phase 3 — Repack the HWPX
1. Repack the working directory into a ZIP file at `/root/safety_audit_brief_final.hwpx`.
2. If a `mimetype` file exists in the archive root, add it as the **first entry** with `ZIP_STORED` (no compression). All other files should use `ZIP_DEFLATED`.
3. Preserve the original directory structure exactly.

#### Phase 4 — Validation
1. Verify `/root/safety_audit_brief_final.hwpx` exists and is a valid ZIP.
2. Unzip it to a new temp directory and re-read the section XML files.
3. Confirm:
   - No `{{...}}` placeholders remain anywhere in any XML file.
   - The risk tier value appears with the correct severity note appended.
   - All dates are in `YYYY.MM.DD` format (no `YYYY-MM-DD` remains).
   - The three corrective actions appear in the correct order.
   - No `<hp:lineSegArray>` elements remain in paragraphs that were modified.
   - Section titles and row labels are unchanged from the template.
4. If the task directory contains a test script (e.g., `test_output.py`), run it with `pytest -xvs` and confirm it passes.

### Critical Reminders
- **Namespace handling**: HWPX XML uses namespaces extensively. When parsing, register all namespaces before modifying to avoid namespace prefix changes on output. Use `ET.register_namespace()` for each namespace found.
- **Fragmented placeholders**: `{{placeholder_name}}` is very likely split across multiple `<hp:t>` nodes. You MUST merge text within each `<hp:p>` before doing replacements.
- **Do not modify section titles or row labels** — only fill in value/placeholder areas.
- **Every single `{{...}}` must be replaced** — scan all XML files in the package, not just section0.xml.
- **Date format**: The replacement must happen everywhere the date appears, including if it was already substituted from a placeholder. Do a final pass replacing `YYYY-MM-DD` with `YYYY.MM.DD` across all text nodes.
- **Risk tier + severity note**: If the risk tier is e.g. `High`, the text should read `High 즉시조치` (or the Korean equivalent of the tier value if it's in Korean — check the JSON). Apply the mapping based on the English value in the mapping provided.

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