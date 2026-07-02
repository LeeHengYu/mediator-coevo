# Task Instruction

Complete the following task step-by-step.

## Goal
Revise the existing renewal playbook `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, then save the updated file to `/root/renewal_playbook_updated.hwpx`.

## Step 0 — Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (OFD-based Korean document format). The main editable content is typically in XML files under a path like `Contents/section0.xml` (or similar). Explore the structure first.

## Step 1 — Inspect inputs
1. Find and read `renewal_update.json` — it contains updated field values (customer name, current owner, renewal window, pricing band, escalation contact, pricing note).
2. Find and read `followups.csv` — it contains follow-up items with a `sequence` column for ordering.
3. Locate `renewal_playbook.hwpx` in the workspace.

## Step 2 — Explore the HWPX package
1. Copy `renewal_playbook.hwpx` to a working location.
2. Unzip it to a temporary directory (e.g., `/tmp/hwpx_work/`).
3. List the full directory tree to understand the package structure.
4. Identify the XML file(s) containing the document body/content (likely under `Contents/`).
5. Read the content XML file(s) fully. Identify:
   - Where each of the six updatable fields appears (customer name, current owner, renewal window, pricing band, escalation contact, pricing note).
   - Where the three follow-up lines are.
   - Where the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` appears.

## Step 3 — Plan the edits
For each field in `renewal_update.json`, identify the OLD value currently in the XML and the NEW value from the JSON. Every occurrence of each old value in editable sections must be replaced with the new value.

For the follow-ups:
- Identify the exact XML elements/paragraphs that contain the three existing follow-up lines.
- The CSV items sorted by `sequence` column will replace them. Match the count: if there are 3 existing lines and 3 CSV rows, replace 1-for-1 in sequence order. If counts differ, remove all old follow-up paragraphs and insert new ones in the same location, replicating the XML paragraph structure of the originals.

## Step 4 — Write a Python script to perform the edits
Write a Python script that:
1. Extracts the HWPX zip to a temp directory.
2. Reads `renewal_update.json` and `followups.csv`.
3. Parses the content XML using `xml.etree.ElementTree` (with proper namespace handling — register all namespaces before parsing to preserve them on write-back).
4. Performs text replacements for the six fields across all text nodes in editable sections.
5. Identifies follow-up paragraphs and replaces their text content with CSV items in sequence order.
6. **Critical: For any paragraph whose text was modified, remove all child elements that serve as layout cache / character-position cache.** These are typically elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, or similar layout-related elements within paragraph (`<hp:p>`) elements. Inspect the actual element names in the XML to identify them. Remove these from modified paragraphs so the document renders cleanly.
7. Verifies the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is still present and unmodified.
8. Writes the modified XML back (preserving XML declaration, encoding, namespaces).
9. Re-zips the entire directory structure into `/root/renewal_playbook_updated.hwpx`, preserving the original ZIP structure (no extra directory nesting). Use `zipfile.ZIP_DEFLATED` compression.

## Step 5 — Execute and validate
1. Run the script.
2. Verify `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
3. Extract it and diff the content XML against the original to confirm:
   - All old field values are gone (no stale duplicates).
   - All new field values are present.
   - Follow-up lines match CSV in sequence order.
   - Appendix sentence is preserved exactly.
   - Layout cache elements are removed from modified paragraphs.
4. If any check fails, fix and re-run.

## Step 6 — Run the verifier
If there is a test script (e.g., `test_output.py` or similar) in the task directory, run it with `pytest` to confirm the output passes all checks.

## Important constraints
- Do NOT add new lines beside old ones; replace/remove old values entirely.
- Do NOT modify the appendix sentence.
- The output must be a valid `.hwpx` ZIP package with correct internal structure.
- Remove layout-cache elements (e.g., `linesegarray`, `lineSegArray`, or any positional-cache children) from every paragraph you modify, so the document opens without overlapping characters.
- Register XML namespaces before parsing to avoid `ns0:` prefix pollution in output.

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