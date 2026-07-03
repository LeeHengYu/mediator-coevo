# Task Instruction

You must produce the file `/root/safety_audit_brief_final.hwpx` by filling in the HWPX template with data from the two JSON files. Follow these steps exactly:

1. **Inspect the workspace.** List the contents of `/root/` and identify `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`. Print both JSON files so you know every field and value.

2. **Explore the HWPX template.** A `.hwpx` file is a ZIP archive. Unzip `safety_audit_template.hwpx` into a temporary directory (e.g., `/tmp/hwpx_work/`). List the extracted files. Read every `section*.xml` file inside `Contents/` to find all `{{...}}` placeholders and understand the document structure (summary section, audit table, corrective-action lines).

3. **Write and run a single Python script** (`/root/fill_template.py`) that does all of the following:

   a. **Load the two JSON files** (`audit_overview.json`, `corrective_actions.json`).

   b. **Copy the template HWPX** to the output path and open it as a ZIP for modification (use `zipfile` to read the original, build a new ZIP at `/root/safety_audit_brief_final.hwpx` with modified XML files and all other entries copied byte-for-byte).

   c. **For each `section*.xml`** inside `Contents/`:
      - Parse with `lxml.etree` (or `xml.etree.ElementTree`).
      - Collect all text content (including tail text) across all elements.
      - **Replace every `{{placeholder}}` token** with the corresponding value from the JSON data. Match placeholder names to JSON keys (e.g., `{{facility_name}}`, `{{audit_date}}`, `{{risk_tier}}`, `{{inspector}}`, `{{corrective_action_1}}`, etc.). Be thorough: scan every text and tail attribute of every element.
      - **Fill the three corrective-action lines** in the exact order they appear in `corrective_actions.json`.
      - **Update every occurrence of the risk tier** throughout the document.
      - **Rewrite dates**: convert every occurrence of the inspection/audit date from `YYYY-MM-DD` format to `YYYY.MM.DD` format. Use a global regex replacement on all text/tail content: `re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1.\2.\3', text)` — but only after placeholders are filled so the actual date values are present.
      - **Add severity note**: After filling the risk tier, append the Korean severity note immediately after every occurrence of the risk tier string using this mapping: `High -> 즉시조치`, `Medium -> 계획보완`, `Low -> 모니터링`. For example, if risk tier is `High`, every occurrence of `High` (that was a risk-tier value) becomes `High 즉시조치`.
      - **Strip stale layout caches**: For every `<hp:p>` paragraph element whose text content was modified, find and remove any child `<hp:linesegarray>` element (and its descendants). Use the HWPML namespace (`urn:hancom:hwpml:...` — inspect the XML to get the exact namespace URI). This is critical to prevent overlapping-character rendering bugs.
      - **Verify no `{{` remains** in any text or tail of any element. If any placeholder survives, print an error and exit.
      - Serialize the modified XML back (preserve the XML declaration and encoding).

   d. **Build the output HWPX**: Write all files into a new ZIP at `/root/safety_audit_brief_final.hwpx`, using `ZIP_DEFLATED` compression. Copy every non-section-XML entry unchanged; write modified section XMLs.

4. **Run the script**: `python3 /root/fill_template.py`

5. **Validate the output**:
   - Confirm `/root/safety_audit_brief_final.hwpx` exists and is a valid ZIP (`python3 -c "import zipfile; z=zipfile.ZipFile('/root/safety_audit_brief_final.hwpx'); z.testzip(); print('OK')"`).
   - Unzip the output and grep for `{{` in all XML files — must find zero matches.
   - Grep for the `YYYY.MM.DD` formatted date — must find matches.
   - Grep for the severity note (e.g., `즉시조치` or `계획보완` or `모니터링`) — must find matches.
   - Grep for `linesegarray` in modified paragraphs — should confirm removal from modified paragraphs.
   - Print a few lines of context around the corrective actions to confirm order matches JSON.

6. **Run the verifier** if a test script exists: `cd /root && python -m pytest test_output*.py -v` (or whatever test file is present). If tests fail, read the failure output, fix the script, and re-run.

Key pitfalls to avoid:
- Do NOT leave any `{{...}}` placeholders in the output.
- Do NOT forget to handle tail text on XML elements (placeholders can span element boundaries in HWPML).
- Do NOT skip the `hp:linesegarray` removal step — this was the failure mode in a related task.
- Do NOT hardcode namespace URIs without checking the actual XML; read them from the document.
- Make sure date reformatting is applied globally after all placeholder substitution, so no `YYYY-MM-DD` dates survive.
- The severity note must appear immediately after the risk tier text with a space separator, everywhere the risk tier appears.

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