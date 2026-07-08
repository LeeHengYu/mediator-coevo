# Task Instruction

Execute the following steps to update the HWPX renewal playbook:

1. **Inspect input files.** Read `renewal_update.json` and `followups.csv` in the task directory to understand the exact field values and follow-up items (with their `sequence` ordering).

2. **Unzip the HWPX package.**
   ```bash
   mkdir -p /tmp/hwpx_work
   cd /tmp/hwpx_work
   unzip -o /root/renewal_playbook.hwpx -d hwpx_contents
   ```
   Then list the extracted structure and read `hwpx_contents/Contents/section0.xml` to understand the XML layout and namespace prefixes.

3. **Write a Python script** (`/tmp/hwpx_work/update_hwpx.py`) that does the following:

   a. **Parse** `Contents/section0.xml` with `xml.etree.ElementTree`, registering all namespaces found in the root element so they are preserved on output (use `ET.register_namespace` for each).

   b. **Load** `renewal_update.json` (dict of field→new-value) and `followups.csv` (columns include at least `sequence` and a text/description column).

   c. **Field replacement:** For every paragraph (`hp:p`) in the document, inspect the text content of its text runs (`hp:run/hp:t` or similar). For each field listed in the JSON (customer name, current owner, renewal window, pricing band, escalation contact, pricing note), find the old value and replace it with the new value everywhere it appears in editable paragraph text. Do NOT touch the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.`.

   d. **Follow-up replacement:** Identify the three existing follow-up lines (they will be consecutive paragraphs with recognizable follow-up content). Remove all three old follow-up paragraphs from their parent element. Then, for each row in `followups.csv` sorted by `sequence`, clone the structure of one of the removed follow-up paragraphs (to preserve formatting), set its text to the CSV item's text, and insert it at the position where the old follow-ups were.

   e. **Layout cache cleanup:** For every paragraph (`hp:p`) whose text content was modified (either by field replacement or follow-up replacement), find and remove any `hp:linesegarray` child element. This prevents stale layout-cache from causing overlapping characters.

   f. **Write back** the modified XML to `Contents/section0.xml`, using `xml_declaration=True, encoding='UTF-8'`.

4. **Repackage the HWPX ZIP** with `mimetype` as the first entry, stored (no compression):
   ```python
   import zipfile, os
   out_path = '/root/renewal_playbook_updated.hwpx'
   base = '/tmp/hwpx_work/hwpx_contents'
   with zipfile.ZipFile(out_path, 'w') as zf:
       # mimetype first, stored
       mt = os.path.join(base, 'mimetype')
       if os.path.exists(mt):
           zf.write(mt, 'mimetype', compress_type=zipfile.ZIP_STORED)
       # all other files
       for root, dirs, files in os.walk(base):
           for f in files:
               full = os.path.join(root, f)
               arcname = os.path.relpath(full, base)
               if arcname == 'mimetype':
                   continue
               zf.write(full, arcname, compress_type=zipfile.ZIP_DEFLATED)
   ```

5. **Validate the output:**
   - Unzip `/root/renewal_playbook_updated.hwpx` to a temp dir and parse `Contents/section0.xml` to confirm it is well-formed XML.
   - Grep for the new field values from the JSON to confirm they appear.
   - Grep for old field values to confirm they do NOT appear.
   - Confirm the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unchanged.
   - Confirm follow-up items match the CSV rows in `sequence` order.
   - Confirm no `hp:linesegarray` elements remain in modified paragraphs.
   - Confirm `mimetype` is the first entry in the ZIP.

6. **Run the verifier** if a test script exists:
   ```bash
   cd /root && python -m pytest test_output.py -v
   ```
   If it fails, read the error output, diagnose, fix, and re-run.

Key constraints to remember:
- The `mimetype` file MUST be first in the ZIP and stored without compression.
- All XML namespaces must be preserved exactly.
- Old values must be removed, not duplicated alongside new values.
- The appendix Korean sentence must remain untouched.
- Every modified paragraph must have its `hp:linesegarray` child removed.
- The result must be a valid HWPX (ZIP) package.

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