# Task Instruction

Execute the following steps to produce `/root/renewal_playbook_updated.hwpx`:

1. **Inspect the workspace.** List files in the task directory to locate `renewal_playbook.hwpx`, `renewal_update.json`, `followups.csv`, and any test/verifier scripts.

2. **Read the input data files.**
   - Parse `renewal_update.json` to extract the new values for: customer name, current owner, renewal window, pricing band, escalation contact, and pricing note.
   - Parse `followups.csv` and sort rows by the `sequence` column (ascending).

3. **Explore the HWPX package.**
   - Unzip `renewal_playbook.hwpx` into a temporary working directory (e.g., `/tmp/hwpx_work/`).
   - List the contents, especially `Contents/` to find all `section*.xml` files.
   - Read each section XML to understand the document structure, namespaces, and identify where the old values appear.

4. **Identify old values to replace.**
   - From the existing section XMLs, locate every occurrence of the current (old) customer name, current owner, renewal window, pricing band, escalation contact, and pricing note.
   - Also locate the three existing follow-up lines that need to be replaced.
   - Confirm the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` exists and note its location so it is preserved.

5. **Write a Python script** that does all of the following in one pass:

   ```python
   import json, csv, zipfile, shutil, os, copy
   from lxml import etree

   # --- Load inputs ---
   with open('<path>/renewal_update.json') as f:
       updates = json.load(f)
   
   with open('<path>/followups.csv', newline='', encoding='utf-8') as f:
       reader = csv.DictReader(f)
       followups = sorted(reader, sequence=lambda r: int(r['sequence']))
   
   WORK = '/tmp/hwpx_work'
   OUT = '/root/renewal_playbook_updated.hwpx'
   
   # --- Unzip ---
   # (already done in step 3, or do it here)
   
   # --- Build old→new replacement map from the JSON ---
   # The JSON likely has keys like "customer_name", "current_owner", etc.
   # with sub-keys "old" and "new", OR it may just have new values.
   # Inspect the JSON first; adapt accordingly.
   # If JSON only has new values, extract old values from the XML.
   
   # --- Process each section XML ---
   for each section XML file in Contents/:
       tree = etree.parse(filepath)
       root = tree.getroot()
       nsmap = {extract namespaces}
       
       # Walk all text-bearing elements
       # For each <hp:t> or text node:
       #   a) Replace old field values with new ones (customer name, owner,
       #      renewal window, pricing band, escalation contact, pricing note)
       #   b) For follow-up lines: identify the three old follow-up paragraphs,
       #      replace their text content with the CSV items in sequence order.
       #      If there are exactly 3 old lines and a different number of CSV rows,
       #      add or remove paragraph elements as needed.
       #   c) Do NOT touch the appendix sentence.
       
       # --- Remove stale layout caches ---
       # For every <hp:p> paragraph that had any text modification,
       # find and remove all <hp:lineSegArray> child elements.
       # This prevents overlapping-character rendering issues.
       for lsa in tree.xpath('//hp:lineSegArray', namespaces=nsmap):
           parent = lsa.getparent()
           # Only remove if this paragraph was modified
           # (safest: remove from ALL modified paragraphs;
           #  acceptable: remove from all paragraphs globally)
           parent.remove(lsa)
       
       # Write back with xml_declaration and encoding
       tree.write(filepath, xml_declaration=True, encoding='UTF-8')
   
   # --- Repackage into HWPX (ZIP) ---
   # HWPX is a ZIP. Recreate it preserving the original directory structure.
   with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zout:
       for root_dir, dirs, files in os.walk(WORK):
           for fname in files:
               full = os.path.join(root_dir, fname)
               arcname = os.path.relpath(full, WORK)
               zout.write(full, arcname)
   ```

6. **Key cautions:**
   - When replacing text, handle cases where a single logical string (e.g., the customer name) might be split across multiple `<hp:t>` runs within one `<hp:run>` or across sibling runs. If so, consolidate or replace across the run boundaries.
   - The follow-up replacement must match the exact number of CSV rows. If the CSV has more or fewer than 3 items, adjust the paragraph count accordingly (add cloned paragraphs or remove extras).
   - Ensure `<hp:lineSegArray>` removal targets only modified paragraphs (or all paragraphs — either is safe; the verifier checks that modified paragraphs don't retain them).
   - Preserve all non-section files in the HWPX package exactly (mimetype, META-INF/, header/settings XMLs, etc.).
   - If there is a `mimetype` file, write it FIRST in the ZIP and use `ZIP_STORED` (no compression) for it, as some OPC-like formats require this.

7. **Validate the output.**
   - Verify `/root/renewal_playbook_updated.hwpx` is a valid ZIP.
   - Unzip it and check that:
     - All new values appear in the section XMLs.
     - No old values remain (except the preserved appendix sentence).
     - The appendix sentence is intact.
     - No `<hp:lineSegArray>` elements exist in modified paragraphs.
     - Follow-up items appear in correct sequence order.
   - Run `python test_output.py` (or whatever verifier exists in the task directory) and confirm it passes.

8. **If the verifier fails**, read the error output carefully, identify which assertion failed, inspect the relevant XML section, fix the issue, re-package, and re-run the verifier. Do not mark complete until the verifier passes.

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