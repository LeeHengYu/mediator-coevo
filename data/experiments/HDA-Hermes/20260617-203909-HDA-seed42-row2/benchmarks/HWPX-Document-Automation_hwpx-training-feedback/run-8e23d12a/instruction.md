# Task Instruction

You must fill in the HWPX training feedback template and save the result. Follow these steps precisely:

1. **Inspect the workspace.** List files in the current directory and any subdirectories to locate `training_feedback_template.hwpx` and `training_feedback.json`. Also check if there is a `test_output.py` or any verifier script.

2. **Read the JSON data.** Print the full contents of `training_feedback.json` so you know every key-value pair.

3. **Examine the HWPX structure.** The `.hwpx` file is a ZIP archive. Unzip it to a temp directory (e.g., `/tmp/hwpx_work/`) and list all files inside. Identify which XML files contain `{{...}}` placeholders — typically `Contents/section0.xml`, but check all `section*.xml` files and any other XML files.

4. **Read the raw XML of every file that contains placeholders.** Print the full contents so you can see every placeholder and the surrounding XML structure.

5. **Write a Python script** (`/root/fill_template.py`) that does all of the following:

   a. **Copy** the template HWPX to `/root/training_feedback_ready.hwpx`.
   
   b. **Open** the copy as a ZIP (using `zipfile.ZipFile` in read mode), read all entries, then **rewrite** it (write mode) with modified XML files. This preserves the ZIP structure.
   
   c. **Load** `training_feedback.json`.
   
   d. **For each XML file** in the archive (especially `Contents/section*.xml`):
      - Parse with `lxml.etree` (use the actual bytes, preserve encoding declaration).
      - Walk every text node (element `.text` and `.tail`) looking for `{{...}}` patterns.
      - Replace each `{{key}}` with the corresponding JSON value, applying these transformations:
        - **`참석자수`**: Convert to digits only. If the JSON value is like `"32명"`, output `"32"`. If it's already a number, use the number as a string with no units.
        - **`만족도`**: Rewrite as `"X.X점 (5.0점 만점)"` format, where X.X is the numeric score from the JSON. For example, if JSON has `4.5` or `"4.5/5.0"` or `"4.5점"`, output `"4.5점 (5.0점 만점)"`.
        - **`종합의견`** (or whatever key maps to the overall opinion): After substituting the JSON comment value, append ` 후속 심화반 검토 요망.` (with a space before it) at the end.
        - All other placeholders: substitute the JSON value directly.
      - **Remove stale layout caches**: For every `<hp:p>` element that had any text modified, find and remove all child `<hp:lineSegArray>` elements (and their children). Use the actual namespace URI from the document (likely `http://www.hancom.co.kr/hwpml/2011/paragraph` or similar — extract it from the parsed XML's nsmap or from the root element).
      - Serialize the XML back to bytes, preserving the XML declaration and encoding.
   
   e. **For non-XML files** in the archive, copy them byte-for-byte.
   
   f. **Validate**: After writing, re-open `/root/training_feedback_ready.hwpx` as a ZIP, read back the modified XML files, and assert:
      - No `{{` or `}}` patterns remain anywhere in any XML file.
      - The file is a valid ZIP.
      - Print a summary of substitutions made.

6. **Run the script**: `python3 /root/fill_template.py`

7. **Verify the output**: 
   - Unzip `/root/training_feedback_ready.hwpx` to a temp location and `grep -r '{{' ` to confirm no placeholders remain.
   - Check that the 참석자수 value is digits only (no Korean unit suffix).
   - Check that 만족도 follows the `X.X점 (5.0점 만점)` format.
   - Check that the overall opinion sentence ends with `후속 심화반 검토 요망.`
   - Check that `<hp:lineSegArray>` elements are removed from modified paragraphs.

8. **Run the verifier** if one exists: `cd /root && python -m pytest test_output.py -v` (or whatever test file is present). If tests fail, read the failure output carefully, diagnose, fix, and re-run.

**Important notes:**
- The placeholder keys in the template may not exactly match JSON keys — inspect both carefully and map them correctly. The JSON keys might be in Korean or English; the template placeholders might use either. Match them by meaning.
- When handling namespaces in lxml, use the namespace map from the parsed document rather than hardcoding URIs.
- Preserve all Korean labels and static note lines — only replace `{{...}}` patterns.
- Use `zipfile.ZIP_DEFLATED` compression when rewriting to maintain compatibility.
- If lxml is not available, install it with `pip install lxml`.

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