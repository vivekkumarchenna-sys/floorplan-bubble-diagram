# Google Form user study (typed bubble-diagram evaluation)

> **Scope note.** This folder holds the instrument for a planned follow-up perceived-quality study (one plan per participant, 100 plans). It is **not** the study reported in the paper. The user study reported in Section 6.8 and Appendix E (20 raters, 20 stimuli, paired Wilcoxon analysis) is reproduced by `src/user_study_stats.py` from `User_Studies/original_pilot_study/user_study_workbook_latest.xlsx` in the Zenodo archive.

Each student rates **one floor plan**: they see three paired images (the floor
plan beside the proximity matrix, the proximity chart, and the bubble diagram)
and rate content correctness, per-output readability, an overall score and a
preference, with two quality-control checks. The 100 plans are split one-per-form,
so 100 students cover 100 plans.

## Files
- `user_study_form.gs` - Google Apps Script that builds the forms (100 plan ids embedded).
- `study_images.zip` - 300 images: for each of the 100 plans, three paired PNGs named
  `<stem>_pair_matrix.png`, `<stem>_pair_chart.png`, `<stem>_pair_bubble.png`
  (each shows the labelled floor plan beside one output).

## Questions per plan
- Comprehension check (count the labelled rooms; answer key is in the links sheet)
- Content: room identification, completeness, adjacency correctness, connection-type correctness (1-5)
- Readability of each output: bubble diagram, proximity matrix, proximity chart (1-5)
- Overall plausibility (1-5); which representation was easiest (choice)
- Attention check (instructed response); optional comments

## How to build the forms
1. Unzip `study_images.zip`. Create a folder in Google Drive and upload all 300 images into it.
2. Open the Drive folder and copy its ID from the address bar
   (`drive.google.com/drive/folders/THIS_ID`).
3. Go to `script.google.com`, create a new project, and paste in `user_study_form.gs`.
4. Set `FOLDER_ID` at the top of the script to the ID from step 2.
5. Run the `buildForms` function. Approve the Drive, Forms and Sheets permissions.
6. Apps Script stops each run after a few minutes, so the script builds the forms in
   batches. When the log says "RUN buildForms AGAIN to continue", run `buildForms`
   again; repeat until it logs **DONE**. The DONE message links a Google Sheet
   ("Study form links") with one row per form (form number, plan id, edit URL, respond URL).
7. Give each student a different **Respond URL** from that sheet - one plan per student.

## Options (top of the script)
- `PLANS_PER_STUDENT` - plans shown to each student. `1` (default) = one plan per
  form, so 100 forms. Set to e.g. `5` to build 20 forms of 5 plans each.
- `FORMS_PER_RUN` - how many forms to build per execution (default 20); lower it if a
  run ever times out.
- `buildTestForm` - builds a single one-plan form so you can preview the layout first.
- `resetProgress` - clears saved progress so `buildForms` starts a fresh set.

## Notes
- The plan order is shuffled once, when the first `buildForms` run starts (the randomiser).
- Images are copied into each form when it is built, so the Drive folder can be
  archived afterwards.
- The five dimensions match the earlier study: plausibility, adjacency correctness,
  connection-type accuracy, room-type accuracy, and readability.
