/**
 * Typed bubble-diagram user study - Google Form builder (Google Apps Script).
 *
 * Model: ONE plan per student. The 100 plans are split into groups of
 * PLANS_PER_STUDENT (default 1); one Google Form is built per group, so 100
 * students cover the 100 plans. Each form shows, for its plan(s): three paired
 * images (the floor plan beside the proximity matrix, the proximity chart, and
 * the bubble diagram) and a set of rating questions plus two quality checks. Every form link (and the
 * attention-check answer key) is written to a Google Sheet for you to distribute.
 *
 * SETUP (one-time)
 *  1. Unzip study_images.zip. Create a Google Drive folder and upload every image
 *     into it. Each plan <stem> has three paired PNGs (floor plan beside each output):
 *        <stem>_pair_chart.png  (floor plan + proximity chart, the diamond view)
 *        <stem>_pair_matrix.png (floor plan + adjacency matrix, the square view)
 *        <stem>_pair_bubble.png (floor plan + bubble diagram)
 *  2. Open the Drive folder; copy its ID from the address bar
 *     (drive.google.com/drive/folders/THIS_ID) into FOLDER_ID below.
 *  3. script.google.com -> New project, paste this whole file, Save.
 *  4. Run -> buildForms. Approve the Drive, Forms and Sheets permissions.
 *  5. It builds forms in batches (the 6-min limit). When the log says "RUN
 *     buildForms AGAIN to continue", run buildForms again until it logs DONE.
 *     The DONE message links a "Study form links" Sheet with one row per form
 *     (form #, plan id, correct room count, edit URL, respond URL).
 *  6. resetProgress starts a fresh set; buildTestForm builds one form to preview.
 *
 * QUALITY CONTROL (foolproofing)
 *  - Comprehension check: each page asks the rater to count the labelled rooms in
 *    the plan; the correct answer is in the links Sheet - discard raters who miss it.
 *  - Attention check: one instructed-response item ("select Strongly agree");
 *    discard raters who do not.
 */

var FOLDER_ID = 'PASTE_YOUR_DRIVE_FOLDER_ID_HERE';
var PLANS_PER_STUDENT = 1;    // plans per student (one form per group)
var FORMS_PER_RUN = 20;       // forms per execution (keeps under the 6-min limit)
var FORM_TITLE = 'Floor-plan diagram evaluation study';

// The 100 plan ids; shuffled once, then split into groups (the randomiser).
var STEMS = [
  '6153', '5744', '6608', '2748', '2814', '12359', '13277', '2248', '8161', '4036',
  '5538', '14124', '11855', '4045', '14041', '10591', '2864', '3675', '3412', '4632',
  '5706', '10194', '16356', '5259', '5251', '16532', '1670', '7161', '15295', '11647',
  '959', '5674', '944', '12164', '5445', '5005', '15705', '7101', '16192', '9347',
  '13616', '16002', '2480', '4992', '11655', '8890', '4387', '3313', '8134', '10296',
  '6617', '17017', '3378', '12487', '4063', '7533', '6103', '3156', '3667', '12400',
  '1838', '7422', '5178', '5575', '10211', '12686', '12175', '2388', '7296', '4726',
  '7906', '2441', '3084', '4494', '12106', '625', '3261', '15674', '15677', '1968',
  '1254', '2521', '1295', '1781', '9961', '695', '5186', '6000', '13284', '1293',
  '8563', '12308', '1136', '15707', '822', '11215', '651', '279', '9675', '145'];

// Correct number of labelled rooms per plan (comprehension-check answer key).
var ROOMCOUNT = {
  '6153': 6, '5744': 6, '6608': 6, '2748': 6, '2814': 6, '12359': 5, '13277': 6, '2248': 6,
  '8161': 5, '4036': 6, '5538': 5, '14124': 5, '11855': 6, '4045': 6, '14041': 6, '10591': 6,
  '2864': 6, '3675': 6, '3412': 6, '4632': 6, '5706': 6, '10194': 6, '16356': 6, '5259': 5,
  '5251': 5, '16532': 7, '1670': 7, '7161': 7, '15295': 7, '11647': 8, '959': 7, '5674': 7,
  '944': 7, '12164': 8, '5445': 8, '5005': 8, '15705': 8, '7101': 8, '16192': 8, '9347': 7,
  '13616': 8, '16002': 7, '2480': 8, '4992': 8, '11655': 7, '8890': 8, '4387': 8, '3313': 8,
  '8134': 8, '10296': 7, '6617': 9, '17017': 10, '3378': 10, '12487': 9, '4063': 9, '7533': 9,
  '6103': 9, '3156': 9, '3667': 9, '12400': 9, '1838': 9, '7422': 9, '5178': 9, '5575': 10,
  '10211': 9, '12686': 10, '12175': 9, '2388': 9, '7296': 9, '4726': 9, '7906': 9, '2441': 10,
  '3084': 10, '4494': 9, '12106': 10, '625': 13, '3261': 11, '15674': 12, '15677': 16, '1968': 11,
  '1254': 13, '2521': 11, '1295': 15, '1781': 11, '9961': 17, '695': 12, '5186': 11, '6000': 12,
  '13284': 13, '1293': 12, '8563': 11, '12308': 12, '1136': 11, '15707': 12, '822': 14, '11215': 12,
  '651': 13, '279': 15, '9675': 13, '145': 11
};

// Rating questions. Each: [title, help text]. 1 = Strongly disagree, 5 = Strongly agree.
var CONTENT_Q = [
  ['Room identification', 'The rooms are labelled with the correct types (living, kitchen, bedroom, bathroom, balcony, etc.).'],
  ['Completeness', 'Every room in the floor plan appears in the diagrams - none is missing and none is invented.'],
  ['Adjacency correctness', 'The connections between rooms match the adjacencies you can see in the floor plan - no missing or extra connections.'],
  ['Connection-type correctness', 'Each connection is given the correct type - door, open passage, or shared wall.']
];
var READ_Q = [
  ['Bubble diagram - readability', 'The bubble diagram is clear and easy to read.'],
  ['Proximity chart - readability', 'The proximity chart (diamond grid) is clear and easy to read.'],
  ['Adjacency matrix - readability', 'The adjacency matrix (square grid) is clear and easy to read.']
];

function shuffle_(a){for(var i=a.length-1;i>0;i--){var j=Math.floor(Math.random()*(i+1));var t=a[i];a[i]=a[j];a[j]=t;}return a;}

function imgBlob_(folder, name){
  var it = folder.getFilesByName(name);
  if(!it.hasNext()) throw new Error('Image not found in the Drive folder: ' + name);
  return it.next().getBlob();
}

function scale_(form, title, help){
  form.addScaleItem().setTitle(title).setHelpText(help)
      .setBounds(1,5).setLabels('Strongly disagree','Strongly agree').setRequired(true);
}

function addPlanPage_(form, folder, stem, k, total){
  form.addPageBreakItem().setTitle('Floor plan ' + k + (total>1 ? ' of ' + total : ''));
  form.addImageItem().setTitle('Floor plan with proximity chart').setImage(imgBlob_(folder, stem + '_pair_chart.png'));
  form.addImageItem().setTitle('Floor plan with adjacency matrix').setImage(imgBlob_(folder, stem + '_pair_matrix.png'));
  form.addImageItem().setTitle('Floor plan with bubble diagram').setImage(imgBlob_(folder, stem + '_pair_bubble.png'));

  // comprehension check (answer key = ROOMCOUNT[stem], in the links sheet)
  form.addTextItem().setTitle('Check: how many rooms are labelled in the floor plan above? (enter a number)')
      .setRequired(true);

  // content correctness
  for (var i=0;i<CONTENT_Q.length;i++) scale_(form, CONTENT_Q[i][0], CONTENT_Q[i][1]);
  // per-output readability
  for (var j=0;j<READ_Q.length;j++) scale_(form, READ_Q[j][0], READ_Q[j][1]);
  // overall
  scale_(form, 'Overall plausibility', 'Taken together, the three diagrams are a plausible representation of this floor plan.');
  // preference
  form.addMultipleChoiceItem().setTitle('Which representation did you find easiest to understand?')
      .setChoiceValues(['Bubble diagram','Proximity matrix (diamond)','Proximity chart (square)','All equally clear','None was clear'])
      .setRequired(true);
  // confidence
  form.addScaleItem().setTitle('How confident are you in your ratings for this plan?')
      .setBounds(1,5).setLabels('Not at all confident','Very confident').setRequired(true);
  // attention check (instructed response; correct answer = 5)
  form.addScaleItem().setTitle('Attention check: please select "Strongly agree" (5) for this item.')
      .setBounds(1,5).setLabels('Strongly disagree','Strongly agree').setRequired(true);
  // free text
  form.addParagraphTextItem().setTitle('Optional comments on this plan');
}

function buildOneForm_(folder, stems, setNum, setTotal){
  var title = FORM_TITLE + (setTotal>1 ? ' (set ' + setNum + ' of ' + setTotal + ')' : '');
  var form = FormApp.create(title);
  form.setDescription('You will be shown a floor plan and three diagrams derived from it: a proximity '
    + 'matrix, a proximity chart, and a bubble diagram. Please rate how well the diagrams represent the '
    + 'plan. There are no right or wrong answers - we value your professional judgement.');
  form.setProgressBar(true).setCollectEmail(false).setShowLinkToRespondAgain(false);

  form.addSectionHeaderItem().setTitle('Consent and participant information');
  form.addMultipleChoiceItem().setTitle('I consent to take part in this study.')
      .setChoiceValues(['Yes, I consent','No']).setRequired(true);
  form.addTextItem().setTitle('Participant ID (if one was given to you)');
  form.addListItem().setTitle('Your role')
      .setChoiceValues(['Architecture student','Practising architect','Other']).setRequired(true);
  form.addListItem().setTitle('Years of architectural training or experience')
      .setChoiceValues(['Less than 1','1 to 3','4 to 6','More than 6']).setRequired(true);
  form.addScaleItem().setTitle('Your familiarity with bubble diagrams')
      .setBounds(1,5).setLabels('Not at all familiar','Very familiar').setRequired(true);

  form.addSectionHeaderItem().setTitle('How to read the diagrams').setHelpText(
      'Floor plan: rooms are coloured and labelled (Living, Kitchen, Bedroom, etc.). '
    + 'Bubble diagram: each bubble is a room, coloured to match the plan; edges are '
    + 'thin solid black = door, thick purple = open passage (a gap with no door), '
    + 'dotted grey = shared wall (a wall with no opening). '
    + 'Proximity chart (diamond) and adjacency matrix (square): each pair of rooms is coloured - '
    + 'green = door (D), purple = open passage (OP), orange = shared wall (SW), light grey = not connected.');

  for (var i=0;i<stems.length;i++) addPlanPage_(form, folder, stems[i], i+1, stems.length);
  return form;
}

/** Build the forms (resumable). Run repeatedly until it logs DONE. */
function buildForms(){
  if (FOLDER_ID === 'PASTE_YOUR_DRIVE_FOLDER_ID_HERE')
    throw new Error('Set FOLDER_ID to your Drive image-folder ID first.');
  var folder = DriveApp.getFolderById(FOLDER_ID);
  var props = PropertiesService.getScriptProperties();
  var groups = JSON.parse(props.getProperty('GROUPS') || 'null');
  var sheetId = props.getProperty('SHEET_ID');
  var idx = parseInt(props.getProperty('IDX') || '0', 10);
  if (!groups){
    var s = shuffle_(STEMS.slice()); groups = [];
    for (var i=0;i<s.length;i+=PLANS_PER_STUDENT) groups.push(s.slice(i, i+PLANS_PER_STUDENT));
    props.setProperty('GROUPS', JSON.stringify(groups));
    var ss = SpreadsheetApp.create('Study form links');
    ss.getActiveSheet().appendRow(['Form #','Plan id(s)','Room-count answer(s)','Edit URL','Respond URL']);
    sheetId = ss.getId(); props.setProperty('SHEET_ID', sheetId);
    idx = 0; props.setProperty('IDX','0');
  }
  var sheet = SpreadsheetApp.openById(sheetId).getActiveSheet();
  var urls = JSON.parse(props.getProperty('FORM_URLS') || '[]');   // rotation order for the router
  var end = Math.min(idx + FORMS_PER_RUN, groups.length);
  for (var g=idx; g<end; g++){
    var form = buildOneForm_(folder, groups[g], g+1, groups.length);
    var ans = groups[g].map(function(s){ return ROOMCOUNT[s]; }).join(' ');
    urls.push(form.getPublishedUrl());
    sheet.appendRow([g+1, groups[g].join(' '), ans, form.getEditUrl(), form.getPublishedUrl()]);
  }
  props.setProperty('FORM_URLS', JSON.stringify(urls));
  props.setProperty('IDX', String(end));
  if (end < groups.length){
    Logger.log('Built forms ' + (idx+1) + '-' + end + ' of ' + groups.length + '. RUN buildForms AGAIN to continue.');
  } else {
    Logger.log('DONE: built all ' + groups.length + ' forms.');
    Logger.log('Links sheet: ' + SpreadsheetApp.openById(sheetId).getUrl());
    Logger.log('Now Deploy > New deployment > Web app (Execute as: me; Who has access: Anyone) and share the');
    Logger.log('web-app URL with participants - each visit is auto-assigned the next plan (no repeat until all ' + groups.length + ').');
    props.deleteProperty('IDX'); props.deleteProperty('GROUPS'); props.deleteProperty('SHEET_ID');
    // FORM_URLS is kept - the doGet router uses it for rotation.
  }
}

function resetProgress(){
  var p = PropertiesService.getScriptProperties();
  ['IDX','GROUPS','SHEET_ID','FORM_URLS','ROT'].forEach(function(k){ p.deleteProperty(k); });
  Logger.log('Progress reset (including built-form list and rotation). Run buildForms to start over.');
}

/**
 * ROUND-ROBIN ROUTER (web app). Distribute the web-app URL to all participants.
 * Each visit is redirected to the next form in order; no plan repeats until all
 * have been used once, then the cycle repeats. Deploy AFTER buildForms finishes:
 *   Deploy > New deployment > type Web app > Execute as: Me > Who has access: Anyone.
 * resetRotation() sends the next visitor back to the first form.
 */
function doGet(e){
  var props = PropertiesService.getScriptProperties();
  var urls = JSON.parse(props.getProperty('FORM_URLS') || '[]');
  if (!urls.length)
    return HtmlService.createHtmlOutput('<p style="font-family:Arial">The study is not ready yet. Please check back shortly.</p>');
  var lock = LockService.getScriptLock(); lock.waitLock(15000);
  var c = parseInt(props.getProperty('ROT') || '0', 10);
  props.setProperty('ROT', String(c + 1));
  lock.releaseLock();
  var url = urls[c % urls.length];
  var html = '<!DOCTYPE html><html><head><base target="_top">'
    + '<meta http-equiv="refresh" content="0; url=' + url + '">'
    + '</head><body style="font-family:Arial;text-align:center;margin-top:48px">'
    + '<p>Loading your floor plan, please wait...</p>'
    + '<p>If you are not redirected, <a target="_top" href="' + url + '">click here to start</a>.</p>'
    + '<script>window.top.location.href=' + JSON.stringify(url) + ';</script>'
    + '</body></html>';
  return HtmlService.createHtmlOutput(html).setTitle(FORM_TITLE);
}

function resetRotation(){
  PropertiesService.getScriptProperties().deleteProperty('ROT');
  Logger.log('Rotation reset: the next visitor gets the first form again.');
}

function buildTestForm(){
  var folder = DriveApp.getFolderById(FOLDER_ID);
  var form = buildOneForm_(folder, [STEMS[0]], 1, 1);
  Logger.log('Test form: ' + form.getEditUrl());
}
