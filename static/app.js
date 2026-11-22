const form = document.querySelector('#prediction-form');
const select = document.querySelector('#examples');
const statusLine = document.querySelector('#result-status');
const observed = document.querySelector('#observed');
const labels = {metadata: 'Metadata', text: 'Synopsis', combined: 'Combined'};
let examples = [];
let chosenExample = null;

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}

async function initialise() {
  const exampleResponse = await fetch('/api/examples');
  examples = await exampleResponse.json();
  examples.forEach((item, index) => {
    const option = element('option', '', item.name);
    option.value = String(index);
    select.append(option);
  });
  const response = await fetch('/api/experiment');
  const data = await response.json();
  if (!response.ok) {
    document.querySelector('#experiment-conclusion').textContent = data.error;
    return;
  }
  const table = element('table');
  const head = element('tr');
  ['Inputs', 'Outer CV', 'Test'].forEach(value => head.append(element('th', '', value)));
  table.append(head);
  for (const [role, model] of Object.entries(data.models)) {
    const row = element('tr');
    [labels[role], Number(model.cv_f1).toFixed(3), Number(model.test_f1).toFixed(3)].forEach(value => row.append(element('td', '', value)));
    table.append(row);
  }
  if (data.controlled_metadata) {
    const row = element('tr');
    ['Metadata LR control', Number(data.controlled_metadata.cv_f1).toFixed(3),
      Number(data.controlled_metadata.test_f1).toFixed(3)].forEach(value => row.append(element('td', '', value)));
    table.append(row);
  }
  document.querySelector('#experiment-table').append(table);
  document.querySelector('#experiment-conclusion').textContent = data.conclusion;
}

select.addEventListener('change', () => {
  document.querySelector('#prediction-cards').classList.add('stale');
  chosenExample = select.value === '' ? null : examples[Number(select.value)];
  if (!chosenExample) {
    form.reset();
    document.querySelector('#name').value = '';
    observed.hidden = true;
    statusLine.textContent = 'Enter your own series and run a new prediction.';
    return;
  }
  document.querySelector('#name').value = chosenExample.name;
  for (const input of form.querySelectorAll('[name]')) {
    const value = chosenExample[input.name];
    input.value = Array.isArray(value) ? value.join(', ') : (value ?? '');
  }
  observed.hidden = true;
  statusLine.textContent = 'Held-out snapshot example loaded. You can edit its inputs before predicting.';
});

form.addEventListener('input', () => {
  chosenExample = null;
  observed.hidden = true;
  document.querySelector('#prediction-cards').classList.add('stale');
  statusLine.textContent = 'Inputs changed. Run prediction to update the results.';
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = form.querySelector('button');
  const error = document.querySelector('#error');
  error.hidden = true;
  button.disabled = true;
  statusLine.textContent = 'Applying the saved models…';
  const payload = Object.fromEntries(new FormData(form));
  payload.genres = payload.genres.split(',').map(value => value.trim()).filter(Boolean);
  try {
    const response = await fetch('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The prediction could not be completed.');
    const cards = document.querySelector('#prediction-cards');
    cards.replaceChildren();
    cards.classList.remove('stale');
    result.models.forEach(model => {
      const card = element('article', `prediction-card ${model.available ? (model.predicted ? 'positive' : 'negative') : 'empty'}`);
      const label = element('div', 'card-label', labels[model.role]);
      if (model.role === result.primary_role) label.append(element('span', 'selected-label', 'CV-selected primary'));
      card.append(label);
      if (!model.available) {
        card.append(element('p', 'empty-score', '—'), element('p', '', 'Add a synopsis to use this model.'));
      } else {
        card.append(element('p', 'outcome', model.label));
        const digits = Math.abs(model.score - model.threshold) < 0.0005 ? 6 : 3;
        card.append(element('div', 'score-value', model.score.toFixed(digits)));
        card.append(element('p', '', `${model.score_kind} · threshold ${model.threshold} · ${model.family.replaceAll('_', ' ')}`));
        if (model.score_kind === 'model score') {
          const track = element('div', 'score-track');
          const fill = element('div', 'score-fill');
          fill.style.width = `${100 * Math.max(0, Math.min(1, model.score))}%`;
          track.append(fill); card.append(track);
        }
      }
      cards.append(card);
    });
    statusLine.textContent = result.has_text ? 'Saved-model inference complete.' : 'Metadata prediction complete. Add a synopsis to compare all three inputs.';
    if (chosenExample) {
      observed.textContent = `Snapshot reference for ${chosenExample.name}: ${chosenExample.observed_rating}/10. This observed rating was excluded from the model input.`;
      observed.hidden = false;
    }
  } catch (problem) {
    error.textContent = problem.message;
    error.hidden = false;
    statusLine.textContent = 'Check the input and try again.';
  } finally {
    button.disabled = false;
  }
});

initialise().catch(() => {document.querySelector('#experiment-conclusion').textContent = 'Could not load the experiment. Check that the local server is running.';});
