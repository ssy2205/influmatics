'use strict';

// Three-seed mean absolute errors reported in Blueprint, Main Table 2.
const periods = {
  all: {
    label: '4개년 평균',
    scores: [0.5541, 0.5776, 0.5582, 0.5641, 0.5889],
    observation: '기존 기준 모델과 B2의 4개년 평균 오차 차이는 0.0041 AU였습니다. 2019년부터 차례로 선택하면 두 모델의 순위가 바뀌는 시점을 확인할 수 있습니다.'
  },
  2019: {
    label: '2019년',
    scores: [0.5676, 0.6438, 0.6377, 0.6402, 0.6489],
    observation: '2018년까지의 자료로 학습했습니다. 이듬해 자료에서는 기존 기준 모델의 평균 오차가 가장 작았습니다.'
  },
  2020: {
    label: '2020년',
    scores: [0.6472, 0.7202, 0.6786, 0.6965, 0.7373],
    observation: '2019년까지의 자료로 학습했습니다. 다섯 모델 모두 2019년 평가 때보다 오차가 커졌습니다.'
  },
  2021: {
    label: '2021년',
    scores: [0.5109, 0.4881, 0.4793, 0.4805, 0.4950],
    observation: '2020년까지의 자료로 학습했습니다. 이 해에는 영역 강조 모델인 B2에서 가장 낮은 오차를 관측했습니다.'
  },
  2022: {
    label: '2022년',
    scores: [0.4907, 0.4582, 0.4370, 0.4392, 0.4746],
    observation: '2021년까지의 자료로 학습했습니다. 2021년 평가에 이어 B2의 평균 오차가 가장 낮았습니다.'
  }
};

const picker = document.getElementById('period-picker');
const buttons = Array.from(picker.querySelectorAll('button'));
const plot = document.getElementById('score-plot');
const rows = Array.from(plot.querySelectorAll('.score-row'));

function selectPeriod(key) {
  const { scores, label, observation } = periods[key];
  const minimum = Math.min(...scores);
  const selected = rows[scores.indexOf(minimum)];
  document.getElementById('period-label').textContent = `${label} · 최저 오차`;
  document.getElementById('score-value').textContent = minimum.toFixed(4);
  document.getElementById('lowest-model').textContent = `${selected.dataset.model} · ${selected.querySelector('span').textContent}`;
  document.getElementById('period-story').textContent = observation;
  plot.setAttribute('aria-label', `${label} 오차 비교. 막대가 짧을수록 오차가 작습니다.`);
  rows.forEach((row, index) => {
    const score = scores[index];
    row.classList.toggle('best', score === minimum);
    row.style.setProperty('--width', `${(score / 0.8) * 100}%`);
    row.querySelector('strong').textContent = score.toFixed(4);
  });
  buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.period === key)));
}

buttons.forEach(button => button.addEventListener('click', () => selectPeriod(button.dataset.period)));
picker.hidden = false;
