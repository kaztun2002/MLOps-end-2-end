"""Evidently-based model and data drift monitoring for the sales forecaster."""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.core.datasets import DataDefinition, Dataset, Regression
from evidently.presets import DataDriftPreset, RegressionPreset
from sklearn.metrics import (
	explained_variance_score,
	mean_absolute_error,
	mean_squared_error,
	median_absolute_error,
	r2_score,
)

from mlops_end_2_end.utils import get_logger


class EvidentlyModelMonitoring:
	"""Generate simulated drift runs and monitor regression quality with Evidently."""

	def __init__(
		self,
		project_root: str | Path | None = None,
		data_path: str | Path | None = None,
		model_path: str | Path | None = None,
		output_dir: str | Path | None = None,
		target_column: str = "sales",
		datetime_column: str = "date",
		n_runs: int = 10,
		random_state: int = 42,
	) -> None:
		self.project_root = Path(project_root) if project_root else Path.cwd()
		self.data_path = Path(data_path) if data_path else (
			self.project_root / "data" / "engineered_features" / "engineered_features.csv"
		)
		self.model_path = Path(model_path) if model_path else (
			self.project_root / "models" / "trained" / "lightgbm_model.pkl"
		)
		self.output_dir = Path(output_dir) if output_dir else (
			self.project_root / "reports" / "monitoring"
		)
		self.target_column = target_column
		self.datetime_column = datetime_column
		self.n_runs = n_runs
		self.random_state = random_state
		self.logger = get_logger("model_monitoring")
		self.dashboard_path = self.output_dir / "monitoring_dashboard.html"

		if self.n_runs < 1:
			raise ValueError("n_runs must be at least 1.")

	def _load_inputs(self) -> tuple[pd.DataFrame, Any]:
		if not self.data_path.is_file():
			raise FileNotFoundError(
				f"Feature data not found at {self.data_path}. Run feature engineering first."
			)
		if not self.model_path.is_file():
			raise FileNotFoundError(
				f"Trained model not found at {self.model_path}. Run model training first."
			)

		data = pd.read_csv(self.data_path)
		required_columns = {self.target_column, self.datetime_column}
		missing_columns = required_columns.difference(data.columns)
		if missing_columns:
			raise ValueError(
				f"Feature data is missing required columns: {sorted(missing_columns)}"
			)

		data[self.datetime_column] = pd.to_datetime(data[self.datetime_column])
		data = data.sort_values(self.datetime_column).reset_index(drop=True)
		with self.model_path.open("rb") as model_file:
			model = pickle.load(model_file)
		return data, model

	def _regression_metrics(
		self,
		actual: pd.Series | np.ndarray,
		predicted: np.ndarray,
		feature_count: int,
		naive_scale: float,
	) -> dict[str, float | None]:
		actual_values = np.asarray(actual, dtype=float)
		predicted_values = np.asarray(predicted, dtype=float)
		errors = predicted_values - actual_values
		absolute_errors = np.abs(errors)
		epsilon = np.finfo(float).eps
		sample_count = len(actual_values)

		mse = float(mean_squared_error(actual_values, predicted_values))
		rmse = float(np.sqrt(mse))
		r2 = float(r2_score(actual_values, predicted_values))
		adjusted_r2 = None
		if sample_count > feature_count + 1:
			adjusted_r2 = float(
				1 - (1 - r2) * (sample_count - 1) / (sample_count - feature_count - 1)
			)

		actual_nonnegative = np.maximum(actual_values, 0)
		predicted_nonnegative = np.maximum(predicted_values, 0)
		rmsle = float(
			np.sqrt(
				np.mean(
					(
						np.log1p(actual_nonnegative)
						- np.log1p(predicted_nonnegative)
					) ** 2
				)
			)
		)

		actual_range = float(np.ptp(actual_values))
		actual_mean_absolute = float(np.mean(np.abs(actual_values)))
		return {
			"MAE": float(mean_absolute_error(actual_values, predicted_values)),
			"MSE": mse,
			"RMSE": rmse,
			"R2": r2,
			"Adjusted R2": adjusted_r2,
			"Explained variance": float(
				explained_variance_score(actual_values, predicted_values)
			),
			"MAPE (%)": float(
				np.mean(absolute_errors / np.maximum(np.abs(actual_values), epsilon)) * 100
			),
			"sMAPE (%)": float(
				np.mean(
					2 * absolute_errors
					/ np.maximum(np.abs(actual_values) + np.abs(predicted_values), epsilon)
				)
				* 100
			),
			"WAPE (%)": float(
				np.sum(absolute_errors)
				/ max(float(np.sum(np.abs(actual_values))), epsilon)
				* 100
			),
			"RMSLE": rmsle,
			"Median absolute error": float(
				median_absolute_error(actual_values, predicted_values)
			),
			"Maximum absolute error": float(np.max(absolute_errors)),
			"Mean error (bias)": float(np.mean(errors)),
			"Median error": float(np.median(errors)),
			"MASE": float(np.mean(absolute_errors) / naive_scale)
			if naive_scale > epsilon
			else None,
			"NRMSE (range, %)": float(rmse / actual_range * 100)
			if actual_range > epsilon
			else None,
			"NRMSE (mean, %)": float(rmse / actual_mean_absolute * 100)
			if actual_mean_absolute > epsilon
			else None,
		}

	def _simulate_drift(
		self,
		reference: pd.DataFrame,
		severity: float,
		run_number: int,
	) -> pd.DataFrame:
		rng = np.random.default_rng(self.random_state + run_number)
		current = reference.copy()

		for column in reference.columns:
			values = reference[column]
			if pd.api.types.is_numeric_dtype(values.dtype):
				numeric_values = values.to_numpy(dtype=float)
				unique_values = set(np.unique(numeric_values))
				if unique_values.issubset({0.0, 1.0}):
					flip_rate = min(0.45, 0.05 + severity * 0.4)
					flip_mask = rng.random(len(values)) < flip_rate
					current[column] = np.where(
						flip_mask,
						1.0 - numeric_values,
						numeric_values,
					)
					continue

				scale = float(np.std(numeric_values))
				if scale <= np.finfo(float).eps:
					scale = max(abs(float(np.mean(numeric_values))), 1.0)
				shift = scale * severity * 0.8
				noise = rng.normal(0, scale * severity * 0.2, len(values))
				current[column] = numeric_values + shift + noise
			else:
				categories = values.dropna().unique()
				if len(categories) > 1:
					replace_mask = rng.random(len(values)) < min(0.45, severity * 0.4)
					replacement = categories[-1]
					current.loc[replace_mask, column] = replacement

		return current

	@staticmethod
	def _snapshot_drift_summary(snapshot: Any) -> dict[str, Any]:
		result = snapshot.dict()
		summary: dict[str, Any] = {
			"drifted_columns": 0,
			"drift_share": 0.0,
			"prediction_drift_p_value": None,
		}

		for metric in result.get("metrics", []):
			config = metric.get("config", {})
			value = metric.get("value")
			metric_type = config.get("type", "").rsplit(":", 1)[-1]
			if metric_type == "DriftedColumnsCount" and isinstance(value, dict):
				summary["drifted_columns"] = int(round(float(value.get("count", 0))))
				summary["drift_share"] = float(value.get("share", 0.0))
			elif metric_type == "ValueDrift" and config.get("column") == "prediction":
				summary["prediction_drift_p_value"] = float(value)

		p_value = summary["prediction_drift_p_value"]
		summary["prediction_drift_detected"] = p_value is not None and p_value < 0.05
		return summary

	def run(self) -> pd.DataFrame:
		"""Run ten (by default) progressively stronger drift simulations and save reports."""
		data, model = self._load_inputs()
		validation_start = max(1, int(len(data) * 0.8))
		validation = data.iloc[validation_start:].copy()
		if validation.empty:
			raise ValueError("At least two rows are required for monitoring.")

		feature_columns = [
			column
			for column in data.columns
			if column not in {self.target_column, self.datetime_column}
		]
		expected_features = list(getattr(model, "feature_name_", feature_columns))
		missing_features = set(expected_features).difference(feature_columns)
		if missing_features:
			raise ValueError(
				"The engineered data does not contain all model features: "
				f"{sorted(missing_features)}"
			)

		reference_features = validation.reindex(columns=expected_features).reset_index(drop=True)
		actual = validation[self.target_column].astype(float).reset_index(drop=True)
		baseline_predictions = np.asarray(model.predict(reference_features), dtype=float)
		naive_scale = float(np.mean(np.abs(np.diff(actual.to_numpy()))))
		baseline_metrics = self._regression_metrics(
			actual,
			baseline_predictions,
			len(expected_features),
			naive_scale,
		)

		self.output_dir.mkdir(parents=True, exist_ok=True)
		detail_dir = self.output_dir / "evidently_runs"
		detail_dir.mkdir(parents=True, exist_ok=True)
		reference_report_data = reference_features.copy()
		reference_report_data["target"] = actual
		reference_report_data["prediction"] = baseline_predictions

		numerical_columns = list(
			reference_report_data.select_dtypes(include=["number", "bool"]).columns
		)
		categorical_columns = [
			column
			for column in reference_report_data.columns
			if column not in numerical_columns
		]
		data_definition = DataDefinition(
			regression=[Regression(name="sales_forecast", target="target", prediction="prediction")],
			numerical_columns=numerical_columns,
			categorical_columns=categorical_columns,
		)
		monitored_columns = [*expected_features, "prediction"]
		reference_dataset = Dataset.from_pandas(
			reference_report_data,
			data_definition=data_definition,
		)

		runs: list[dict[str, Any]] = []
		summary_rows: list[dict[str, Any]] = []
		severities = np.linspace(1 / self.n_runs, 1.0, self.n_runs)

		for run_number, severity in enumerate(severities, start=1):
			drifted_features = self._simulate_drift(
				reference_features,
				float(severity),
				run_number,
			)
			predictions = np.asarray(model.predict(drifted_features), dtype=float)
			current_report_data = drifted_features.copy()
			current_report_data["target"] = actual.to_numpy()
			current_report_data["prediction"] = predictions
			current_dataset = Dataset.from_pandas(
				current_report_data,
				data_definition=data_definition,
			)

			report = Report(
				metrics=[
					DataDriftPreset(
						columns=monitored_columns,
						num_method="ks",
					),
					RegressionPreset(
						regression_name="sales_forecast",
					),
				],
				metadata={
					"run": f"{run_number:02d}",
					"drift_severity": f"{severity:.2f}",
					"scenario": "synthetic covariate and prediction drift",
				},
			)
			snapshot = report.run(
				current_data=current_dataset,
				reference_data=reference_dataset,
				name=f"Simulated drift run {run_number:02d}",
				tags=["simulated", "regression", f"run-{run_number:02d}"],
			)

			detail_name = f"run_{run_number:02d}.html"
			detail_path = detail_dir / detail_name
			snapshot.save_html(str(detail_path))
			drift_summary = self._snapshot_drift_summary(snapshot)
			metrics = self._regression_metrics(
				actual,
				predictions,
				len(expected_features),
				naive_scale,
			)
			run_result = {
				"run": run_number,
				"severity": float(severity),
				**drift_summary,
				"metrics": metrics,
				"report": f"evidently_runs/{detail_name}",
			}
			runs.append(run_result)
			summary_rows.append(
				{
					"Run": run_number,
					"Drift severity": float(severity),
					"Drifted columns": drift_summary["drifted_columns"],
					"Drift share": drift_summary["drift_share"],
					"Prediction drift p-value": drift_summary[
						"prediction_drift_p_value"
					],
					"Prediction drift detected": drift_summary[
						"prediction_drift_detected"
					],
					**metrics,
				}
			)
			self.logger.info(
				"Monitoring run %02d/%02d complete: severity=%.2f, drifted=%d/%d, RMSE=%.3f",
				run_number,
				self.n_runs,
				severity,
				drift_summary["drifted_columns"],
				len(monitored_columns),
				metrics["RMSE"],
			)

		summary = pd.DataFrame(summary_rows)
		summary.to_csv(self.output_dir / "monitoring_runs.csv", index=False)
		dashboard_data = {
			"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
			"reference_rows": len(reference_features),
			"feature_count": len(expected_features),
			"baseline_metrics": baseline_metrics,
			"metric_names": list(baseline_metrics),
			"runs": runs,
		}
		(self.output_dir / "monitoring_runs.json").write_text(
			json.dumps(dashboard_data, indent=2, allow_nan=False),
			encoding="utf-8",
		)
		self.dashboard_path.write_text(
			self._render_dashboard(dashboard_data),
			encoding="utf-8",
		)
		self.logger.info("Monitoring dashboard written to %s", self.dashboard_path)
		return summary

	@staticmethod
	def _render_dashboard(data: dict[str, Any]) -> str:
		payload = json.dumps(data, allow_nan=False).replace("</", "<\\/")
		return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model monitoring | Sales forecast</title>
<style>
:root{color-scheme:light;--ink:#17262d;--muted:#64757c;--paper:#f2f5f3;--panel:#fff;--line:#dce5e1;--green:#a5d86e;--green-dark:#48733b;--cyan:#2c8490;--coral:#d8755d;--shadow:0 18px 50px rgba(27,48,48,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 "Segoe UI",sans-serif}button,select{font:inherit}a{color:var(--cyan);font-weight:700;text-decoration:none}a:hover{text-decoration:underline}.shell{max-width:1420px;margin:auto;padding:34px clamp(18px,4vw,60px) 64px}.topline{display:flex;align-items:center;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding-bottom:20px}.brand{display:flex;align-items:center;gap:12px;font-size:12px;letter-spacing:.12em;font-weight:800;text-transform:uppercase}.brand-mark{width:34px;height:34px;border-radius:11px;background:var(--ink);display:grid;place-items:center;color:var(--green);font-size:19px}.stamp{color:var(--muted);font-size:12px}.hero{display:flex;justify-content:space-between;align-items:end;gap:28px;padding:38px 0 27px}.eyebrow{color:var(--green-dark);font-size:11px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}.hero h1{font-size:clamp(32px,4vw,52px);line-height:1.03;margin:8px 0 10px;letter-spacing:0}.hero p{margin:0;color:var(--muted);max-width:650px}.badge{white-space:nowrap;background:#e3eed8;color:#416c36;border:1px solid #ccdfbd;padding:9px 13px;border-radius:999px;font-size:12px;font-weight:800}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:16px}.stat,.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow)}.stat{padding:17px 19px}.stat-label{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:800}.stat-value{font-size:27px;font-weight:750;margin-top:5px;line-height:1.1}.stat-note{font-size:12px;color:var(--muted);margin-top:6px}.layout{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(300px,.85fr);gap:16px}.panel{padding:22px}.panel-head{display:flex;align-items:start;justify-content:space-between;gap:16px;margin-bottom:12px}.panel h2{margin:0;font-size:17px}.panel-sub{margin:4px 0 0;color:var(--muted);font-size:12px}.control{border:1px solid var(--line);background:#fff;border-radius:7px;color:var(--ink);padding:8px 10px;min-width:155px}.chart-wrap{height:280px;position:relative}.chart-wrap canvas{width:100%;height:100%;display:block}.legend{display:flex;gap:18px;color:var(--muted);font-size:12px;margin-top:6px}.legend span{display:flex;align-items:center;gap:7px}.swatch{width:9px;height:9px;border-radius:50%;background:var(--cyan)}.swatch.base{background:var(--coral)}.run-list{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px;margin:16px 0}.run-button{border:1px solid var(--line);background:#f8faf9;color:var(--muted);border-radius:7px;padding:9px 4px;cursor:pointer;font-weight:750;font-size:12px}.run-button:hover,.run-button.active{border-color:var(--cyan);background:#e8f3f2;color:#1e666e}.run-button span{display:block;font-size:10px;font-weight:500;margin-top:2px}.detail-head{display:flex;align-items:center;justify-content:space-between;margin:2px 0 14px}.detail-title{font-size:20px;font-weight:800}.detail-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;max-height:320px;overflow:auto;padding-right:3px}.metric{background:#f6f8f7;border:1px solid #e8eeeb;border-radius:7px;padding:10px 11px}.metric-label{font-size:10px;color:var(--muted);font-weight:750;text-transform:uppercase;letter-spacing:.06em}.metric-value{font-size:16px;font-weight:750;margin-top:3px;font-variant-numeric:tabular-nums}.detail-link{display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--line);margin-top:16px;padding-top:14px;font-size:13px}.drift-panel{margin-top:16px}.table-wrap{overflow:auto;margin-top:14px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums;font-size:12px}th{text-align:right;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-size:10px;padding:10px;border-bottom:1px solid var(--line)}td{text-align:right;padding:11px 10px;border-bottom:1px solid #edf1ef}th:first-child,td:first-child{text-align:left}.status{display:inline-flex;align-items:center;gap:6px;font-weight:750}.status:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--green-dark)}.status.no:before{background:#b9c4bf}.foot{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:22px}.empty{color:var(--muted);font-size:13px;padding:20px 0}@media(max-width:900px){.layout{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.hero{align-items:start;flex-direction:column}.badge{white-space:normal}}@media(max-width:520px){.shell{padding:20px 14px 40px}.stats{gap:8px}.stat{padding:13px}.stat-value{font-size:22px}.panel{padding:15px}.panel-head{flex-direction:column}.control{width:100%}.chart-wrap{height:230px}.detail-metrics{grid-template-columns:1fr}.run-list{gap:5px}.run-button{font-size:11px}}
</style>
</head>
<body>
<div class="shell">
  <header class="topline"><div class="brand"><span class="brand-mark">M</span><span>Model observability</span></div><div class="stamp" id="generated"></div></header>
  <section class="hero"><div><div class="eyebrow">Regression monitoring / synthetic stress test</div><h1>Sales forecast health</h1><p>Ten progressively stronger feature-shift scenarios, evaluated against the same held-out observations. Explore drift signals, prediction stability, and regression quality run by run.</p></div><div class="badge">10 monitored scenarios</div></section>
  <section class="stats" id="stats"></section>
  <main class="layout">
	<section>
	  <div class="panel"><div class="panel-head"><div><h2>Regression quality across drift</h2><p class="panel-sub">Choose a metric to track as simulated drift increases.</p></div><select class="control" id="metric-select" aria-label="Regression metric"></select></div><div class="chart-wrap"><canvas id="metric-chart" aria-label="Regression metric by run"></canvas></div><div class="legend"><span><i class="swatch"></i>Drifted run</span><span><i class="swatch base"></i>Reference baseline</span></div><div class="run-list" id="run-list"></div></div>
	  <div class="panel drift-panel"><div class="panel-head"><div><h2>Drift footprint</h2><p class="panel-sub">Evidently-detected drifted columns and prediction-distribution p-value.</p></div></div><div class="chart-wrap"><canvas id="drift-chart" aria-label="Drift footprint by run"></canvas></div></div>
	  <div class="panel drift-panel"><div class="panel-head"><div><h2>All run results</h2><p class="panel-sub">Full metric matrix is also available in monitoring_runs.csv.</p></div></div><div class="table-wrap"><table><thead><tr><th>Run</th><th>Severity</th><th>Drifted</th><th>Pred. p-value</th><th>MAE</th><th>RMSE</th><th>R²</th><th>MAPE</th></tr></thead><tbody id="results-body"></tbody></table></div></div>
	</section>
	<aside class="panel"><div class="detail-head"><div><div class="eyebrow">Selected scenario</div><div class="detail-title" id="run-title"></div></div></div><div class="detail-metrics" id="metric-grid"></div><div class="detail-link"><span>Evidently deep-dive</span><a id="report-link" target="_blank" rel="noopener">Open report ↗</a></div></aside>
  </main>
  <footer class="foot"><span>Simulated drift is not production telemetry.</span><span>Generated locally · Evidently AI</span></footer>
</div>
<script>
const DATA=__MONITORING_DATA__;
const metricSelect=document.getElementById('metric-select');
const runList=document.getElementById('run-list');
let selectedRun=0;
const fmt=(value,digits=3)=>value===null||!Number.isFinite(Number(value))?'—':Number(value).toLocaleString(undefined,{maximumFractionDigits:digits});
const metricNames=DATA.metric_names;
document.getElementById('generated').textContent=`Generated ${DATA.generated_at} · ${DATA.reference_rows} validation rows · ${DATA.feature_count} input features`;
metricNames.forEach(name=>{const option=document.createElement('option');option.value=name;option.textContent=name;metricSelect.append(option)});
const maxDrift=Math.max(...DATA.runs.map(run=>run.drift_share));
document.getElementById('stats').innerHTML=[
  ['Reference MAE',fmt(DATA.baseline_metrics['MAE']),'held-out baseline'],
  ['Reference RMSE',fmt(DATA.baseline_metrics['RMSE']),'held-out baseline'],
  ['Peak drift share',`${fmt(maxDrift*100,1)}%`,'highest Evidently drift share'],
  ['Prediction drift',`${DATA.runs.filter(run=>run.prediction_drift_detected).length} / ${DATA.runs.length}`,'runs with p-value below 0.05']
].map(([label,value,note])=>`<article class="stat"><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-note">${note}</div></article>`).join('');
DATA.runs.forEach((run,index)=>{const button=document.createElement('button');button.className='run-button';button.innerHTML=`Run ${String(run.run).padStart(2,'0')}<span>${Math.round(run.severity*100)}% shift</span>`;button.addEventListener('click',()=>selectRun(index));runList.append(button)});
document.getElementById('results-body').innerHTML=DATA.runs.map(run=>`<tr><td>Run ${String(run.run).padStart(2,'0')}</td><td>${fmt(run.severity*100,0)}%</td><td>${run.drifted_columns} / ${DATA.feature_count+1}</td><td>${fmt(run.prediction_drift_p_value,4)}</td><td>${fmt(run.metrics['MAE'])}</td><td>${fmt(run.metrics['RMSE'])}</td><td>${fmt(run.metrics['R2'])}</td><td>${fmt(run.metrics['MAPE (%)'],2)}%</td></tr>`).join('');
function selectRun(index){selectedRun=index;const run=DATA.runs[index];document.querySelectorAll('.run-button').forEach((button,i)=>button.classList.toggle('active',i===index));document.getElementById('run-title').textContent=`Run ${String(run.run).padStart(2,'0')}`;document.getElementById('metric-grid').innerHTML=metricNames.map(name=>`<div class="metric"><div class="metric-label">${name}</div><div class="metric-value">${fmt(run.metrics[name])}</div></div>`).join('');const link=document.getElementById('report-link');link.href=run.report;drawCharts()}
function canvasContext(canvas){const rect=canvas.getBoundingClientRect();const ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));const context=canvas.getContext('2d');context.scale(ratio,ratio);return{context,width:rect.width,height:rect.height}}
function drawLine(canvasId,series,baseline,label,color){const canvas=document.getElementById(canvasId);const{context:ctx,width,height}=canvasContext(canvas);const pad={l:48,r:18,t:18,b:34};const values=series.filter(Number.isFinite);const low=Math.min(...values,baseline);const high=Math.max(...values,baseline);const range=(high-low)||1;const yMin=low-range*.12;const yMax=high+range*.12;const px=i=>pad.l+i*(width-pad.l-pad.r)/Math.max(series.length-1,1);const py=v=>height-pad.b-(v-yMin)/(yMax-yMin)*(height-pad.t-pad.b);ctx.clearRect(0,0,width,height);ctx.font='11px Segoe UI';ctx.fillStyle='#74847f';ctx.strokeStyle='#e6ece9';ctx.lineWidth=1;for(let tick=0;tick<5;tick++){const y=pad.t+tick*(height-pad.t-pad.b)/4;const value=yMax-tick*(yMax-yMin)/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(width-pad.r,y);ctx.stroke();ctx.textAlign='right';ctx.fillText(fmt(value,2),pad.l-8,y+4)}ctx.strokeStyle='#d8755d';ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(pad.l,py(baseline));ctx.lineTo(width-pad.r,py(baseline));ctx.stroke();ctx.setLineDash([]);ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.beginPath();series.forEach((value,index)=>index===0?ctx.moveTo(px(index),py(value)):ctx.lineTo(px(index),py(value)));ctx.stroke();series.forEach((value,index)=>{ctx.fillStyle=index===selectedRun?'#d8755d':color;ctx.beginPath();ctx.arc(px(index),py(value),index===selectedRun?5:3.5,0,Math.PI*2);ctx.fill()});ctx.fillStyle='#74847f';ctx.textAlign='center';series.forEach((_,index)=>ctx.fillText(String(index+1),px(index),height-11));canvas.title=`${label}: select a run below for exact values`}
function drawDrift(){const canvas=document.getElementById('drift-chart');const{context:ctx,width,height}=canvasContext(canvas);const pad={l:45,r:18,t:20,b:34};ctx.clearRect(0,0,width,height);ctx.font='11px Segoe UI';ctx.fillStyle='#74847f';ctx.strokeStyle='#e6ece9';for(let tick=0;tick<=4;tick++){const y=pad.t+tick*(height-pad.t-pad.b)/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(width-pad.r,y);ctx.stroke();ctx.textAlign='right';ctx.fillText(`${100-tick*25}%`,pad.l-8,y+4)}const slot=(width-pad.l-pad.r)/DATA.runs.length;DATA.runs.forEach((run,index)=>{const x=pad.l+index*slot+slot*.2;const barW=slot*.58;const barH=Math.max(2,run.drift_share*(height-pad.t-pad.b));ctx.fillStyle=index===selectedRun?'#d8755d':'#2c8490';ctx.fillRect(x,height-pad.b-barH,barW,barH);ctx.fillStyle='#74847f';ctx.textAlign='center';ctx.fillText(String(index+1),x+barW/2,height-11)})}
function drawCharts(){const field=metricSelect.value;drawLine('metric-chart',DATA.runs.map(run=>Number(run.metrics[field])).map(value=>Number.isFinite(value)?value:0),Number(DATA.baseline_metrics[field]),field,'#2c8490');drawDrift()}
metricSelect.addEventListener('change',drawCharts);window.addEventListener('resize',drawCharts);selectRun(0);
</script>
</body>
</html>""".replace("__MONITORING_DATA__", payload)
