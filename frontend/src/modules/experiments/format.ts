import type { MetricKind } from "./api";

/* Numbers of an experiment, as the engine writes its sentences (experiments/analyze.py): rates in
   %, differences of rates in points, others with three significant digits at least. */

const MINUS = "−";

export function number(value: number): string {
  const magnitude = Math.abs(value);
  const digits = magnitude >= 100 ? 0 : magnitude >= 10 ? 1 : magnitude >= 1 || magnitude === 0 ? 2 : -1;
  const text =
    digits >= 0
      ? value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })
      : value.toLocaleString("en-US", { maximumSignificantDigits: 2 });
  return text.replace("-", MINUS);
}

const sign = (value: number, text: string) => (value >= 0 && !text.startsWith(MINUS) ? `+${text}` : text);

export function value(kind: MetricKind, v: number | null): string {
  if (v === null) return "—";
  return kind === "rate" ? `${(v * 100).toFixed(1)} %` : number(v);
}

/** A difference: points for a rate, the metric's own unit otherwise; always signed. */
export function gap(kind: MetricKind, v: number | null): string {
  if (v === null) return "—";
  return kind === "rate" ? `${sign(v, (v * 100).toFixed(1).replace("-", MINUS))} pts` : sign(v, number(v));
}

export function percent(v: number | null, signed = true): string {
  if (v === null) return "—";
  const text = `${(v * 100).toFixed(1).replace("-", MINUS)} %`;
  return signed ? sign(v, text) : text;
}

export function pValue(p: number | null): string {
  if (p === null) return "—";
  if (p < 0.001) return "< 0.001";
  return p < 0.01 ? p.toFixed(3) : p.toFixed(2);
}

export const interval = (low: string, high: string) => `[${low} ; ${high}]`;

/** P(best) as the engine writes it (experiments/bandit.py): 94 % · 99.7 % · > 99.9 % · < 0.1 %. */
export function probability(p: number): string {
  if (p >= 0.9995) return "> 99.9 %";
  if (p < 0.0005) return "< 0.1 %";
  const shown = p * 100;
  return shown >= 1 && shown <= 99 ? `${shown.toFixed(0)} %` : `${shown.toFixed(1)} %`;
}

/** An expected loss: points for a rate, the metric's unit otherwise; tiny losses as "< 0.01". */
export function loss(kind: string, v: number): string {
  const shown = kind === "rate" ? v * 100 : v;
  const unit = kind === "rate" ? " pt" : "";
  if (shown < 0.005) return `< 0.01${unit}`;
  return `${number(shown)}${unit}`;
}
