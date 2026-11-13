import { CircleAlert, FlaskConical, RotateCcw, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";

import { Folded } from "../../features/text/Folded";
import { imageUrl } from "../../features/import/imageSetApi";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { CompareStrip, type StripPanel } from "./CompareStrip";
import type { WhyImage } from "./explainApi";
import type { ImageRow } from "./model";
import {
  conceptProbabilities,
  useImageConcepts,
  useLooksLike,
  useProtoOverview,
  useTcav,
  useTcavOptions,
  type ProtoOverview,
  type WhyConcepts,
  type WhyTcav,
} from "./moreApi";

/**
 * 118: the evidence of the Advanced › More models, and TCAV.
 * - `LooksLike` ("Why?" of ProtoPNet-lite): this patch looks like that prototype, the prototype's
 *   own training image beside it; the numbers are the score itself (R-VPP-06).
 * - `ConceptsBlock` ("Why?" of the concept bottleneck): the concepts read in the image; change one
 *   and the class probabilities are recomputed here from the kept head (R-VCB-05).
 * - `ExplainMore` (Explain › More): ProtoPNet's prototypes by class, and TCAV on demand (R-VTC).
 */

const two = (value: number) => value.toFixed(2);

export function LooksLike({ datasetId, runId, row, nameOf, openRef }: { datasetId: string; runId: string; row: ImageRow; nameOf: (ref: WhyImage) => string; openRef: (ref: WhyImage) => (() => void) | undefined }) {
  const looks = useLooksLike(datasetId, runId, row.id, true);
  if (looks.isPending) return <Loading compact label={t("vision.more.loadingLooks")} className="h-10 rounded-md" />;
  if (looks.isError) return <ErrorNote>{(looks.error as Error).message}</ErrorNote>;
  const data = looks.data;
  const panels: StripPanel[] = [
    {
      key: "image",
      label: t("vision.why.thisImage"),
      src: imageUrl(datasetId, row.id, "file"),
      alt: row.name,
      boxes: data.looks.map((look, i) => ({ box: look.box, label: String(i + 1) })),
    },
    ...data.looks.map((look, i) => ({
      key: `proto-${look.prototype}`,
      label: t("vision.more.thatPrototype", { n: String(i + 1), label: look.label, name: nameOf(look.source) }),
      src: imageUrl(datasetId, look.source.image_id, "file"),
      alt: nameOf(look.source),
      boxes: [{ box: look.source_box, label: String(i + 1) }],
      caption: <span className="tabular-nums">{t("vision.more.points", { similarity: two(look.similarity), weight: two(look.weight), points: two(look.points) })}</span>,
      onOpen: openRef(look.source),
    })),
  ];
  return (
    <section className="min-w-0 space-y-3" aria-labelledby="why-looks">
      <div className="space-y-1">
        <h3 id="why-looks" className="text-sm font-semibold text-ink-1">
          {t("vision.more.looksTitle")}
        </h3>
        <p className="max-w-[75ch] text-ink-1">{data.sentence}</p>
      </div>
      {data.looks.length > 0 && (
        <ol className="grid max-w-3xl gap-3 sm:grid-cols-3" aria-label={t("vision.more.pairs")}>
          {data.looks.map((look, i) => (
            <li key={look.prototype} className="min-w-0 space-y-1">
              <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-1.5">
                <PatchCrop src={imageUrl(datasetId, row.id, "file")} box={look.box} alt={t("vision.more.thisPatch", { n: String(i + 1) })} />
                <span aria-hidden className="text-xs text-ink-3">
                  ≈
                </span>
                <PatchCrop src={imageUrl(datasetId, look.source.image_id, "file")} box={look.source_box} alt={t("vision.more.protoAlt", { label: look.label, name: nameOf(look.source) })} />
              </div>
              <p className="text-xs text-ink-2 tabular-nums">
                <span className="font-medium text-ink-1">{i + 1}.</span> {t("vision.more.pairCaption", { label: look.label, points: two(look.points) })}
              </p>
            </li>
          ))}
        </ol>
      )}
      {data.looks.length > 0 && <CompareStrip panels={panels} label={t("vision.more.looksTitle")} />}
      <p className="flex max-w-[75ch] gap-1.5 text-xs text-ink-2">
        <ShieldCheck size={13} strokeWidth={1.5} className="mt-px shrink-0 text-ok" aria-hidden />
        {t("vision.more.looksOwn")}
      </p>
      {data.notes.map((note) => (
        <p key={note} className="text-xs text-ink-3">
          {note}
        </p>
      ))}
    </section>
  );
}

export function ConceptsBlock({ datasetId, runId, row }: { datasetId: string; runId: string; row: ImageRow }) {
  const found = useImageConcepts(datasetId, runId, row.id, true);
  if (found.isPending) return <Loading compact label={t("vision.more.loadingConcepts")} className="h-10 rounded-md" />;
  if (found.isError) return <ErrorNote>{(found.error as Error).message}</ErrorNote>;
  return <Concepts key={row.id} data={found.data} />;
}

function Concepts({ data }: { data: WhyConcepts }) {
  const read = useMemo(() => data.concepts.map((c) => c.predicted), [data]);
  const [values, setValues] = useState<number[]>(read);
  const before = useMemo(() => conceptProbabilities(data.head, read), [data.head, read]);
  const after = conceptProbabilities(data.head, values);
  const changed = values.some((value, j) => value !== read[j]);
  const order = before.map((_, k) => k).sort((a, b) => (after[b] ?? 0) - (after[a] ?? 0));
  const set = (j: number, value: number) => setValues((current) => current.map((v, i) => (i === j ? value : v)));
  return (
    <section className="min-w-0 space-y-3" aria-labelledby="why-concepts">
      <div className="space-y-1">
        <h3 id="why-concepts" className="text-sm font-semibold text-ink-1">
          {t("vision.more.conceptsTitle")}
        </h3>
        <p className="max-w-[75ch] text-ink-1">{data.sentence}</p>
        <p className="max-w-[75ch] text-xs text-ink-3">{t("vision.why.space", { space: data.space })}</p>
      </div>
      <div className="max-w-full overflow-x-auto">
        <table className="w-full max-w-3xl text-xs tabular-nums">
          <caption className="sr-only">{t("vision.more.conceptsTitle")}</caption>
          <thead>
            <tr className="text-left text-ink-3">
              <th scope="col" className="py-1 pr-3 font-normal">{t("vision.more.concept")}</th>
              <th scope="col" className="py-1 pr-3 font-normal">{t("vision.more.read")}</th>
              <th scope="col" className="py-1 pr-3 font-normal">{t("vision.more.truth")}</th>
              <th scope="col" className="py-1 font-normal">{t("vision.more.setTo")}</th>
            </tr>
          </thead>
          <tbody>
            {data.concepts.map((concept, j) => (
              <tr key={concept.key} className="border-t border-line align-middle">
                <th scope="row" className="py-2 pr-3 text-left font-normal text-ink-1">
                  {concept.name}
                </th>
                <td className="py-2 pr-3 text-ink-1">{concept.kind === "binary" ? two(concept.predicted) : concept.predicted.toPrecision(3)}</td>
                <td className="py-2 pr-3 text-ink-2">{concept.truth === null || concept.truth === undefined ? "—" : concept.kind === "binary" ? (concept.truth >= 0.5 ? t("vision.yes") : t("vision.no")) : concept.truth.toPrecision(3)}</td>
                <td className="py-1.5">
                  {concept.kind === "binary" ? (
                    <SegmentedControl
                      label={t("vision.more.setConcept", { name: concept.name })}
                      value={values[j] === read[j] ? "read" : (values[j] ?? 0) >= 0.5 ? "yes" : "no"}
                      onChange={(value) => set(j, value === "read" ? (read[j] ?? 0) : value === "yes" ? 1 : 0)}
                      options={[
                        { value: "read", label: t("vision.more.asRead") },
                        { value: "yes", label: t("vision.more.has") },
                        { value: "no", label: t("vision.more.hasNot") },
                      ]}
                    />
                  ) : (
                    <TextInput
                      type="number"
                      aria-label={t("vision.more.setConcept", { name: concept.name })}
                      className="w-28"
                      value={String(Number((values[j] ?? 0).toPrecision(4)))}
                      onChange={(event) => {
                        const next = Number(event.target.value);
                        if (Number.isFinite(next)) set(j, next);
                      }}
                    />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="space-y-1.5" aria-live="polite">
        <p className="text-xs text-ink-2">{changed ? t("vision.more.afterChange") : t("vision.more.asReadNow")}</p>
        <ul className="grid max-w-md grid-cols-[minmax(0,auto)_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 text-xs" aria-label={t("vision.why.probabilities")}>
          {order.map((k, index) => (
            <li key={data.head.classes[k]} className="contents">
              <span className="truncate font-mono text-ink-1">{data.head.classes[k]}</span>
              <span className="h-1.5 min-w-0" aria-hidden>
                <span className={`block h-full rounded-r-[2px] ${index === 0 ? "bg-(--chart-1)" : "bg-ink-3/60"}`} style={{ width: `${Math.max(1, (after[k] ?? 0) * 100)}%` }} />
              </span>
              <span className="text-right text-ink-1">
                {two(after[k] ?? 0)}
                {changed && <span className="ml-1.5 text-ink-3">{t("vision.more.was", { value: two(before[k] ?? 0) })}</span>}
              </span>
            </li>
          ))}
        </ul>
        {changed && (
          <Button variant="ghost" icon={RotateCcw} onClick={() => setValues(read)}>
            {t("vision.more.resetConcepts")}
          </Button>
        )}
      </div>
      {data.notes.map((note) => (
        <p key={note} className="text-xs text-ink-3">
          {note}
        </p>
      ))}
    </section>
  );
}

// --- Explain › More --------------------------------------------------------------------------

export function ExplainMore({ datasetId, runId, byId, onWhy }: { datasetId: string; runId: string; byId: ReadonlyMap<string, ImageRow>; onWhy: (row: ImageRow) => void }) {
  return (
    <Folded nested title={t("vision.more.explainTitle")} hint={t("vision.more.explainHint")}>
      <div className="min-w-0 space-y-8">
        <Prototypes datasetId={datasetId} runId={runId} byId={byId} onWhy={onWhy} />
        <Tcav datasetId={datasetId} runId={runId} />
      </div>
    </Folded>
  );
}

function Prototypes({ datasetId, runId, byId, onWhy }: { datasetId: string; runId: string; byId: ReadonlyMap<string, ImageRow>; onWhy: (row: ImageRow) => void }) {
  const found = useProtoOverview(datasetId, runId);
  if (found.isPending) return <Loading compact label={t("vision.more.loadingPrototypes")} className="h-10 rounded-md" />;
  if (found.isError) return <ErrorNote>{(found.error as Error).message}</ErrorNote>;
  return (
    <section className="min-w-0 space-y-3" aria-labelledby="explain-protopnet">
      <h3 id="explain-protopnet" className="text-sm font-semibold text-ink-1">
        {t("vision.more.protoTitle")}
      </h3>
      {found.data ? <PrototypeGrid datasetId={datasetId} data={found.data} byId={byId} onWhy={onWhy} /> : <p className="max-w-[75ch] text-ink-2">{t("vision.more.noProtopnet")}</p>}
    </section>
  );
}

function PrototypeGrid({ datasetId, data, byId, onWhy }: { datasetId: string; data: ProtoOverview; byId: ReadonlyMap<string, ImageRow>; onWhy: (row: ImageRow) => void }) {
  return (
    <div className="min-w-0 space-y-3">
      <p className="max-w-[75ch] text-ink-1">{data.sentence}</p>
      <div className="divide-y divide-line rounded-lg border border-line bg-surface-1">
        {data.classes.map((group) => (
          <div key={group.label} className="grid min-w-0 gap-3 px-4 py-3 lg:grid-cols-[160px_minmax(0,1fr)]">
            <p className="truncate font-mono font-semibold text-ink-1">{group.label}</p>
            <ul className="flex min-w-0 flex-wrap gap-2">
              {group.prototypes.map((proto) => {
                const row = byId.get(proto.image.image_id);
                return (
                  <li key={proto.prototype} className="w-[76px] shrink-0">
                    <button
                      type="button"
                      disabled={!row}
                      onClick={() => row && onWhy(row)}
                      title={row ? t("vision.explain.openWhy", { name: row.name }) : proto.image.image_id}
                      className="group/proto block w-full text-left"
                    >
                      <PatchCrop src={imageUrl(datasetId, proto.image.image_id, "file")} box={proto.box} alt={t("vision.more.protoAlt", { label: proto.label, name: row?.name ?? proto.image.image_id })} />
                      <span className="mt-1 block truncate text-[11px] text-ink-3 tabular-nums">
                        {t("vision.more.weight", { value: two(proto.weight) })}
                        {proto.duplicate_of !== null && proto.duplicate_of !== undefined ? ` · ${t("vision.more.duplicate")}` : ""}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>
      {data.notes.map((note) => (
        <p key={note} className="text-xs text-ink-3">
          {note}
        </p>
      ))}
    </div>
  );
}

/** A patch with some context: the region three times the cell's size around it, the cell
 *  outlined (cells are square in pixels, so the region's height follows from its width). */
export function PatchCrop({ src, box, alt, context = 3 }: { src: string; box: readonly number[]; alt: string; context?: number }) {
  const [x0 = 0, y0 = 0, x1 = 1, y1 = 1] = box;
  const w = Math.max(1e-6, x1 - x0) * context;
  const h = Math.max(1e-6, y1 - y0) * context;
  const left = Math.min(Math.max(0, (x0 + x1) / 2 - w / 2), Math.max(0, 1 - w));
  const top = Math.min(Math.max(0, (y0 + y1) / 2 - h / 2), Math.max(0, 1 - h));
  return (
    <span className="relative block aspect-square w-full overflow-hidden rounded-md bg-surface-2">
      <img
        src={src}
        alt={alt}
        loading="lazy"
        decoding="async"
        className="absolute max-w-none"
        style={{ width: `${100 / w}%`, height: `${100 / h}%`, left: `${(-left / w) * 100}%`, top: `${(-top / h) * 100}%` }}
      />
      <span
        aria-hidden
        className="pointer-events-none absolute rounded-[2px] ring-2 ring-accent"
        style={{ left: `${((x0 - left) / w) * 100}%`, top: `${((y0 - top) / h) * 100}%`, width: `${((x1 - x0) / w) * 100}%`, height: `${((y1 - y0) / h) * 100}%` }}
      />
      <span aria-hidden className="pointer-events-none absolute inset-0 rounded-md ring-1 ring-ink-1/8 ring-inset group-hover/proto:ring-2 group-hover/proto:ring-accent" />
    </span>
  );
}

function Tcav({ datasetId, runId }: { datasetId: string; runId: string }) {
  const options = useTcavOptions(datasetId, runId);
  const [concept, setConcept] = useState<string>("");
  const [layer, setLayer] = useState<string>("");
  const [asked, setAsked] = useState<{ concept: string; layer: string | null } | null>(null);
  const result = useTcav(datasetId, runId, asked);
  if (options.isPending) return <Loading compact label={t("vision.more.loadingTcav")} className="h-10 rounded-md" />;
  if (options.isError) return <ErrorNote>{(options.error as Error).message}</ErrorNote>;
  const data = options.data;
  const chosen = concept || data.concepts[0]?.key || "";
  return (
    <section className="min-w-0 space-y-3" aria-labelledby="explain-tcav">
      <div className="space-y-1">
        <h3 id="explain-tcav" className="text-sm font-semibold text-ink-1">
          {t("vision.more.tcavTitle")}
        </h3>
        <p className="max-w-[75ch] text-ink-2">{t("vision.more.tcavIntro")}</p>
      </div>
      {!data.available ? (
        <p className="flex max-w-[75ch] gap-1.5 text-ink-2">
          <CircleAlert size={14} strokeWidth={1.5} className="mt-0.5 shrink-0 text-ink-3" aria-hidden />
          {data.reason}
        </p>
      ) : (
        <>
          <div className="grid max-w-3xl gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,10rem)_auto] sm:items-end">
            <label className="flex min-w-0 flex-col gap-1">
              <span className="text-xs text-ink-2">{t("vision.more.tcavConcept")}</span>
              <Select value={chosen} onChange={(event) => setConcept(event.target.value)} className="max-w-full">
                {data.concepts.map((c) => (
                  <option key={c.key} value={c.key}>
                    {t(c.source === "label" ? "vision.more.optionLabel" : c.source === "group" ? "vision.more.optionGroup" : "vision.more.optionColumn", { name: c.name, count: formatCount(c.images) })}
                  </option>
                ))}
              </Select>
            </label>
            <label className="flex min-w-0 flex-col gap-1">
              <span className="text-xs text-ink-2">{t("vision.more.tcavLayer")}</span>
              <Select value={layer || data.layer || ""} onChange={(event) => setLayer(event.target.value)}>
                {data.layers.map((name) => (
                  <option key={name} value={name}>
                    {name === data.layer ? t("vision.more.layerDefault", { name }) : name}
                  </option>
                ))}
              </Select>
            </label>
            <Button variant="secondary" icon={FlaskConical} onClick={() => setAsked({ concept: chosen, layer: layer && layer !== data.layer ? layer : null })} disabled={!chosen || result.isFetching}>
              {t("vision.more.tcavRun")}
            </Button>
          </div>
          <p className="text-xs text-ink-3">{t("vision.more.tcavCost", { cost: data.cost })}</p>
          {asked &&
            (result.isPending ? (
              <Loading label={t("vision.more.tcavRunning")} className="h-24 rounded-md" />
            ) : result.isError ? (
              <ErrorNote>{(result.error as Error).message}</ErrorNote>
            ) : (
              <TcavResult data={result.data} />
            ))}
        </>
      )}
    </section>
  );
}

/** One row per class: the concept's TCAV score (its CAVs' mean and spread) against random
 *  directions; the verdict of the test is written, never only coloured (R-VTC-05). */
function TcavResult({ data }: { data: WhyTcav }) {
  return (
    <div className="min-w-0 space-y-2">
      <p className="max-w-[75ch] text-ink-1">{data.sentence}</p>
      <p className="text-xs text-ink-3 tabular-nums">
        {t("vision.more.tcavFacts", { images: formatCount(data.images), random: formatCount(data.random_images), accuracy: `${Math.round(data.cav_accuracy * 100)} %`, layer: data.layer, seconds: data.seconds.toFixed(0) })}
      </p>
      <div className="max-w-full overflow-x-auto">
        <table className="w-full max-w-3xl text-xs tabular-nums">
          <caption className="sr-only">{t("vision.more.tcavTitle")}</caption>
          <thead>
            <tr className="text-left text-ink-3">
              <th scope="col" className="py-1 pr-3 font-normal">{t("vision.more.class")}</th>
              <th scope="col" className="py-1 pr-3 font-normal sm:w-2/5">
                <span className="max-sm:hidden">{t("vision.more.tcavScore")}</span>
                <span className="sm:hidden">{t("vision.more.tcavScoreShort")}</span>
              </th>
              <th scope="col" className="py-1 pr-3 text-right font-normal">p</th>
              <th scope="col" className="py-1 font-normal">{t("vision.more.verdict")}</th>
            </tr>
          </thead>
          <tbody>
            {data.classes.map((c) => (
              <tr key={c.label} className="border-t border-line">
                <th scope="row" className="py-2 pr-3 text-left font-mono font-normal break-all text-ink-1">
                  {c.label}
                </th>
                <td data-keep-line className="py-2 pr-3">
                  <span className="flex items-center gap-2">
                    <span className="relative h-2 min-w-0 flex-1 rounded-[2px] max-sm:hidden" aria-hidden>
                      <span className="absolute inset-y-0 left-0 rounded-r-[2px] bg-(--chart-1)" style={{ width: `${Math.max(1, c.score * 100)}%` }} />
                      <span className="absolute -inset-y-1 w-0.5 bg-ink-2" style={{ left: `${c.random * 100}%` }} title={t("vision.more.randomMark")} />
                    </span>
                    <span className="shrink-0 text-right whitespace-nowrap text-ink-1 sm:w-28">
                      {Math.round(c.score * 100)} % <span className="text-ink-3 max-sm:hidden">{t("vision.more.vsRandom", { value: `${Math.round(c.random * 100)} %` })}</span>
                      <span className="text-ink-3 sm:hidden">/ {Math.round(c.random * 100)} %</span>
                    </span>
                  </span>
                </td>
                <td data-keep-line className="py-2 pr-3 text-right whitespace-nowrap text-ink-2">{c.p_value === null || c.p_value === undefined ? "—" : c.p_value < 0.001 ? "< 0.001" : c.p_value.toFixed(3)}</td>
                <td data-keep-line className={`py-2 whitespace-nowrap ${c.significant ? "font-medium text-ink-1" : "text-ink-3"}`}>{c.significant ? t("vision.more.found") : t("vision.more.notSignificant")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">{t("vision.more.randomLegend")}</p>
      {data.notes.map((note) => (
        <p key={note} className="text-xs text-ink-3">
          {note}
        </p>
      ))}
    </div>
  );
}
