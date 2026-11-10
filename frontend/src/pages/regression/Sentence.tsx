import { Fragment } from "react";

/** A sentence from the engine: `column` names in mono. */
export function Sentence({ text: sentence }: { text: string }) {
  const parts = sentence.split("`");
  return (
    <>
      {parts.map((part, i) =>
        i % 2 ? (
          <code key={i} className="font-mono text-[0.92em] text-ink-1">
            {part}
          </code>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}
