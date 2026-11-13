import { createContext, useContext } from "react";

import type { Step, Suggestion } from "./api";

/** Write a suggestion's step as given (null: out of the draft): N38's review of a merge. */
export type Replace = (suggestion: Suggestion, step: Step | null) => void;

/** What a suggestion's line needs beyond its own tick, given once by the page for every layer
 *  (Basic, Advanced, More checks): the draft (N38 reads a merge's ticks from its step), every
 *  suggestion found (a step goes at its place, C-06) and the dataset (N41 previews its rows).
 *  `onHide` (N05): the suggestions a dismissal just hid, whose steps leave the draft. */
export type ReviewContextValue = { datasetId: string; steps: Step[]; found: Suggestion[]; onReplace: Replace; onHide?: (ids: string[]) => void };

export const ReviewContext = createContext<ReviewContextValue | null>(null);

export const useReview = () => useContext(ReviewContext);
