import { useEffect, useState } from "react";

/**
 * A UI preference stored in localStorage and mirrored as a `data-*` attribute on <html>,
 * so CSS tokens can react to it. The default value removes the attribute.
 */
export type Preference<T extends string> = {
  storageKey: string;
  attribute: string;
  values: readonly T[];
  defaultValue: T;
};

export const themePreference: Preference<"system" | "light" | "dark"> = {
  storageKey: "de.theme",
  attribute: "theme",
  values: ["system", "light", "dark"],
  defaultValue: "system",
};

export const densityPreference: Preference<"compact" | "comfortable"> = {
  storageKey: "de.density",
  attribute: "density",
  values: ["compact", "comfortable"],
  defaultValue: "compact",
};

/** "reduce" cuts animations; "system" follows the OS setting; "full" keeps them even if the OS asks. */
export const motionPreference: Preference<"system" | "reduce" | "full"> = {
  storageKey: "de.motion",
  attribute: "motion",
  values: ["system", "reduce", "full"],
  defaultValue: "system",
};

/** Interface language (127). "auto" follows the browser; a change reloads the page (texts are read once). */
export const languagePreference: Preference<"auto" | "en" | "fr"> = {
  storageKey: "de.lang",
  attribute: "lang",
  values: ["auto", "en", "fr"],
  defaultValue: "auto",
};

export function readPreference<T extends string>(preference: Preference<T>): T {
  try {
    const stored = localStorage.getItem(preference.storageKey);
    return preference.values.find((value) => value === stored) ?? preference.defaultValue;
  } catch {
    return preference.defaultValue;
  }
}

export function applyPreference<T extends string>(
  preference: Preference<T>,
  value: T,
  root: HTMLElement = document.documentElement,
): void {
  if (value === preference.defaultValue) delete root.dataset[preference.attribute];
  else root.dataset[preference.attribute] = value;
}

export function savePreference<T extends string>(preference: Preference<T>, value: T): void {
  try {
    localStorage.setItem(preference.storageKey, value);
  } catch {
    // Storage can be unavailable (private mode); the choice still applies to this session.
  }
  applyPreference(preference, value);
  // Other controls showing the same preference (sidebar switcher, Settings, palette) follow.
  window.dispatchEvent(new CustomEvent(PREFERENCE_EVENT, { detail: preference.storageKey }));
}

const PREFERENCE_EVENT = "de:preference";

export function usePreference<T extends string>(preference: Preference<T>): [T, (value: T) => void] {
  const [value, setValue] = useState(() => readPreference(preference));
  useEffect(() => {
    const onChange = (event: Event) => {
      if ((event as CustomEvent<string>).detail === preference.storageKey) setValue(readPreference(preference));
    };
    window.addEventListener(PREFERENCE_EVENT, onChange);
    return () => window.removeEventListener(PREFERENCE_EVENT, onChange);
  }, [preference]);
  const update = (next: T) => {
    savePreference(preference, next);
    setValue(next);
  };
  return [value, update];
}
