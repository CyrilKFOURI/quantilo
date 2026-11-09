/** True in the browser-only build (`npm run build:lite`): Python runs in the page, no server. */
export const LITE = import.meta.env.VITE_LITE === "1";

/** Where the full app's one-command install is explained. */
export const FULL_APP_URL = "https://github.com/CyrilKFOURI/quantilo#run";
