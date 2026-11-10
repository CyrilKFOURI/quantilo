/** The Clean module (065) is on unless the build says `VITE_CLEAN_ENABLED=0` (the server has its
 *  own switch, QUANTILO_CLEAN_ENABLED: its routes then answer 404 and the app shows no versions). */
export const CLEAN_ENABLED = import.meta.env.VITE_CLEAN_ENABLED !== "0";
