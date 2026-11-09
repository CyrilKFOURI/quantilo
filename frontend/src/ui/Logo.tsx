/** The Quantilo mark (043): a Q drawn as a measuring dial, ring open at the top right, four quartile ticks,
 *  the tail of the Q, and a needle in the accent colour rising through the opening. Ink in currentColor.
 *  Under 24 px the ticks would blur: the small mark keeps the ring, the tail and the needle, thicker. */
export function Logo({ size = 20, className = "" }: { size?: number; className?: string }) {
  const small = size < 24;
  const stroke = small ? 7.5 : 5.5;
  return (
    <svg
      viewBox="2 2 58 58"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      aria-hidden
      className={className}
    >
      <path d="M47.54 23.51A20 20 0 1 1 36.49 12.46" strokeWidth={stroke} />
      {!small && <path d="M29 17v4M15 31h4M29 45v-4M43 31h-4" strokeWidth={3.5} />}
      <path d="M37.49 39.49 52.33 54.33" strokeWidth={stroke} />
      <path d="M29 31 53.04 6.96" strokeWidth={small ? 7 : 4.75} className="stroke-accent" />
    </svg>
  );
}
