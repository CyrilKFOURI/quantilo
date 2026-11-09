/** The CYFOURI brand monogram (CK): two nested arcs whose inner stroke runs into a K, with the
 *  name under it. Stays on the left of the app's own mark (043). Drawn in currentColor. */
export function CyfouriLogo({ size = 20, className = "" }: { size?: number; className?: string }) {
  return (
    <span className={`flex shrink-0 flex-col items-center gap-0.5 ${className}`}>
      <svg
        viewBox="0 0 92 70"
        width={(size * 92) / 70}
        height={size}
        fill="none"
        stroke="currentColor"
        strokeWidth={7}
        aria-hidden
      >
        <path d="M50.84 9.04 A30 30 0 1 0 52.97 57.84" />
        <path d="M43.25 19.04 A17.5 17.5 0 1 0 48.29 44.98 L87 4.5" />
        <path d="M58 35.5 L86 66.5" />
      </svg>
      <span className="font-[Georgia,serif] text-[7px] leading-none tracking-[0.18em]">CYFOURI</span>
    </span>
  );
}
