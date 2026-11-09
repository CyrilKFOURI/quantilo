import type { LucideIcon } from "lucide-react";
import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  icon?: LucideIcon;
  iconOnly?: boolean;
};

const variants: Record<Variant, string> = {
  primary: "bg-accent text-accent-ink hover:brightness-95 font-medium",
  secondary: "border border-line bg-surface-1 text-ink-1 hover:bg-surface-2",
  ghost: "text-ink-2 hover:bg-surface-2 hover:text-ink-1",
  danger: "bg-bad text-surface-1 hover:brightness-95 font-medium",
};

export function Button({
  variant = "secondary",
  icon: Icon,
  iconOnly = false,
  className = "",
  children,
  type = "button",
  ...props
}: ButtonProps) {
  const size = iconOnly ? "size-(--control-h) justify-center" : "h-(--control-h) px-3";
  return (
    <button
      type={type}
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-md text-[13px] max-sm:whitespace-nowrap transition-[background-color,filter] disabled:pointer-events-none disabled:opacity-50 ${size} ${variants[variant]} ${className}`}
      {...props}
    >
      {Icon && <Icon size={15} strokeWidth={1.5} aria-hidden />}
      {!iconOnly && children}
    </button>
  );
}
