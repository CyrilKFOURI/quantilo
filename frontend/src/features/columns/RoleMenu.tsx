import * as Menu from "@radix-ui/react-dropdown-menu";
import { Check, ChevronDown } from "lucide-react";

import type { Role } from "../../api/client";
import { roleName, t } from "../../lib/i18n";

type RoleMenuProps = {
  label: string;
  value: string;
  roles: Role[];
  onChange: (role: string) => void;
};

/** Compact role picker; the list scrolls inside the space left on screen. */
export function RoleMenu({ label, value, roles, onChange }: RoleMenuProps) {
  const current = roles.find((role) => role.id === value);
  const options = [{ id: "", label: t("columns.noRole") }, ...roles];

  return (
    <Menu.Root>
      <Menu.Trigger
        aria-label={label}
        className="inline-flex h-7 max-w-36 items-center gap-1 rounded-md border border-transparent px-1.5 text-[13px] hover:border-line hover:bg-surface-1 data-[state=open]:border-line data-[state=open]:bg-surface-1"
      >
        <span className={`truncate ${current ? "text-ink-1" : "text-ink-3"}`}>
          {current ? roleName(current.id, current.label) : t("columns.noRole")}
        </span>
        <ChevronDown size={13} strokeWidth={1.5} className="shrink-0 text-ink-3" aria-hidden />
      </Menu.Trigger>
      <Menu.Portal>
        <Menu.Content
          align="start"
          sideOffset={4}
          collisionPadding={12}
          className="z-50 max-h-[min(320px,var(--radix-dropdown-menu-content-available-height))] w-48 overflow-y-auto overscroll-contain rounded-lg border border-line bg-surface-1 p-1 shadow-[0_12px_32px_-12px_rgb(0_0_0/0.35)]"
        >
          <Menu.RadioGroup value={value} onValueChange={onChange}>
            {options.map((role) => (
              <Menu.RadioItem
                key={role.id || "none"}
                value={role.id}
                className={`flex h-7 cursor-default items-center justify-between gap-2 rounded-md px-2 text-[13px] outline-none data-[highlighted]:bg-surface-2 ${
                  role.id ? "text-ink-1" : "text-ink-3"
                }`}
              >
                <span className="truncate">{role.id ? roleName(role.id, role.label) : role.label}</span>
                <Menu.ItemIndicator>
                  <Check size={14} strokeWidth={1.5} className="text-accent" aria-hidden />
                </Menu.ItemIndicator>
              </Menu.RadioItem>
            ))}
          </Menu.RadioGroup>
        </Menu.Content>
      </Menu.Portal>
    </Menu.Root>
  );
}
