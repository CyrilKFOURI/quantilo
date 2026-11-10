import * as RadixDialog from "@radix-ui/react-dialog";
import { Outlet } from "@tanstack/react-router";
import { useState } from "react";

import { KeyboardLayer } from "../../features/command/Shortcuts";
import { t } from "../../lib/i18n";
import { BELOW_LAPTOP, PHONE, useMedia } from "../../lib/useMedia";
import type { ModuleManifest, PanelManifest } from "../../modules/types";
import { DatasetGuard } from "../../pages/not-found/NotFoundPage";
import { ActivityBar } from "../../ui/ActivityBar";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

const COLLAPSED_KEY = "de.sidebarCollapsed";

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSED_KEY) === "true";
  } catch {
    return false;
  }
}

export function Shell({ modules, panels = [] }: { modules: ModuleManifest[]; panels?: PanelManifest[] }) {
  const [collapsed, setCollapsed] = useState(readCollapsed);
  const [drawer, setDrawer] = useState(false);
  // 041: a phone has no sidebar (the menu button opens the drawer); a tablet keeps the icon
  // rail, whose button opens the drawer with the labels. A laptop is unchanged.
  const phone = useMedia(PHONE);
  const belowLaptop = useMedia(BELOW_LAPTOP);

  const toggle = () => {
    const next = !collapsed;
    setCollapsed(next);
    try {
      localStorage.setItem(COLLAPSED_KEY, String(next));
    } catch {
      // Not persisted when storage is unavailable.
    }
  };

  return (
    <div className="flex h-dvh overflow-hidden bg-surface-0 text-ink-1">
      <a
        href="#main"
        className="sr-only z-50 rounded-md bg-accent px-3 py-2 text-accent-ink focus:not-sr-only focus:fixed focus:top-2 focus:left-2"
      >
        {t("nav.skip")}
      </a>
      {!phone && (
        <Sidebar
          collapsed={belowLaptop || collapsed}
          onToggle={belowLaptop ? () => setDrawer(true) : toggle}
          modules={modules}
        />
      )}
      {belowLaptop && (
        <RadixDialog.Root open={drawer} onOpenChange={setDrawer}>
          <RadixDialog.Portal>
            <RadixDialog.Overlay className="fixed inset-0 z-40 bg-ink-1/30" />
            <RadixDialog.Content
              aria-describedby={undefined}
              className="fixed inset-y-0 left-0 z-50 w-[min(85vw,300px)] border-r border-line bg-surface-1 pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)] shadow-[0_12px_40px_-12px_rgb(0_0_0/0.35)]"
            >
              <RadixDialog.Title className="sr-only">{t("nav.label")}</RadixDialog.Title>
              <Sidebar
                drawer
                collapsed={false}
                onToggle={() => setDrawer(false)}
                onNavigate={() => setDrawer(false)}
                modules={modules}
              />
            </RadixDialog.Content>
          </RadixDialog.Portal>
        </RadixDialog.Root>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onMenu={phone ? () => setDrawer(true) : undefined} appMark={!phone && (belowLaptop || collapsed)} />
        <ActivityBar />
        <main
          id="main"
          tabIndex={-1}
          className="relative min-h-0 flex-1 overflow-y-auto overscroll-contain outline-none"
        >
          <DatasetGuard>
            <Outlet />
          </DatasetGuard>
        </main>
      </div>
      {panels.map(({ id, component: Panel }) => (
        <Panel key={id} />
      ))}
      <KeyboardLayer modules={modules} />
    </div>
  );
}
