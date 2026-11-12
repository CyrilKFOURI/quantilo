import { Link, useRouterState } from "@tanstack/react-router";
import {
  BookOpen,
  BookMarked,
  House,
  LayoutGrid,
  PanelLeftClose,
  PanelLeftOpen,
  ScatterChart,
  Settings,
  Toolbox,
  type LucideIcon,
} from "lucide-react";

import { useColumns, useDataset } from "../../api/hooks";
import { useCurrentDataset } from "../../features/datasets/current";
import { ThemeSwitcher } from "../../features/theme/ThemeSwitcher";
import { ToolboxMenu } from "../../features/toolbox/ToolboxMenu";
import { inToolbox } from "../../features/toolbox/tools";
import { LockBadge } from "../../lite/Locked";
import { t, type MessageKey } from "../../lib/i18n";
import { meets, type ModuleManifest } from "../../modules/types";
import { CyfouriLogo } from "../../ui/CyfouriLogo";
import { Logo } from "../../ui/Logo";
import { groupEntry, groupOf, type PageGroup } from "./PageGroups";

/** Pages of the open dataset (046): two entries, each a group of pages with its own tabs
 *  (Overview: summary, columns, data, changes; Relationships: two variables, correlations). */
const DATASET_PAGES: { labelKey: MessageKey; icon: LucideIcon; group: PageGroup }[] = [
  { labelKey: "nav.overview", icon: LayoutGrid, group: "overview" },
  { labelKey: "nav.relationships", icon: ScatterChart, group: "relationships" },
];

const activeClass = "bg-accent-soft !text-ink-1 font-medium";

const itemClass =
  "flex h-(--control-h) items-center gap-2.5 rounded-md px-2 text-ink-2 transition-colors";

type SidebarProps = {
  collapsed: boolean;
  onToggle: () => void;
  modules: ModuleManifest[];
  /** In the phone and tablet drawer (041): full width of the drawer, a link closes it. */
  drawer?: boolean;
  onNavigate?: () => void;
};

export function Sidebar({ collapsed, onToggle, modules, drawer = false, onNavigate }: SidebarProps) {
  // 098: on Home or Recipes, the dataset entries still lead back to the dataset one works on
  const { datasetId } = useCurrentDataset();
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const current = groupOf(pathname);
  const datasetModules = modules.filter((module): module is DatasetModule => module.scope === "dataset" && module.section === "dataset");
  const analyses = modules.filter((module) => module.nav !== false && module.section !== "dataset");
  return (
    <nav
      aria-label={t("nav.label")}
      onClick={(event) => {
        if (onNavigate && (event.target as Element).closest("a")) onNavigate();
      }}
      className={`flex h-full flex-col bg-surface-1 ${
        drawer
          ? "w-full"
          : `border-r border-line transition-[width] duration-150 ${
              collapsed ? "w-(--sidebar-w-collapsed)" : "w-(--sidebar-w)"
            }`
      }`}
    >
      <div
        className={`flex h-(--topbar-h) items-center border-b border-line ${collapsed ? "justify-center" : "px-3"}`}
      >
        <Link
          to="/"
          className="flex min-w-0 items-center gap-2.5 rounded-md text-ink-1"
          aria-label={t("app.name")}
        >
          <CyfouriLogo size={18} />
          {!collapsed && (
            <>
              <span className="h-7 w-px shrink-0 bg-line" aria-hidden />
              <Logo size={24} className="shrink-0" />
              <span className="flex min-w-0 flex-col">
                <span className="truncate text-sm leading-tight font-semibold tracking-tight">
                  {t("app.name")}
                </span>
                <span className="truncate text-[11px] leading-tight text-ink-3">{t("app.tagline")}</span>
              </span>
            </>
          )}
        </Link>
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto px-2 py-3">
        <ul className="space-y-0.5">
          <li>
            <NavLink
              to="/"
              icon={House}
              label={t("nav.home")}
              collapsed={collapsed}
            />
          </li>
          <li>
            <NavLink
              to="/recipes"
              icon={BookMarked}
              label={t("nav.recipes")}
              collapsed={collapsed}
            />
          </li>
        </ul>

        <NavSection title={t("nav.section.dataset")} collapsed={collapsed}>
          {DATASET_PAGES.map(({ labelKey, icon: Icon, group }, index) => [
            // 065: the dataset's own modules (Clean) come right under Overview
            ...(index === 1
              ? datasetModules.map((module) => (
                  <li key={module.id}>
                    <DatasetModuleLink module={module} datasetId={datasetId} collapsed={collapsed} active={false} />
                  </li>
                ))
              : []),
            <li key={labelKey}>
              {datasetId ? (
                <Link
                  to={groupEntry(group) as never}
                  params={{ datasetId } as never}
                  title={collapsed ? t(labelKey) : undefined}
                  aria-label={collapsed ? t(labelKey) : undefined}
                  aria-current={current === group ? "page" : undefined}
                  className={`${itemClass} hover:bg-surface-2 hover:text-ink-1 ${current === group ? activeClass : ""}`}
                >
                  <Icon size={16} strokeWidth={1.5} aria-hidden />
                  {!collapsed && t(labelKey)}
                </Link>
              ) : (
                <span
                  aria-disabled="true"
                  title={
                    datasetId ? t("nav.comingSoon") : t("nav.needsDataset")
                  }
                  className={`${itemClass} cursor-not-allowed opacity-45`}
                >
                  <Icon size={16} strokeWidth={1.5} aria-hidden />
                  {!collapsed && t(labelKey)}
                </span>
              )}
            </li>,
          ])}
        </NavSection>

        {analyses.length > 0 && (
          <NavSection title={t("nav.section.modules")} collapsed={collapsed}>
            {analyses.map((module) => (
              <li key={module.id}>
                {module.scope === "global" ? (
                  <NavLink
                    to={module.path}
                    icon={module.icon}
                    label={t(module.labelKey)}
                    collapsed={collapsed}
                  />
                ) : (
                  <DatasetModuleLink
                    module={module}
                    datasetId={datasetId}
                    collapsed={collapsed}
                    active={Boolean(module.group) && current === module.group}
                  />
                )}
              </li>
            ))}
          </NavSection>
        )}

        {/* 098: the manual tools (time series, experiments, statistics, rows), last of the list */}
        <ul className="space-y-0.5 border-t border-line pt-3">
          <li>
            {datasetId ? (
              <ToolboxMenu
                datasetId={datasetId}
                modules={modules}
                collapsed={collapsed}
                drawer={drawer}
                itemClass={itemClass}
                active={inToolbox(pathname)}
              />
            ) : (
              <span
                aria-disabled="true"
                title={t("nav.needsDataset")}
                className={`${itemClass} cursor-not-allowed opacity-45`}
              >
                <Toolbox size={16} strokeWidth={1.5} aria-hidden />
                {!collapsed && t("nav.toolbox")}
              </span>
            )}
          </li>
        </ul>
      </div>

      <div className="space-y-2 border-t border-line px-2 py-2">
        <NavLink
          to="/help"
          icon={BookOpen}
          label={t("nav.help")}
          collapsed={collapsed}
        />
        <NavLink
          to="/settings"
          icon={Settings}
          label={t("nav.settings")}
          collapsed={collapsed}
        />
        <div
          className={`flex items-center gap-1 ${collapsed ? "flex-col" : "justify-between"}`}
        >
          {!collapsed && <ThemeSwitcher iconOnly />}
          <button
            type="button"
            onClick={onToggle}
            aria-label={drawer ? t("nav.closeMenu") : collapsed ? t("nav.expand") : t("nav.collapse")}
            title={drawer ? t("nav.closeMenu") : collapsed ? t("nav.expand") : t("nav.collapse")}
            className="inline-flex size-(--control-h) items-center justify-center rounded-md text-ink-2 hover:bg-surface-2 hover:text-ink-1"
          >
            {collapsed ? (
              <PanelLeftOpen size={16} strokeWidth={1.5} aria-hidden />
            ) : (
              <PanelLeftClose size={16} strokeWidth={1.5} aria-hidden />
            )}
          </button>
        </div>
      </div>
    </nav>
  );
}

type DatasetModule = Extract<ModuleManifest, { scope: "dataset" }>;

/** A module page of the open dataset; greyed out without a dataset or the columns it needs. */
function DatasetModuleLink({
  module,
  datasetId,
  collapsed,
  active,
}: {
  module: DatasetModule;
  datasetId: string | undefined;
  collapsed: boolean;
  /** One of its group's tabs is open (Machine learning while on Segments). */
  active: boolean;
}) {
  // A group's entry leads to its first tab the build can run (browser version: Segments, 046).
  const to = module.group ? groupEntry(module.group) : module.path;
  const locked = module.locked && to === module.path;
  const dataset = useDataset(datasetId);
  const columns = useColumns(
    datasetId ?? "",
    Boolean(datasetId) && dataset.data?.status === "ready",
  );
  const label = t(module.labelKey);
  const Icon = module.icon;
  const missing =
    module.requires &&
    columns.data &&
    !module.requires.every((requirement) => meets(requirement, dataset.data?.format, columns.data));
  if (!datasetId || missing) {
    return (
      <span
        aria-disabled="true"
        title={
          !datasetId
            ? t("nav.needsDataset")
            : module.requiresKey
              ? t(module.requiresKey)
              : undefined
        }
        className={`${itemClass} cursor-not-allowed opacity-45`}
      >
        <Icon size={16} strokeWidth={1.5} aria-hidden />
        {!collapsed && label}
      </span>
    );
  }
  return (
    <Link
      to={to as never}
      params={{ datasetId } as never}
      title={collapsed ? label : undefined}
      aria-label={collapsed ? label : undefined}
      aria-current={active ? "page" : undefined}
      className={`${itemClass} hover:bg-surface-2 hover:text-ink-1 ${active ? activeClass : ""}`}
      activeOptions={{ includeSearch: false }}
      activeProps={{ className: activeClass }}
    >
      <Icon size={16} strokeWidth={1.5} aria-hidden />
      {!collapsed && label}
      {!collapsed && locked && <LockBadge className="ml-auto" />}
    </Link>
  );
}

function NavSection({
  title,
  collapsed,
  children,
}: {
  title: string;
  collapsed: boolean;
  children: React.ReactNode;
}) {
  return (
    <div>
      {collapsed ? (
        <div className="mx-2 mb-2 border-t border-line" />
      ) : (
        <h2 className="mb-1 px-2 text-xs text-ink-3">{title}</h2>
      )}
      <ul className="space-y-0.5">{children}</ul>
    </div>
  );
}

function NavLink({
  to,
  icon: Icon,
  label,
  collapsed,
}: {
  to: string;
  icon: LucideIcon;
  label: string;
  collapsed: boolean;
}) {
  return (
    <Link
      to={to}
      title={collapsed ? label : undefined}
      aria-label={collapsed ? label : undefined}
      className={`${itemClass} hover:bg-surface-2 hover:text-ink-1`}
      activeOptions={{ exact: to === "/" }}
      activeProps={{ className: activeClass }}
    >
      <Icon size={16} strokeWidth={1.5} aria-hidden />
      {!collapsed && label}
    </Link>
  );
}
