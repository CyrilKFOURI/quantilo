import {
  createBrowserHistory,
  createRootRoute,
  createRoute,
  createRouter,
  retainSearchParams,
  type RouterHistory,
} from "@tanstack/react-router";
import type { ReactElement } from "react";

import { validateFilterSearch, type FilterSearch } from "../api/filters";
import { modules as registeredModules, panels as registeredPanels } from "../modules/registry";
import type { ModuleManifest, PanelManifest } from "../modules/types";
import { ChangesPage, validateChangesSearch } from "../pages/changes/ChangesPage";
import { ColumnsPage } from "../pages/columns/ColumnsPage";
import { HelpPage } from "../pages/help/HelpPage";
import { HomePage, validateHomeSearch } from "../pages/home/HomePage";
import { LegalPage } from "../pages/legal/LegalPage";
import { NotFoundPage } from "../pages/not-found/NotFoundPage";
import { OverviewPage } from "../pages/overview/OverviewPage";
import { RecipePage, validateRecipeSearch } from "../pages/recipes/RecipePage";
import { RecipesPage } from "../pages/recipes/RecipesPage";
import { SettingsPage } from "../pages/settings/SettingsPage";
import { ToolboxPage } from "../pages/toolbox/ToolboxPage";
import {
  ComparePage,
  validateCompareSearch,
} from "../pages/compare/ComparePage";
import { CorrelationsPage } from "../pages/correlations/CorrelationsPage";
import { DataPage, validateDataSearch } from "../pages/data/DataPage";
import { RegressionPage, validateRegressionSearch } from "../pages/regression/RegressionPage";
import { VariablePage } from "../pages/variable/VariablePage";
import { Grouped, type PageGroup } from "./layout/PageGroups";
import { Shell } from "./layout/Shell";

/** 046: a page of a group gets the group's tabs above it. */
function inGroup(group: PageGroup, Page: () => ReactElement | null) {
  return function GroupedPage() {
    return (
      <Grouped group={group}>
        <Page />
      </Grouped>
    );
  };
}

type RouterOptions = {
  history?: RouterHistory;
  modules?: ModuleManifest[];
  panels?: PanelManifest[];
};

export function createAppRouter({
  history = createBrowserHistory(),
  modules = registeredModules,
  panels = registeredPanels,
}: RouterOptions = {}) {
  const rootRoute = createRootRoute({
    // Filters follow the user from page to page (see api/filters.ts).
    validateSearch: validateFilterSearch,
    search: { middlewares: [retainSearchParams<FilterSearch>(["filters"])] },
    component: () => <Shell modules={modules} panels={panels} />,
    notFoundComponent: () => <NotFoundPage />,
  });

  const homeRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/",
    // 087: `?project=<id>` opens one project
    validateSearch: validateHomeSearch,
    component: HomePage,
  });
  const settingsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/settings",
    component: SettingsPage,
  });
  const helpRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/help",
    component: HelpPage,
  });
  // 081: Privacy, Terms of use, Legal notice (reached from the Guide, Settings and Datasets footers).
  const legalRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/legal/$doc",
    component: LegalPage,
  });
  const recipesRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/recipes",
    component: RecipesPage,
  });
  const recipeRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/recipes/$recipeId",
    validateSearch: validateRecipeSearch,
    component: RecipePage,
  });
  const datasetRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId",
    component: inGroup("overview", OverviewPage),
  });
  const columnsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/columns",
    component: inGroup("overview", ColumnsPage),
  });
  const variableRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/variable/$column",
    component: VariablePage,
  });
  const compareRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/compare",
    validateSearch: validateCompareSearch,
    component: inGroup("relationships", ComparePage),
  });
  const correlationsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/correlations",
    component: inGroup("relationships", CorrelationsPage),
  });
  const regressionRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/regression",
    validateSearch: validateRegressionSearch,
    component: inGroup("relationships", RegressionPage),
  });
  const dataRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/data",
    validateSearch: validateDataSearch,
    component: inGroup("overview", DataPage),
  });
  const changesRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/changes",
    validateSearch: validateChangesSearch,
    component: inGroup("overview", ChangesPage),
  });
  // 098: the manual tools, grouped by what they do (the last entry of the sidebar)
  const toolboxRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/toolbox",
    component: () => <ToolboxPage modules={modules} />,
  });
  const moduleRoutes = modules.map((module) =>
    createRoute({
      getParentRoute: () => rootRoute,
      path: module.path,
      component: module.group ? inGroup(module.group, module.component) : module.component,
      ...(module.validateSearch
        ? { validateSearch: module.validateSearch }
        : {}),
    }),
  );

  return createRouter({
    routeTree: rootRoute.addChildren([
      homeRoute,
      settingsRoute,
      helpRoute,
      legalRoute,
      recipesRoute,
      recipeRoute,
      datasetRoute,
      columnsRoute,
      variableRoute,
      compareRoute,
      correlationsRoute,
      regressionRoute,
      dataRoute,
      changesRoute,
      toolboxRoute,
      ...moduleRoutes,
    ]),
    history,
  });
}

export type AppRouter = ReturnType<typeof createAppRouter>;

declare module "@tanstack/react-router" {
  interface Register {
    router: AppRouter;
  }
}
