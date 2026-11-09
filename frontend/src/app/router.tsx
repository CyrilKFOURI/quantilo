import {
  createBrowserHistory,
  createRootRoute,
  createRoute,
  createRouter,
  retainSearchParams,
  type RouterHistory,
} from "@tanstack/react-router";

import { validateFilterSearch, type FilterSearch } from "../api/filters";
import { modules as registeredModules, panels as registeredPanels } from "../modules/registry";
import type { ModuleManifest, PanelManifest } from "../modules/types";
import { ChangesPage, validateChangesSearch } from "../pages/changes/ChangesPage";
import { ColumnsPage } from "../pages/columns/ColumnsPage";
import { HelpPage } from "../pages/help/HelpPage";
import { HomePage } from "../pages/home/HomePage";
import { NotFoundPage } from "../pages/not-found/NotFoundPage";
import { OverviewPage } from "../pages/overview/OverviewPage";
import { RecipePage, validateRecipeSearch } from "../pages/recipes/RecipePage";
import { RecipesPage } from "../pages/recipes/RecipesPage";
import { SettingsPage } from "../pages/settings/SettingsPage";
import {
  ComparePage,
  validateCompareSearch,
} from "../pages/compare/ComparePage";
import { CorrelationsPage } from "../pages/correlations/CorrelationsPage";
import { DataPage, validateDataSearch } from "../pages/data/DataPage";
import { VariablePage } from "../pages/variable/VariablePage";
import { Shell } from "./layout/Shell";

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
    notFoundComponent: NotFoundPage,
  });

  const homeRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/",
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
    component: OverviewPage,
  });
  const columnsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/columns",
    component: ColumnsPage,
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
    component: ComparePage,
  });
  const correlationsRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/correlations",
    component: CorrelationsPage,
  });
  const dataRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/data",
    validateSearch: validateDataSearch,
    component: DataPage,
  });
  const changesRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/d/$datasetId/changes",
    validateSearch: validateChangesSearch,
    component: ChangesPage,
  });
  const moduleRoutes = modules.map((module) =>
    createRoute({
      getParentRoute: () => rootRoute,
      path: module.path,
      component: module.component,
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
      recipesRoute,
      recipeRoute,
      datasetRoute,
      columnsRoute,
      variableRoute,
      compareRoute,
      correlationsRoute,
      dataRoute,
      changesRoute,
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
