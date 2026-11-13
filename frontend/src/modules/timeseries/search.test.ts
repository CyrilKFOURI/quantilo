import { describe, expect, it } from "vitest";

import { forecastTuning, seriesParams, validateTimeSeriesSearch } from "./search";

describe("forecast options in the URL (028 A7)", () => {
  it("keeps the offered values only and sends them as the tuning", () => {
    const search = validateTimeSeriesSearch({
      time: "day",
      arima_p: "5",
      arima_q: "9", // not offered
      arima_ic: "bic",
      gbm_grid: "huge", // not offered
      cp_scale: "0.5",
      fc_limit: "60",
    });

    expect(search.arima_q).toBeUndefined();
    expect(search.gbm_grid).toBeUndefined();
    expect(forecastTuning(search)).toEqual({
      max_p: 5,
      criterion: "bic",
      changepoint_scale: 0.5,
      time_limit: 60,
    });
    expect(seriesParams(search)).toEqual(expect.not.objectContaining({ arima_p: 5 }));
    expect(forecastTuning(validateTimeSeriesSearch({}))).toEqual({});
  });
});
