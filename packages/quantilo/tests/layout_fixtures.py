"""Small synthetic workbooks, one per layout family of the table detection (feature 080).

Built with openpyxl in the test's tmp folder: no corpus file is ever copied into the repository.
Rows and columns in the comments are 1-based, as Excel shows them.
"""

import datetime as dt
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Border, Font, PatternFill, Side
from openpyxl.worksheet.worksheet import Worksheet

BOLD = Font(bold=True)
FILL = PatternFill("solid", fgColor="FFDDEBF7")
LINE = Border(bottom=Side(style="thin"))


def _active(wb: Workbook) -> Worksheet:
    sheet = wb.active
    assert isinstance(sheet, Worksheet)
    return sheet


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _style_header(sheet: Worksheet, row: int, first: int, last: int) -> None:
    for col in range(first, last + 1):
        cell = sheet.cell(row, col)
        cell.font, cell.fill, cell.border = BOLD, FILL, LINE


def cash_book(path: Path) -> Path:
    """Title rows, 'Income' (not merged) over 4 accounts, a totals row under the header, empty
    columns inside the table, 'Total Money In' = sum of the accounts, one row with two amounts."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Example"
    ws["B1"], ws["F1"] = "My Example Business", "EXAMPLE PAGE"
    ws["B2"], ws["E2"], ws["H2"] = "January", "Month 1", "MONEY IN"
    ws["B3"], ws["H3"], ws["N3"] = "Details", "Income", "Expenses"
    ws["L3"] = "Total Money In"
    ws.merge_cells("L3:L4")
    ws["P3"] = "Total Money Out"
    ws.merge_cells("P3:P4")
    names = {
        2: "Date",
        3: "Name",
        4: "Ref",
        8: "Capital",
        9: "Coaching",
        10: "Sales",
        11: "Travel",
        14: "Rent",
        15: "Phone",
    }
    for col, name in names.items():
        ws.cell(4, col, name)
    _style_header(ws, 3, 2, 16)
    _style_header(ws, 4, 2, 16)
    income = [(8, 2500), (9, 900), (10, 345.45), (9, 120), (11, 60), (10, 80), (8, 40), (9, 75)]
    expense = [(14, 600), (15, 45), (14, 600), (15, 30), (15, 25), (14, 10), (15, 12), (14, 7)]
    rows = []
    for i in range(16):
        account, amount = income[i // 2] if i % 2 == 0 else expense[i // 2]
        rows.append({account: amount})
    rows[2][11] = 244.64  # a payment split over two income accounts
    for offset, amounts in enumerate(rows):
        r = 6 + offset
        ws.cell(r, 2, dt.datetime(2026, 1, 1 + offset))
        ws.cell(r, 3, f"Party {offset + 1}")
        for col, amount in amounts.items():
            ws.cell(r, col, amount)
        ws.cell(r, 12, sum(v for c, v in amounts.items() if 8 <= c <= 11))
        ws.cell(r, 16, sum(v for c, v in amounts.items() if 14 <= c <= 15))
    ws["B5"] = "DO NOT DELETE THIS TOTALS ROW"
    for col in (8, 9, 10, 11, 12, 14, 15, 16):
        ws.cell(5, col, round(sum(_number(ws.cell(r, col).value) for r in range(6, 22)), 2))
        ws.cell(5, col).font = BOLD
    ws.freeze_panes = "C5"
    wb.save(path)
    return path


def multi_level(path: Path) -> Path:
    """Census-like: title, a merged parent over 4 years, merged stub, a bold total of the 4
    regions, '.' prefixed states, a footer."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "POP"
    ws["A1"] = "Annual Estimates of the Resident Population"
    ws.merge_cells("A1:F1")
    ws["A2"], ws["B2"], ws["C2"] = "Geographic Area", "April 1, 2020 Base", "Population Estimate"
    ws.merge_cells("A2:A3")
    ws.merge_cells("B2:B3")
    ws.merge_cells("C2:F2")
    for i, year in enumerate((2020, 2021, 2022, 2023)):
        ws.cell(3, 3 + i, year)
    _style_header(ws, 2, 1, 6)
    _style_header(ws, 3, 1, 6)
    regions = {"Northeast": 100, "Midwest": 200, "South": 300, "West": 400}
    states = [".Alabama", ".Alaska", ".Arizona", ".Arkansas", ".California", ".Colorado"]
    r = 5
    for name, base in regions.items():
        ws.cell(r, 1, name).font = BOLD
        for c in range(2, 7):
            ws.cell(r, c, base + c)
        r += 1
    for i, name in enumerate(states):
        ws.cell(r, 1, name)
        for c in range(2, 7):
            ws.cell(r, c, 10 * i + c)
        r += 1
    ws.cell(4, 1, "United States").font = BOLD
    for c in range(2, 7):
        ws.cell(4, c, sum(_number(ws.cell(x, c).value) for x in range(5, 9)))
    ws.cell(r + 1, 1, "Note: The estimates are based on the 2020 Census.")
    ws.cell(r + 2, 1, "Source: U.S. Census Bureau")
    wb.save(path)
    return path


def indented(path: Path) -> Path:
    """A hierarchy by NBSP indentation (NCES): 2 groups with their children, a column-number row,
    a units row, footnote references in the names."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Table"
    ws["A1"] = "Table 1. Earnings by education level"
    ws.append(["Level", "Earnings\\2\\", "Workers"])
    ws.append(["", "(dollars)", "(thousands)"])
    ws.append([1, 2, 3])
    rows = [
        ("All levels", 50, 900),
        ("\xa0\xa0High school", 30, 400),
        ("\xa0\xa0\xa0\xa0Diploma", 31, 300),
        ("\xa0\xa0\xa0\xa0No diploma", 25, 100),
        ("\xa0\xa0College", 70, 500),
        ("\xa0\xa0\xa0\xa0Bachelor", 65, 350),
        ("\xa0\xa0\xa0\xa0Master", 80, 150),
    ]
    for row in rows:
        ws.append(list(row))
    ws.append([])
    ws.append(["\\2\\Constant 2022 dollars."])
    ws.append(["SOURCE: Survey of earnings."])
    wb.save(path)
    return path


def side_by_side(path: Path) -> Path:
    """Two tables separated by one empty column, of different lengths."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Two"
    ws.append(["Product", "Units", "Price", None, "Region", "Target"])
    for i in range(8):
        ws.append([f"P{i}", i * 3, 1.5 * i] + ([None, f"R{i}", 100 + i] if i < 3 else []))
    for col in (1, 2, 3, 5, 6):
        ws.cell(1, col).font = BOLD
    wb.save(path)
    return path


def stacked(path: Path) -> Path:
    """Two blocks with the same header, each under a section title (h08, hstpov2)."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Stack"
    ws["A1"] = "Median income by state"
    r = 3
    for section, factor in (("Current dollars", 1.0), ("2025 dollars", 1.2)):
        ws.cell(r, 1, section).font = BOLD
        for c, name in enumerate(("State", "Median income", "Standard error"), start=1):
            ws.cell(r + 1, c, name).font = BOLD
        for i, state in enumerate(("Alabama", "Alaska", "Arizona", "Arkansas")):
            ws.cell(r + 2 + i, 1, state)
            ws.cell(r + 2 + i, 2, round(50000 * factor + 1000 * i))
            ws.cell(r + 2 + i, 3, 500 + i)
        r += 8
    wb.save(path)
    return path


def transposed(path: Path) -> Path:
    """Field labels in column A, one record per column."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Records"
    ws.append(["Field", "Alice", "Bob", "Carol"])
    ws.append(["City", "Paris", "Lyon", "Nice"])
    ws.append(["Age", 31, 45, 28])
    ws.append(["Joined", dt.datetime(2020, 1, 1), dt.datetime(2019, 6, 1), dt.datetime(2021, 3, 1)])
    ws.append(["Score", 7.5, 6.0, 9.1])
    for col in range(1, 5):
        ws.cell(1, col).font = BOLD
    wb.save(path)
    return path


def pivot_export(path: Path) -> Path:
    """A pivot-table export: labels not repeated, '(blank)', 'Grand Total'."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Pivot"
    ws.append(["Sum of Sales"])
    ws.append([])
    ws.append(["Row Labels", "Product", "Sales", "Units"])
    data = [
        ("East", "A", 10, 1),
        (None, "B", 20, 2),
        (None, "C", 5, 1),
        ("West", "A", 7, 3),
        (None, "B", 8, 1),
        ("(blank)", "C", 3, 1),
    ]
    for row in data:
        ws.append(list(row))
    ws.append(["Grand Total", None, 53, 9])
    for col in range(1, 5):
        ws.cell(3, col).font = BOLD
    wb.save(path)
    return path


def timesheet(path: Path) -> Path:
    """Dates across the columns, one row per person."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Hours"
    ws["A1"] = "Timesheet — week 2"
    days = [dt.datetime(2026, 1, 5 + i) for i in range(5)]
    ws.append([])
    ws.append(["Employee", *days])
    for name, hours in (("Ann", 8), ("Ben", 7), ("Cleo", 6), ("Dan", 8)):
        ws.append([name] + [hours - (i % 2) for i in range(5)])
    for col in range(1, 7):
        ws.cell(3, col).font = BOLD
    wb.save(path)
    return path


def owner_like(path: Path) -> Path:
    """The owner's case: title A1, note A2, styled header row 4, frozen pane B5, an all-text body
    with placeholders, a section title and a second text table; a sheet with formulas written by
    a program (no cached value)."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Players"
    ws["A1"] = "Market attractiveness — existing players"
    ws["A1"].font = BOLD
    ws["A2"] = "n/d = not disclosed / not found in public sources"
    header = ["Player", "Type", "HQ", "Founded", "Ticket"]
    for c, name in enumerate(header, start=1):
        ws.cell(4, c, name)
    _style_header(ws, 4, 1, 5)
    rows = [
        ("Kiwi", "Donation", "Casablanca", "2023", "800 MAD"),
        ("Akkan", "Lending", "Morocco", "n/d", "n/d"),
        ("Cotizi", "Pools", "Morocco", "~2015", "n/a"),
        ("Ulule", "Reward", "France", "2009-10", "—"),
        ("Eureeca", "Equity", "Dubai", "2013", "n/d"),
    ]
    for i, row in enumerate(rows):
        for c, value in enumerate(row, start=1):
            ws.cell(5 + i, c, value)
    ws.freeze_panes = "B5"
    ws["A11"] = "Regulatory frame"
    ws["A11"].font = BOLD
    for c, name in enumerate(("Rule", "Value", "Source"), start=1):
        ws.cell(12, c, name)
    _style_header(ws, 12, 1, 3)
    for i, row in enumerate(
        (
            ("Cap per project", "5M MAD", "ammc.ma"),
            ("Cap per person", "500k MAD", "ammc.ma"),
            ("Supervisor", "AMMC", "ammc.ma"),
        )
    ):
        for c, value in enumerate(row, start=1):
            ws.cell(13 + i, c, value)
    tam = wb.create_sheet("TAM")
    tam["A1"] = "TAM model"
    tam["A1"].font = BOLD
    for c, name in enumerate(("Input", "Value", "Source"), start=1):
        tam.cell(3, c, name)
    _style_header(tam, 3, 1, 3)
    for i, (name, value) in enumerate(
        (("Startups", 1000), ("Share raising", 0.3), ("Round", 2e6), ("Take rate", 0.1))
    ):
        tam.cell(4 + i, 1, name)
        tam.cell(4 + i, 2, value)
        tam.cell(4 + i, 3, "Assumption")
    for c, name in enumerate(("Level", "Money raised", "Revenue", "Formula"), start=1):
        tam.cell(10, c, name)
    _style_header(tam, 10, 1, 4)
    for i, level in enumerate(("TAM", "SAM", "SOM")):
        tam.cell(11 + i, 1, level)
        tam.cell(11 + i, 2, f"=B4*B5*B6*{1 - i / 4}")
        tam.cell(11 + i, 3, f"=B{11 + i}*B7")
        tam.cell(11 + i, 4, "startups × share × round")
    wb.save(path)
    return path


def templates(path: Path) -> Path:
    """An empty monthly template: header, a totals row and rows of formulas giving 0 (C11)."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Month1"
    for c, name in enumerate(("Date", "Name", None, "Income", "Expense", None, "Balance"), start=1):
        if name:
            ws.cell(1, c, name).font = BOLD
    for r in range(2, 12):
        ws.cell(r, 4, 0)
        ws.cell(r, 5, 0)
        ws.cell(r, 7, 0)
    wb.save(path)
    return path


def typed_text(path: Path) -> Path:
    """Numbers written as text (decimal comma, percent, thousands), codes, placeholders, flags."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Text"
    ws.append(["Code", "Turnout", "Voters", "Population", "Note"])
    rows = [
        ("01", "69,55%", "1 234", "74794 e", "ok"),
        ("2A", "60,10%", "2 345", "75100", "ok"),
        ("ZX", "28,67%", "n/d", "75900 p", "late"),
        ("12", "55,00%", "3 100", ":", "ok"),
        ("75", "61,20%", "4 000", "76100 e", "ok"),
        ("13", "58,90%", "1 500", "76500", "ok"),
    ]
    for row in rows:
        ws.append(list(row))
    for col in range(1, 6):
        ws.cell(1, col).font = BOLD
    wb.save(path)
    return path


def numbered_groups(path: Path) -> Path:
    """Repeated numbered column groups (election results: 'Nuance 1, Voix 1, Nuance 2, Voix 2…')."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Results"
    header = ["Dept", "Registered"]
    for n in (1, 2, 3):
        header += [f"Nuance {n}", f"Voix {n}"]
    ws.append(header)
    for d in range(5):
        ws.append([f"D{d}", 1000 + d, "ENS", 300 + d, "RN", 250 + d, "LFI", 200 + d])
    wb.save(path)
    return path


def plain(path: Path) -> Path:
    """A clean table, header on row 1: nothing should change."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Data"
    ws.append(["id", "city", "amount"])
    for i in range(10):
        ws.append([i, f"city{i % 3}", 10.5 * i])
    wb.save(path)
    return path


def text_sheet(path: Path) -> Path:
    """An 'About' sheet: paragraphs only, no table."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "About"
    for r, text in enumerate(
        (
            "About this template",
            "This free cash book helps you track money in and out of your business.",
            "Visit the website for more help.",
        ),
        start=1,
    ):
        ws.cell(r * 2, 2, text)
    wb.save(path)
    return path


def long_sheet(path: Path, rows: int = 120) -> Path:
    """More rows than a small analysis window: the body end is streamed (B1)."""
    wb = Workbook()
    ws = _active(wb)
    ws.title = "Long"
    ws["A1"] = "A long series"
    ws.append([])
    ws.append(["day", "value"])
    for i in range(rows):
        ws.append([i, i * 2.5])
    ws.append([])
    ws.append(["Source: generated"])
    wb.save(path)
    return path


VERBOSE_CSV = (
    "Consumer prices, monthly\n"
    "Source: Statistics office,\n"
    "Extracted on 2026-11-10\n"
    "month,region,index,change\n"
    "2026-01,North,101.2,0.3\n"
    "2026-02,North,101.9,0.7\n"
    "2026-03,North,102.4,0.5\n"
    "2026-01,South,99.8,0.1\n"
)

METADATA_CSV = (
    '"Series Description","Yield 1-month","Yield 3-month"\n'
    '"Unit:","Percent","Percent"\n'
    '"Multiplier:","1","1"\n'
    '"Unique Identifier: ","H15/M01","H15/M03"\n'
    '"Time Period","M01","M03"\n'
    "2026-01-02,5.55,5.40\n"
    "2026-01-03,5.54,5.41\n"
    "2026-01-04,ND,5.39\n"
    "2026-01-05,5.52,5.38\n"
)

COMBINED_TSV = (
    "freq,geo\\TIME_PERIOD\t2015 \t2016 \t2017 \t2018 \n"
    "A,AD\t: \t71732 \t: \t74794 e\n"
    "A,AL\t2885796 \t2875592 \t2876591 \t2870324 \n"
    "A,AM\t3010598 \t2998577 \t2986151 \t2972732 p\n"
    "A,AT\t8576261 \t8700471 \t8772865 \t8822267 \n"
)


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
