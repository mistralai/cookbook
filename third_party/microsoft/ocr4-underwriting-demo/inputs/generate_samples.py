"""Generate the demo's sample mortgage documents as scanned-looking PDFs.

One persona (Dana R. Lee) runs across the whole package, so the extractor and the reviewer
see a consistent borrower. Pillow only, no browser or external service. Re-run with:

    uv run python inputs/generate_samples.py

Each document is drawn crisp, then passed through scanify() for a paper-scan look (tint,
grain, softness, a slight skew) and saved as a single-page PDF next to this script.
"""
from __future__ import annotations

import math
import os
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(7)  # deterministic scan artefacts, so re-runs are reproducible

# Letter at ~200 dpi.
W, H = 1700, 2200
M = 120  # page margin
INK = (24, 24, 28)
MUTED = (95, 100, 110)
PAPER = (247, 245, 239)

_AR = "/System/Library/Fonts/Supplemental/Arial.ttf"
_ARB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
_CN = "/System/Library/Fonts/Supplemental/Courier New.ttf"
# Handwriting faces (synthetic, so no real person's signature is used). SnellRoundhand is a
# flowing script for signatures; Bradley Hand is a printed hand for notes and filled fields.
_SIG = "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc"
_HAND = "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf"

# Ink shades for handwritten marks, so pen strokes read differently from printed text.
PEN_BLUE = (28, 42, 96)
PEN_BLACK = (30, 30, 34)


def f(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_CN if mono else (_ARB if bold else _AR), size)


def _hand_font(size: int, sig: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_SIG if sig else _HAND, size)


def hand(img, x, y, s, size=34, sig=False, fill=PEN_BLUE, jitter=2.0, rot=0.0):
    """Write text as if by hand onto `img`: each glyph nudged and the mark on a slight
    baseline wobble, so OCR sees handwriting rather than a clean font line. Rendered on a
    transparent layer (optionally rotated) then composited, so ink can sit at an angle over
    printed text."""
    ft = _hand_font(size, sig=sig)
    measure = ImageDraw.Draw(img)
    total = int(measure.textlength(s, font=ft)) + size
    layer = Image.new("RGBA", (total + 40, size * 2 + 40), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    cx = 20
    for i, ch in enumerate(s):
        dy = 20 + random.uniform(-jitter, jitter) + math.sin(i * 0.6) * (jitter * 0.6)
        ld.text((cx, dy), ch, font=ft, fill=fill + (255,))
        cx += measure.textlength(ch, font=ft) + random.uniform(-1.0, 1.5)
    if rot:
        layer = layer.rotate(rot, resample=Image.BICUBIC, expand=True)
    img.paste(layer, (int(x), int(y)), layer)
    return y


# ---- persona: matches loan_application_1003 ----
P = {
    "name": "Dana R. Lee",
    "ssn": "XXX-XX-4821",
    "addr": "500 Market Street, Seattle, WA 98104",
    "employer": "Northwind Analytics LLC",
    "employer_addr": "1200 5th Avenue, Seattle, WA 98101",
    "ein": "91-1234567",
    "employee_id": "NW-10293",
    "hire_date": "2020-05-11",
    "years": 6,
    "annual_income": 145000,
    "credit": 760,
    "monthly_debts": 700,
    "monthly_housing": 1850,
    "loan_amount": 340000,
    "property_value": 500000,
    "purpose": "Purchase",
}


def money(n, cents=False) -> str:
    return f"${n:,.2f}" if cents else f"${n:,.0f}"


# ---- drawing primitives ----
def page():
    img = Image.new("RGB", (W, H), "white")
    return img, ImageDraw.Draw(img)


def t(d, x, y, s, size=26, bold=False, mono=False, fill=INK, align="l", w=None):
    ft = f(size, bold, mono)
    if align != "l" and w is not None:
        tw = d.textlength(s, font=ft)
        x = x + (w - tw) if align == "r" else x + (w - tw) / 2
    d.text((x, y), s, font=ft, fill=fill)
    return y


def bar(d, x, y, w, label, bg=(38, 55, 84)):
    h = 46
    d.rectangle([x, y, x + w, y + h], fill=bg)
    d.text((x + 16, y + 9), label, font=f(24, bold=True), fill=(255, 255, 255))
    return y + h + 12


def rule(d, y, x0=M, x1=W - M, fill=(180, 180, 185), width=2):
    d.line([x0, y, x1, y], fill=fill, width=width)
    return y + 1


def cell(d, x, y, w, h, label, value, vsize=30, vmono=False):
    d.rectangle([x, y, x + w, y + h], outline=(170, 170, 175), width=2)
    d.text((x + 14, y + 10), label.upper(), font=f(17, bold=True), fill=MUTED)
    d.text((x + 14, y + 40), value, font=f(vsize, mono=vmono), fill=INK)
    return x + w


def row(d, x, y, cols, widths, aligns=None, size=24, bold=False, fill=INK):
    aligns = aligns or ["l"] * len(cols)
    cx = x
    for s, w, a in zip(cols, widths, aligns):
        t(d, cx, y, str(s), size=size, bold=bold, fill=fill, align=a, w=w - 20)
        cx += w
    return y


# ---- scanned look ----
def scanify(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = ImageChops.multiply(img, Image.new("RGB", img.size, (236, 232, 222)))  # warm paper

    # Uneven lighting: keep the center bright, let the edges fall off (a scan bed look).
    vig = Image.new("L", img.size, 0)
    ImageDraw.Draw(vig).ellipse(
        [-img.width * 0.28, -img.height * 0.22, img.width * 1.28, img.height * 1.22], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(260))
    darker = ImageChops.multiply(img, Image.new("RGB", img.size, (206, 204, 198)))
    img = Image.composite(img, darker, vig)

    noise = Image.effect_noise(img.size, 26).convert("RGB")           # sensor grain
    img = Image.blend(img, noise, 0.08)
    img = img.filter(ImageFilter.GaussianBlur(0.8))                   # scan softness
    img = img.rotate(random.uniform(-1.5, 1.5), resample=Image.BICUBIC,
                     expand=True, fillcolor=(236, 232, 222))          # visible skew

    # Lay the page on a scanner background with a soft drop shadow.
    pad = 46
    bg = Image.new("RGB", (img.width + pad * 2, img.height + pad * 2), (223, 221, 214))
    shadow = Image.new("L", bg.size, 0)
    ImageDraw.Draw(shadow).rectangle(
        [pad + 6, pad + 10, pad + 6 + img.width, pad + 10 + img.height], fill=110)
    shadow = shadow.filter(ImageFilter.GaussianBlur(20))
    bg.paste((120, 118, 112), (0, 0), shadow)
    bg.paste(img, (pad, pad))

    return ImageOps.autocontrast(bg, cutoff=1)


def save_pdf(img: Image.Image, name: str) -> None:
    out = os.path.join(HERE, name)
    scanify(img).convert("RGB").save(out, "PDF", resolution=150.0)
    print("wrote", os.path.relpath(out))


# ---- documents ----
def build_paystub():
    img, d = page()
    t(d, M, M, "EARNINGS STATEMENT", size=48, bold=True)
    t(d, M, M, P["employer"], size=26, bold=True, align="r", w=W - 2 * M)
    t(d, M, M + 36, P["employer_addr"], size=22, fill=MUTED, align="r", w=W - 2 * M)
    y = rule(d, M + 78)

    y += 18
    colw = (W - 2 * M) / 3
    cell(d, M, y, colw - 12, 92, "Employee name", P["name"])
    cell(d, M + colw, y, colw - 12, 92, "Employee ID", P["employee_id"])
    cell(d, M + 2 * colw, y, colw - 12, 92, "Years of service", str(P["years"]))
    y += 104
    cell(d, M, y, colw - 12, 92, "Pay period", "2026-07-01 to 2026-07-15", vsize=24)
    cell(d, M + colw, y, colw - 12, 92, "Pay date", "2026-07-18")
    cell(d, M + 2 * colw, y, colw - 12, 92, "Hire date", P["hire_date"])
    y += 128

    y = bar(d, M, y, W - 2 * M, "EARNINGS")
    cols_w = [560, 200, 180, 260, 260]
    row(d, M, y, ["DESCRIPTION", "RATE", "HOURS", "CURRENT", "YEAR TO DATE"],
        cols_w, ["l", "r", "r", "r", "r"], size=20, bold=True, fill=MUTED)
    y = rule(d, y + 34)
    per = P["annual_income"] / 24
    ytd = round(per * 13, 2)
    row(d, M, y + 8, ["Regular (salaried)", money(per, True), "86.67",
                      money(per, True), money(ytd, True)], cols_w,
        ["l", "r", "r", "r", "r"])
    y = rule(d, y + 46)
    row(d, M, y + 8, ["Gross Pay", "", "", money(per, True), money(ytd, True)],
        cols_w, ["l", "r", "r", "r", "r"], bold=True)
    y += 64

    y = bar(d, M, y, W - 2 * M, "DEDUCTIONS (CURRENT)")
    fed, ss, med = 980.90, round(per * 0.062, 2), round(per * 0.0145, 2)
    for name, val in [("Federal income tax", fed), ("Social Security", ss),
                      ("Medicare", med), ("State income tax (WA: none)", 0.0)]:
        row(d, M, y + 6, [name, money(val, True)], [900, W - 2 * M - 900],
            ["l", "r"])
        y = rule(d, y + 42, fill=(215, 215, 218))
    net = round(per - fed - ss - med, 2)
    y += 18

    # Sparse schedule: real earnings statements carry an "other income" grid that is mostly
    # blank. Only one row is filled, the rest are empty boxes, so OCR 4 must keep the single
    # value aligned to the right column across otherwise-empty rows.
    # Column widths sum to the printable width (W - 2*M = 1460) so the last column is not clipped.
    y = sparse_table(
        d, y, "OTHER INCOME AND ADJUSTMENTS (CURRENT PERIOD)",
        ["DESCRIPTION", "HOURS", "RATE", "CURRENT", "YEAR TO DATE"],
        [
            ["Overtime", "", "", "", ""],
            ["Bonus", "", "", "", ""],
            ["Commission", "", "", "", ""],
            ["Reimbursement (travel)", "", "", "$120.00", "$360.00"],
            ["Other", "", "", "", ""],
        ],
        [560, 170, 180, 270, 280],
    )

    y = bar(d, M, y, W - 2 * M, "COMPENSATION SUMMARY", bg=(70, 70, 76))
    half = (W - 2 * M) / 2
    cell(d, M, y, half - 12, 100, "Gross annual salary", money(P["annual_income"]), vsize=34)
    cell(d, M + half, y, half - 12, 100, "Federal tax withheld (current)", money(fed, True), vsize=34)
    y += 132

    d.rectangle([M, y, M + 470, y + 120], outline=INK, width=3)
    t(d, M + 20, y + 16, "NET PAY THIS PERIOD", size=22, bold=True, fill=MUTED)
    t(d, M + 20, y + 52, money(net, True), size=52, bold=True)

    t(d, M, H - M, "This is not a check. Retain for your records.", size=20, fill=MUTED)
    t(d, M, H - M, "Statement 2026-14", size=20, fill=MUTED, align="r", w=W - 2 * M)
    save_pdf(img, "paystub.pdf")


def sparse_table(d, y, title, headers, rows, widths):
    """A ruled grid where most cells are intentionally blank, the kind of mostly-empty
    'other income / adjustments' schedule that trips naive text parsers but that OCR 4 keeps
    aligned to the right columns. Empty string cells are drawn as real empty boxes."""
    y = bar(d, M, y, W - 2 * M, title)
    x0 = M
    # header
    row(d, x0, y, headers, widths, ["l"] + ["r"] * (len(headers) - 1),
        size=20, bold=True, fill=MUTED)
    y = rule(d, y + 34)
    hrow = 52
    for r in rows:
        cx = x0
        for j, (val, w) in enumerate(zip(r, widths)):
            d.rectangle([cx, y, cx + w - 8, y + hrow], outline=(205, 205, 210), width=2)
            if val != "":
                t(d, cx + 10, y + 12, str(val), size=24,
                  align="r" if j else "l", w=w - 28)
            cx += w
        y += hrow + 4
    return y + 8


def cells_grid(d, y, items, per_row=3, hcell=96, vsize=30):
    """Draw (label, value) items as bordered cells, wrapping every per_row. Returns new y."""
    colw = (W - 2 * M) / per_row
    for i in range(0, len(items), per_row):
        rowitems = items[i:i + per_row]
        for c, (lab, val) in enumerate(rowitems):
            cell(d, M + c * colw, y, colw - 12, hcell, lab, val, vsize=vsize)
        y += hcell + 8
    return y


def loan_app(out, *, name, address, loan, purpose, file_no,
             credit=None, employer=None, income=None, years=None,
             debts=None, housing=None, prop_value=None, occupancy="Primary Residence",
             signed=False):
    """Uniform Residential Loan Application (Form 1003). Omitted fields are left off the form,
    which is how the 'thin file' variant drives the intake loop to ask for what is missing."""
    img, d = page()
    t(d, M, M, "UNIFORM RESIDENTIAL LOAN APPLICATION", size=38, bold=True)
    t(d, M, M + 50, "Freddie Mac Form 65    Fannie Mae Form 1003", size=22, fill=MUTED)
    t(d, M, M + 80, f"Revised 09/2021    File {file_no}", size=20, fill=MUTED)
    y = rule(d, M + 118) + 16

    y = bar(d, M, y, W - 2 * M, "SECTION 1   BORROWER INFORMATION")
    s1 = [("Borrower name", name)]
    if credit is not None:
        s1.append(("Credit score", str(credit)))
    if years is not None:
        s1.append(("Years of employment", str(years)))
    if employer is not None:
        s1.append(("Current employer", employer))
    if income is not None:
        s1.append(("Gross annual income", money(income)))
    y = cells_grid(d, y, s1, per_row=3, vsize=28) + 6

    if debts is not None or housing is not None:
        y = bar(d, M, y, W - 2 * M, "SECTION 2   FINANCIAL INFORMATION: LIABILITIES")
        s2 = []
        if debts is not None:
            s2.append(("Total monthly debt payments", money(debts)))
        if housing is not None:
            s2.append(("Current monthly housing expense", money(housing)))
        y = cells_grid(d, y, s2, per_row=2, vsize=28) + 6

    y = bar(d, M, y, W - 2 * M, "SECTION 4   LOAN AND PROPERTY INFORMATION")
    y = cells_grid(d, y, [("Subject property address", address)], per_row=1, vsize=28)
    rest = []
    if prop_value is not None:
        rest.append(("Estimated property value", money(prop_value)))
    rest.append(("Loan amount requested", money(loan)))
    rest.append(("Loan purpose", purpose))
    rest.append(("Occupancy", occupancy))
    y = cells_grid(d, y, rest, per_row=2, vsize=28)

    y += 34
    t(d, M, y, "BORROWER SIGNATURE", size=20, bold=True, fill=MUTED)
    t(d, M + 560, y, "DATE", size=20, bold=True, fill=MUTED)
    d.line([M, y + 74, M + 470, y + 74], fill=(120, 120, 120), width=2)
    d.line([M + 560, y + 74, M + 940, y + 74], fill=(120, 120, 120), width=2)

    if signed:
        # OCR 4 showcase: a handwritten signature over the line, a handwritten date, and a
        # borrower's note in the open space below. Synthetic (font-rendered), so no real
        # signature is used.
        hand(img, M + 20, y + 8, name, size=64, sig=True, fill=PEN_BLUE, jitter=2.4, rot=-2.0)
        hand(img, M + 585, y + 20, "08/14/2026", size=40, fill=PEN_BLUE, jitter=1.6, rot=-1.0)
        hand(img, M + 20, y + 150,
             "Note: please use my Northwind salary for income,",
             size=30, fill=PEN_BLACK, jitter=1.8, rot=-1.2)
        hand(img, M + 20, y + 200,
             "bonus not included.  - D.L.",
             size=30, fill=PEN_BLACK, jitter=1.8, rot=-1.2)

    t(d, M, H - M, "Uniform Residential Loan Application. Borrower.", size=20, fill=MUTED)
    t(d, M, H - M, "Page 1 of 9", size=20, fill=MUTED, align="r", w=W - 2 * M)
    save_pdf(img, out)


def build_w2():
    img, d = page()
    t(d, M, M, "Form W-2   Wage and Tax Statement", size=40, bold=True)
    t(d, M, M + 52, "Tax year 2025    Department of the Treasury, Internal Revenue Service",
      size=20, fill=MUTED)
    y = rule(d, M + 96) + 16

    y = bar(d, M, y, W - 2 * M, "EMPLOYEE")
    y = cells_grid(d, y, [("Employee name", P["name"]), ("Social Security number", P["ssn"]),
                          ("Employee address", P["addr"])], per_row=3, vsize=23)
    y = bar(d, M, y, W - 2 * M, "EMPLOYER")
    y = cells_grid(d, y, [("Employer name", P["employer"]), ("Employer EIN", P["ein"]),
                          ("Employer address", P["employer_addr"])], per_row=3, vsize=22)

    y = bar(d, M, y, W - 2 * M, "WAGES AND TAXES")
    wages, fed, ss, med = 145000.00, 23541.60, 8990.00, 2102.50
    y = cells_grid(d, y, [
        ("Box 1  Wages, tips, other comp", money(wages, True)),
        ("Box 2  Federal income tax withheld", money(fed, True)),
        ("Box 3  Social Security wages", money(wages, True)),
        ("Box 4  Social Security tax withheld", money(ss, True)),
        ("Box 5  Medicare wages and tips", money(wages, True)),
        ("Box 6  Medicare tax withheld", money(med, True)),
    ], per_row=3, vsize=26)
    y = cells_grid(d, y, [("Box 15  State", "WA"), ("Box 16  State wages", "$0.00"),
                          ("Box 17  State income tax", "$0.00")], per_row=3, vsize=26)

    t(d, M, H - M, "Copy B. To be filed with the employee's federal tax return.", size=20, fill=MUTED)
    save_pdf(img, "w2.pdf")


def build_bank():
    img, d = page()
    t(d, M, M, "Cascade Mutual Bank", size=44, bold=True)
    t(d, M, M + 58, "Account Statement", size=26, fill=MUTED)
    t(d, M, M, "Member FDIC", size=20, fill=MUTED, align="r", w=W - 2 * M)
    y = rule(d, M + 104) + 16

    y = cells_grid(d, y, [("Account holder", P["name"]), ("Account", "Checking ****4821"),
                          ("Statement period", "2026-06-16 to 2026-07-15")], per_row=3, vsize=24)
    y = cells_grid(d, y, [("Mailing address", P["addr"])], per_row=1, vsize=28)

    txns = [
        ("2026-06-18", "Payroll  Northwind Analytics LLC", 4598.59),
        ("2026-06-20", "Grocery Market", -142.30),
        ("2026-06-27", "Electric Utility", -118.64),
        ("2026-07-01", "Mortgage payment", -1850.00),
        ("2026-07-03", "Payroll  Northwind Analytics LLC", 4598.59),
        ("2026-07-08", "Auto insurance", -186.40),
        ("2026-07-12", "Restaurant", -74.20),
        ("2026-07-14", "Credit card payment", -540.99),
    ]
    begin = 22165.87
    deposits = sum(a for _, _, a in txns if a > 0)
    withdrawals = sum(-a for _, _, a in txns if a < 0)
    end = round(begin + deposits - withdrawals, 2)

    y = bar(d, M, y, W - 2 * M, "SUMMARY")
    y = cells_grid(d, y, [("Beginning balance", money(begin, True)),
                          ("Total deposits", money(deposits, True)),
                          ("Total withdrawals", money(withdrawals, True))], per_row=3, vsize=28)
    y = cells_grid(d, y, [("Ending balance", money(end, True))], per_row=3, vsize=32)

    y = bar(d, M, y, W - 2 * M, "TRANSACTIONS")
    cw = [230, 700, 240, 290]
    row(d, M, y, ["DATE", "DESCRIPTION", "AMOUNT", "BALANCE"], cw, ["l", "l", "r", "r"],
        size=20, bold=True, fill=MUTED)
    y = rule(d, y + 34)
    bal = begin
    for dt, desc, amt in txns:
        bal = round(bal + amt, 2)
        sign = "+" if amt >= 0 else "-"
        row(d, M, y + 6, [dt, desc, sign + money(abs(amt), True), money(bal, True)],
            cw, ["l", "l", "r", "r"])
        y = rule(d, y + 42, fill=(220, 220, 222))

    t(d, M, H - M, "Cascade Mutual Bank, Seattle, WA. Member FDIC.", size=20, fill=MUTED)
    save_pdf(img, "bank_statement.pdf")


def main():
    # Primary package: one borrower across every document.
    loan_app("loan_application_1003.pdf", name=P["name"], address=P["addr"],
             loan=P["loan_amount"], purpose=P["purpose"], file_no="2026-04817",
             credit=P["credit"], employer=P["employer"], income=P["annual_income"],
             years=P["years"], debts=P["monthly_debts"], housing=P["monthly_housing"],
             prop_value=P["property_value"], signed=True)
    build_paystub()
    build_w2()
    build_bank()

    # Thin file: only the essentials, so the intake loop asks for the missing items.
    loan_app("loan_application_thin.pdf", name=P["name"], address=P["addr"],
             loan=P["loan_amount"], purpose=P["purpose"], file_no="2026-05120")

    # Clean file: comfortably inside the auto-approval bounds (loan-to-value 60%, credit 780).
    loan_app("mortgage_clean.pdf", name=P["name"], address=P["addr"],
             loan=300000, purpose=P["purpose"], file_no="2026-05450",
             credit=780, employer=P["employer"], income=160000, years=8,
             debts=300, housing=1700, prop_value=500000)

    # Borderline file: within policy but outside the auto bounds (loan-to-value 74%, credit 690)
    # so it routes to a human.
    loan_app("mortgage_borderline.pdf", name=P["name"], address=P["addr"],
             loan=370000, purpose=P["purpose"], file_no="2026-05502",
             credit=690, employer=P["employer"], income=145000, years=6,
             debts=900, housing=2100, prop_value=500000)


if __name__ == "__main__":
    main()
