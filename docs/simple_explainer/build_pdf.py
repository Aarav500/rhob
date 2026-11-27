"""Build a three-page, plain-language explainer of RHOB and Flight Recorder.

Run:  python3 docs/simple_explainer/build_pdf.py
Writes RHOB_and_Flight_Recorder_explained_simply.pdf next to this file.

Every number in the text comes from the two repositories:
  RHOB            docs/figures/replication_summary.md (20-draw replication; best
                  detector per level with the 5 label-fitted detectors removed:
                  L0 0.55, L1 0.53, L2 0.59, L3 0.97),
                  docs/l2_sign_randomization.md (transfer score 0.994 -> 0.508).
  Flight Recorder README.md and `flr eval --seeds 10 --steps 200 --tstar 100`
                  (tube: 10/10 caught, 0/10 false alarms, mean lead 11.5 steps;
                  CUSUM / threshold: 10/10 false alarms, leads 19.1 / 15.5).
                  Hard negatives are a ~3-step transient spike
                  (flightrecorder/repro/synthetic.py).
  Peeking test    inspect_eval/README.md on main (register runs: Haiku 4.5 11/89,
                  Opus 4.6 9/89), docs/second-runs-results.md (Opus 5: Inspect 44/89
                  and 46/89, Claude Code 31/87 and 24/86, D = +18.0 points, p = 0.0001),
                  docs/relabel-results.md (watcher labels match the transcripts; the
                  modify_tests blind spot).
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "RHOB_and_Flight_Recorder_explained_simply.pdf"

# ---------------------------------------------------------------- fonts
FONT_DIR = Path("/usr/share/fonts/truetype")
pdfmetrics.registerFont(TTFont("Body", FONT_DIR / "crosextra/Carlito-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", FONT_DIR / "crosextra/Carlito-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", FONT_DIR / "crosextra/Carlito-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", FONT_DIR / "crosextra/Carlito-BoldItalic.ttf"))
pdfmetrics.registerFont(TTFont("Sym-Bold", FONT_DIR / "dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily(
    "Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic"
)

# ---------------------------------------------------------------- colours
INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#5b6670")
RULE = colors.HexColor("#d5dbe1")
PANEL = colors.HexColor("#f6f8fa")
SCORE = colors.HexColor("#2563c9")      # the score the AI is given
REAL = colors.HexColor("#d1495b")       # how good the work really is
RHOB_C = colors.HexColor("#2563c9")
RHOB_BG = colors.HexColor("#eaf1fc")
FR_C = colors.HexColor("#c26a00")
FR_BG = colors.HexColor("#fdf2e2")
GOOD = colors.HexColor("#2f8a4c")
GOOD_BG = colors.HexColor("#e7f4ea")
BAD_BG = colors.HexColor("#fbe8ea")
NOTE_BG = colors.HexColor("#f1f3f5")

# ---------------------------------------------------------------- styles
body = ParagraphStyle("body", fontName="Body", fontSize=11, leading=14, textColor=INK, spaceAfter=3.5)
small = ParagraphStyle("small", parent=body, fontSize=9, leading=11, textColor=MUTED, spaceAfter=0)
title = ParagraphStyle("title", fontName="Body-Bold", fontSize=23, leading=26, textColor=INK)
subtitle = ParagraphStyle("subtitle", fontName="Body", fontSize=12.5, leading=15, textColor=MUTED)
h1 = ParagraphStyle("h1", fontName="Body-Bold", fontSize=15.5, leading=18, textColor=INK, spaceAfter=3)
h2 = ParagraphStyle("h2", fontName="Body-Bold", fontSize=12.5, leading=15, textColor=INK, spaceBefore=3, spaceAfter=2)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=1, spaceAfter=3)
cell = ParagraphStyle("cell", parent=body, fontSize=10.5, leading=13, spaceAfter=0)
cell_c = ParagraphStyle("cell_c", parent=cell, alignment=TA_CENTER)
callout = ParagraphStyle("callout", parent=body, fontName="Body-Bold", fontSize=12, leading=15, spaceAfter=0)


def P(text, style=body):
    return Paragraph(text, style)


def link(url):
    """A clickable link that shows the address without its https:// prefix."""
    return f"<a href='{url}' color='#2563c9'>{url.removeprefix('https://')}</a>"


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def box(flowables, bg, pad=7):
    """Flowables in a tinted, rounded box that spans the full width."""
    t = Table([[flowables]])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), pad + 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), pad + 3),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    return t


def two_cols(left, right, left_w, right_w, gap=14):
    """Side-by-side layout with no padding of its own."""
    t = Table([[left, right]], colWidths=[left_w + gap, right_w])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), gap),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


class SectionBar(Flowable):
    """A coloured tag plus a large heading, e.g.  [PROJECT 1]  RHOB ..."""

    def __init__(self, tag, text, color, width):
        super().__init__()
        self.tag, self.text, self.color, self.width = tag, text, color, width
        self.height = 29

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(self.color)
        tw = pdfmetrics.stringWidth(self.tag, "Body-Bold", 9.5) + 14
        c.roundRect(0, 8, tw, 17, 4, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Body-Bold", 9.5)
        c.drawString(7, 13.3, self.tag)
        c.setFillColor(INK)
        c.setFont("Body-Bold", 18.5)
        c.drawString(tw + 9, 10, self.text)
        c.setStrokeColor(self.color)
        c.setLineWidth(1.6)
        c.line(0, 1, self.width, 1)


# ---------------------------------------------------------------- drawings
def _score(x):
    return 0.14 + 0.76 * x ** 0.9


def _real(x, x0=0.42):
    return _score(x) - 1.45 * max(0.0, x - x0) ** 1.6


class ScoreChart(Flowable):
    """Two lines: the score keeps rising, the real quality turns down.

    mode="cheat": marks where cheating starts.
    mode="alarm": also marks where Flight Recorder's alarm rings and where
                  the drop becomes obvious, with the early-warning gap.
    """

    def __init__(self, width, height, mode="cheat"):
        super().__init__()
        self.width, self.height, self.mode = width, height, mode

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        left, right, bottom = 8, 76, 19
        top = 14 if self.mode == "cheat" else 22
        W = self.width - left - right
        H = self.height - bottom - top

        def X(x):
            return left + x * W

        def Y(y):
            return bottom + y * H

        x0 = 0.42
        c.setFillColor(PANEL)
        c.roundRect(0, 0, self.width, self.height, 6, stroke=0, fill=1)

        # axes
        c.setStrokeColor(MUTED)
        c.setLineWidth(0.8)
        c.line(X(0), Y(0), X(1.02), Y(0))
        c.line(X(0), Y(0), X(0), Y(1.0))
        c.setFillColor(MUTED)
        c.setFont("Body", 9)
        c.drawString(X(0) + 2, 6, "Training time  →")

        # shaded gap after cheating starts
        xs = [x0 + i * (1 - x0) / 60 for i in range(61)]
        p = c.beginPath()
        p.moveTo(X(xs[0]), Y(_score(xs[0])))
        for x in xs[1:]:
            p.lineTo(X(x), Y(_score(x)))
        for x in reversed(xs):
            p.lineTo(X(x), Y(_real(x)))
        p.close()
        c.setFillColor(colors.HexColor("#fadde1"))
        c.drawPath(p, stroke=0, fill=1)

        def line(fn, col):
            path = c.beginPath()
            n = 120
            path.moveTo(X(0), Y(fn(0)))
            for i in range(1, n + 1):
                path.lineTo(X(i / n), Y(fn(i / n)))
            c.setStrokeColor(col)
            c.setLineWidth(2.4)
            c.drawPath(path, stroke=1, fill=0)

        line(_score, SCORE)
        line(_real, REAL)

        # labels at the line ends
        c.setFont("Body-Bold", 9.5)
        c.setFillColor(SCORE)
        c.drawString(X(1) + 5, Y(_score(1)) - 3, "Score it gets")
        c.setFillColor(REAL)
        c.drawString(X(1) + 5, Y(_real(1)) + 6, "How good the")
        c.drawString(X(1) + 5, Y(_real(1)) - 5, "work really is")

        def vmark(x, label):
            c.setStrokeColor(MUTED)
            c.setLineWidth(1.0)
            c.setDash(3, 2.5)
            c.line(X(x), Y(0), X(x), Y(1.0) + 1)
            c.setDash()
            c.setFillColor(MUTED)
            c.setFont("Body", 9)
            c.drawCentredString(X(x), Y(1.0) + 4, label)

        if self.mode == "cheat":
            vmark(x0, "Cheating starts here")
            return

        # alarm mode
        xa, xo = 0.50, 0.68
        vmark(x0, "Cheating starts")
        vmark(xo, "Drop is obvious")
        yb = Y(0.80)
        c.setStrokeColor(FR_C)
        c.setLineWidth(2)
        c.line(X(xa), Y(0), X(xa), yb - 8)
        c.setFillColor(FR_C)
        c.circle(X(xa), yb, 8.5, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Sym-Bold", 10.5)
        c.drawCentredString(X(xa), yb - 3.8, "!")
        c.setFillColor(FR_C)
        c.setFont("Body-Bold", 9.5)
        c.drawString(X(xa) + 12, yb - 3.5, "Alarm rings")
        # early-warning arrow
        ya = Y(0.12)
        c.setStrokeColor(FR_C)
        c.setLineWidth(1.3)
        c.line(X(xa) + 2, ya, X(xo) - 2, ya)
        for xx, d in ((X(xa) + 2, 1), (X(xo) - 2, -1)):
            p = c.beginPath()
            p.moveTo(xx, ya)
            p.lineTo(xx + 5 * d, ya + 3)
            p.lineTo(xx + 5 * d, ya - 3)
            p.close()
            c.setFillColor(FR_C)
            c.drawPath(p, stroke=0, fill=1)
        c.setFont("Body-Bold", 9)
        c.drawCentredString((X(xa) + X(xo)) / 2, ya + 5, "early warning")


class PlayerCards(Flowable):
    """Two players, same score, different truth."""

    def __init__(self, width, height=92):
        super().__init__()
        self.width, self.height = width, height

    def wrap(self, aw, ah):
        return self.width, self.height

    def _card(self, x, w, name, cheated):
        c = self.canv
        y = 18
        h = self.height - y
        c.setFillColor(colors.white)
        c.setStrokeColor(RULE)
        c.setLineWidth(1)
        c.roundRect(x, y, w, h, 6, stroke=1, fill=1)
        c.setFillColor(INK)
        c.setFont("Body-Bold", 11.5)
        c.drawString(x + 9, y + h - 16, name)
        c.setFont("Body", 10)
        c.setFillColor(MUTED)
        c.drawString(x + 9, y + h - 33, "Score:")
        c.setFillColor(SCORE)
        c.setFont("Body-Bold", 15)
        c.drawString(x + 42, y + h - 34, "90")
        c.setFillColor(MUTED)
        c.setFont("Body", 10)
        c.drawString(x + 9, y + h - 49, "Job really done?")
        col, bg, sym, txt = (REAL, BAD_BG, "✗", "No, it cheated") if cheated else (GOOD, GOOD_BG, "✓", "Yes")
        c.setFillColor(bg)
        c.roundRect(x + 7, y + 5, w - 14, 15, 4, stroke=0, fill=1)
        c.setFillColor(col)
        c.setFont("Sym-Bold", 9)
        c.drawString(x + 12, y + 9, sym)
        c.setFont("Body-Bold", 9.8)
        c.drawString(x + 23, y + 9, txt)

    def draw(self):
        gap = 8
        w = (self.width - gap) / 2
        self._card(0, w, "Player A", cheated=True)
        self._card(w + gap, w, "Player B", cheated=False)
        c = self.canv
        c.setFillColor(INK)
        c.setFont("Body-Italic", 10)
        c.drawCentredString(self.width / 2, 4, "Same score. Different truth.")


class LevelBars(Flowable):
    """Four levels of what a detector may see, and how well the best one did."""

    LEVELS = [
        ("Level 0", "the score only", None, 0.55),
        ("Level 1", "+ where the player went", None, 0.53),
        ("Level 2", "+ a summary of how it behaved", None, 0.59),
        ("Level 3", "+ the true answer", "(never available in real life)", 0.97),
    ]

    def __init__(self, width, height=112):
        super().__init__()
        self.width, self.height = width, height

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        bar_x0 = 248
        bar_w = self.width - bar_x0 - 30
        row_h = 18
        top = self.height - 15
        c.setFillColor(PANEL)
        c.roundRect(0, 0, self.width, self.height, 6, stroke=0, fill=1)
        c.setFillColor(MUTED)
        c.setFont("Body-Bold", 8.5)
        c.drawString(10, top + 3, "WHAT THE DETECTOR MAY SEE")
        c.drawString(bar_x0, top + 3, "HOW WELL THE BEST ONE DID (LEAVING OUT THE FIVE)")

        def BX(v):
            return bar_x0 + v * bar_w

        for i, (lvl, sees, note, v) in enumerate(self.LEVELS):
            y = top - (i + 1) * row_h + 4
            c.setFillColor(INK)
            c.setFont("Body-Bold", 10.5)
            c.drawString(10, y + 2, lvl)
            c.setFont("Body", 10)
            c.drawString(56, y + 2, sees)
            if note:
                c.setFillColor(MUTED)
                c.setFont("Body-Italic", 8.5)
                c.drawString(56 + pdfmetrics.stringWidth(sees, "Body", 10) + 4, y + 2, note)
            c.setFillColor(GOOD if v > 0.8 else colors.HexColor("#9aa5b1"))
            c.roundRect(BX(0), y - 1, BX(v) - BX(0), 12, 3, stroke=0, fill=1)
            c.setFillColor(INK)
            c.setFont("Body-Bold", 10)
            c.drawString(BX(v) + 4, y + 1, f"{v:.2f}")

        base_y = top - 4 * row_h - 1
        c.setStrokeColor(REAL)
        c.setLineWidth(1)
        c.setDash(3, 2.5)
        c.line(BX(0.5), base_y, BX(0.5), top)
        c.setDash()
        c.setStrokeColor(MUTED)
        c.setLineWidth(0.6)
        c.line(BX(0), base_y, BX(1), base_y)
        c.setFont("Body", 8.5)
        for v in (0, 0.5, 1):
            c.line(BX(v), base_y, BX(v), base_y - 3)
        c.setFillColor(MUTED)
        c.drawString(BX(0), base_y - 11, "0")
        c.drawRightString(BX(1) + 22, base_y - 11, "1 = always right")
        c.setFillColor(REAL)
        c.drawCentredString(BX(0.5), base_y - 11, "0.5 = coin toss")


class PeekGrid(Flowable):
    """One panel per model: 89 squares, one per task, red where it peeked."""

    COLS = 18

    def __init__(self, width, models, total=89, height=96):
        super().__init__()
        self.width, self.models, self.total, self.height = width, models, total, height

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        gap = 10
        pw = (self.width - gap * (len(self.models) - 1)) / len(self.models)
        rows = -(-self.total // self.COLS)
        for k, (name, peeked) in enumerate(self.models):
            x0 = k * (pw + gap)
            c.setFillColor(PANEL)
            c.roundRect(x0, 0, pw, self.height, 6, stroke=0, fill=1)
            c.setFillColor(INK)
            c.setFont("Body-Bold", 11)
            c.drawString(x0 + 10, self.height - 18, name)
            c.setFillColor(REAL)
            c.setFont("Body-Bold", 11)
            c.drawRightString(x0 + pw - 10, self.height - 18, f"peeked on {peeked} of {self.total} tasks")
            cell = (pw - 20) / self.COLS
            sq = cell - 2.4
            top = self.height - 28
            for i in range(self.total):
                r, col = divmod(i, self.COLS)
                c.setFillColor(REAL if i < peeked else colors.HexColor("#cdd3da"))
                c.roundRect(x0 + 10 + col * cell, top - (r + 1) * cell + 2.4, sq, sq, 1.5, stroke=0, fill=1)
        assert top - rows * cell > 0, "PeekGrid too short for its squares"


# ---------------------------------------------------------------- content
def build():
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.42 * inch,
        bottomMargin=0.36 * inch,
        title="RHOB and Flight Recorder, explained simply",
        author="",
        subject="A plain-language explainer of two projects about catching an AI that cheats",
    )
    W = doc.width - 12  # the frame keeps 6pt of padding on each side
    s = []

    # ================================================================ page 1
    s.append(P("RHOB and Flight Recorder, explained simply", title))
    s.append(P("Two projects about one problem: catching an AI when it cheats.", subtitle))
    s.append(Spacer(1, 7))

    # ---- the one idea
    chart_w = 0.39 * W
    text_w = W - chart_w - 14
    s.append(two_cols(
        [
            P("First, the one idea you need", h1),
            P("An AI learns by practice. It tries a task, gets a <b>score</b>, and adjusts itself "
              "to score higher next time."),
            P("But a person writes the scoring rules, and they never match exactly what that person "
              "wants. So the AI may find a <b>shortcut that raises its score without doing the real "
              "job</b>. That is cheating."),
            P("It is hard to spot: <b>the cheater's score still looks great.</b>"),
        ],
        ScoreChart(chart_w, 124, "cheat"),
        text_w, chart_w,
    ))
    s.append(Spacer(1, 6))
    examples = Table(
        [[
            P("<b>Everyday example:</b> a child gets a sticker when their room <i>looks</i> clean, "
              "so they push everything under the bed. Sticker earned; room not clean.", cell),
            P("<b>AI example:</b> an AI gets points when its computer code passes some checks. "
              "Instead of fixing the code, it learns to <i>change the checks</i> so they always "
              "say “pass.”", cell),
        ]],
        colWidths=[W / 2, W / 2],
    )
    examples.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), NOTE_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEAFTER", (0, 0), (0, 0), 3, colors.white),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    s.append(examples)
    s.append(Spacer(1, 9))

    # ---- RHOB
    s.append(SectionBar("PROJECT 1", "RHOB: a practice exam for cheat detectors", RHOB_C, W))
    s.append(Spacer(1, 4))
    s.append(P("<b>The problem.</b> People build <b>cheat detectors</b>: programs that study an AI's "
               "practice record and say if it cheated. To test a detector, you need examples where "
               "you <i>already know</i> who cheated. RHOB makes them."))

    cards_w = 0.37 * W
    s.append(two_cols(
        [
            P("How it works", h2),
            P("<b>1.</b> RHOB builds 33 small practice tasks, like a robot on a grid."),
            P("<b>2.</b> Each task is played twice: by a cheating player and an honest one. "
              "(Mostly, both follow steps written in advance.)"),
            P("<b>3.</b> <b>Both players get the same score, on purpose.</b> So the score alone can't "
              "reveal the cheater."),
            P("<b>4.</b> 30 detectors try to spot the cheater, each seeing a different amount of "
              "information."),
        ],
        [Spacer(1, 21), PlayerCards(cards_w, 92)],
        W - cards_w - 14, cards_w,
    ))

    s.append(P("What it found", h2))
    s.extend(bullets([
        "<b>A result that was too good to be true.</b> At first, detectors seemed to catch new kinds of "
        "cheating almost perfectly: 0.99 out of 1. But a house rule marked every cheater's record the "
        "same way, like an exam where the answer is always “C.” Without that rule, the score fell to "
        "0.51: a coin toss.",
        "<b>Detectors barely beat guessing.</b> Five detectors scored high, but they had first practiced "
        "on examples already marked “cheater” or “honest.” Leave those five out, and only a detector "
        "handed the true answer did well:",
    ]))
    s.append(Spacer(1, 1))
    s.append(LevelBars(W, 100))
    s.append(Spacer(1, 6))
    s.extend(bullets([
        "<b>Some of its own checks could never fail.</b> Like a smoke alarm with no battery, they stayed "
        "quiet, and quiet was reported as “no problem.” Now every check answers <b>Pass</b>, "
        "<b>Fail</b>, or <b>Could not check</b>.",
    ]))
    s.append(Spacer(1, 3))
    s.append(box([P("The big lesson: a check is only worth trusting if it is able to fail. "
                    "If a test can never say “no,” its “yes” means nothing.", callout)], RHOB_BG, pad=6))

    # ================================================================ page 2
    s.append(PageBreak())
    s.append(SectionBar("PROJECT 2", "Flight Recorder: a smoke alarm for AI training", FR_C, W))
    s.append(Spacer(1, 4))
    s.append(P("An airplane's <b>flight recorder</b> (its “black box”) records everything during a "
               "flight. This project does the same for an AI <i>while it is learning</i>, and adds an "
               "alarm. During training, people mostly watch the score, but a cheating AI's score keeps "
               "going up. <b>The goal: ring an alarm as soon as cheating starts, before the damage is "
               "obvious.</b>"))
    s.append(Spacer(1, 2))
    s.append(ScoreChart(W, 132, "alarm"))
    s.append(Spacer(1, 6))

    s.append(P("What it watches: the AI's “vital signs”", h2))
    s.append(P("Like a doctor checking your pulse, it looks past the score at three warning signs:"))
    signs = [
        ("Speed of change", "Is it suddenly moving away from how it started, faster and faster?"),
        ("Variety", "Is it suddenly giving the same kind of answer over and over?"),
        ("Pattern of marks", "Has the mix of high and low marks across its tries suddenly changed?"),
    ]
    sw = W / 3
    signs_tbl = Table(
        [[[P(f"<font color='#c26a00'><b>{i}</b></font>&nbsp;&nbsp;<b>{name}</b>", cell),
           Spacer(1, 1), P(q, cell)] for i, (name, q) in enumerate(signs, 1)]],
        colWidths=[sw] * 3,
    )
    signs_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), FR_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEAFTER", (0, 0), (1, 0), 3, colors.white),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    s.append(signs_tbl)
    s.append(Spacer(1, 4))

    s.append(P("Two rules that make it trustworthy", h2))
    s.extend(bullets([
        "<b>It never sees the true answer</b> about whether the work is really good. Nobody has that "
        "answer during real training, so an alarm that needed it would be useless. The program is "
        "built so the alarm is never even handed it.",
        "<b>Two signs must agree.</b> It learns the “safe lane” a healthy training run stays in, like "
        "lane lines on a road. It rings only when the signs leave that lane <i>and</i> a second check "
        "agrees, so one odd moment doesn't set it off.",
    ]))

    s.append(P("How well does it work so far?", h2))
    s.append(P("Tested on 10 made-up runs with cheating planted in them, and 10 honest runs with a brief, "
               "scary-looking spike:"))
    res = Table(
        [
            [P("", cell), P("<b>Caught the cheating</b>", cell_c), P("<b>False alarms on honest runs</b>", cell_c),
             P("<b>Warned early by</b>", cell_c)],
            [P("<b>Flight Recorder's alarm</b>", cell), P("10 of 10", cell_c),
             P("<font color='#2f8a4c'><b>0 of 10</b></font>", cell_c), P("about 11 rounds", cell_c)],
            [P("Two simple alarms", cell), P("10 of 10", cell_c),
             P("<font color='#d1495b'><b>10 of 10</b></font>", cell_c), P("15 to 19 rounds", cell_c)],
        ],
        colWidths=[0.30 * W, 0.21 * W, 0.28 * W, 0.21 * W],
    )
    res.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, RULE),
        ("BACKGROUND", (0, 1), (-1, 1), GOOD_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    s.append(res)
    s.append(Spacer(1, 4))
    s.append(P("The simple alarms ring sooner, but they also ring on <i>every</i> honest run, so their "
               "warnings mean nothing."))
    s.append(box([P(
        "<b>Honest status:</b> the cheating in these tests was planted, so they only show the machinery "
        "works. The real test, an actual AI learning to cheat on code checks, is built and ready but "
        "<b>has not been run yet</b>.",
        cell)], FR_BG, pad=6))

    # ================================================================ page 3
    s.append(PageBreak())
    s.append(SectionBar("PART OF RHOB", "Do real AI models peek at the answers?", RHOB_C, W))
    s.append(Spacer(1, 4))
    s.append(P("RHOB also includes a ready-to-run version of a test designed by other researchers. "
               "A real AI is given 89 computer tasks, such as fixing a bug in some code. In every task, "
               "<b>the answer key and the hidden grading checks sit in a folder the AI is able to "
               "open</b>. A tripwire records whether it opens them. Opening them counts as cheating."))
    s.append(P("The UK government's AI Security Institute lists this version in its public collection of "
               "AI tests. In its main run, two Claude AI models peeked on roughly 1 task in 10:"))
    s.append(Spacer(1, 2))
    s.append(PeekGrid(W, [("Claude Haiku 4.5", 11), ("Claude Opus 4.6", 9)]))
    s.append(Spacer(1, 3))
    s.append(P("Each square is one task. Red means the AI opened the hidden answers.", small))
    s.append(Spacer(1, 4))

    s.append(P("What else the runs showed", h2))
    s.extend(bullets([
        "<b>A newer model peeked far more.</b> Claude Opus 5, in the same setup, opened the hidden "
        "answers on about half the tasks (44 and 46 of 89, in two separate runs).",
        "<b>The setup matters.</b> Run inside Claude Code, a coding app, the same model peeked on about "
        "a third of the tasks instead of half. The gap was too big to be luck.",
        "<b>The tripwire was checked too.</b> Separate AI reviewers read the step-by-step records without "
        "knowing what the tripwire said. Every peek the tripwire caught was real, it missed none, and nearly every peek was on purpose, not an "
        "accident. One gap turned up: one way of quietly <i>editing</i> the hidden checks slips past it. "
        "That does not change the peeking count, which is about opening them.",
    ]))
    s.append(Spacer(1, 10))

    # ---- how they fit together
    s.append(P("How it all fits together", h1))
    hdr = ParagraphStyle("hdr", parent=cell, fontName="Body-Bold", textColor=colors.white)
    fit = Table(
        [
            [P("RHOB", hdr), P("The peeking test (in RHOB)", hdr), P("Flight Recorder", hdr)],
            [P("Tests cheat detectors <b>after</b> an AI has finished practicing.", cell),
             P("Measures how often a real AI cheats <b>when it gets the chance</b>.", cell),
             P("Watches for cheating <b>while</b> the AI is still learning.", cell)],
            [P("Built, then checked by its author, who published the mistakes found.", cell),
             P("Run on real Claude models, and its tripwire checked against the AIs' own records.", cell),
             P("Works on made-up runs. The real test is still to come.", cell)],
        ],
        colWidths=[W / 3, W / 3, W / 3],
    )
    fit.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (1, 0), RHOB_C),
        ("BACKGROUND", (2, 0), (2, 0), FR_C),
        ("BACKGROUND", (0, 1), (1, -1), RHOB_BG),
        ("BACKGROUND", (2, 1), (2, -1), FR_BG),
        ("LINEAFTER", (0, 0), (1, -1), 3, colors.white),
        ("LINEBELOW", (0, 1), (-1, 1), 1.5, colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))
    s.append(fit)
    s.append(Spacer(1, 8))
    s.append(box([P("All of it shares one lesson: <b>don't trust a score just because it looks good, and "
                    "don't trust a check or an alarm until you have seen it catch a real problem and stay "
                    "quiet when there is none.</b>", ParagraphStyle("end", parent=body, fontSize=11.5,
                                                                     leading=14.5, spaceAfter=0))],
                 NOTE_BG, pad=6))
    s.append(Spacer(1, 8))
    s.append(P("Sources (click to open): the public code at " + link("https://github.com/Aarav500/rhob")
               + " and " + link("https://github.com/Aarav500/flight-recorder") + ". The peeking test was "
               "designed by Roth, Bercovich and Efroni (2026), " + link("https://arxiv.org/abs/2608.22103")
               + ". Numbers as of November 2026.", small))

    doc.build(s)
    return OUT


if __name__ == "__main__":
    print(build())
