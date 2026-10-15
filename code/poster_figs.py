"""포스터(poster.html)에 넣을 그래프 3개와 QR 코드 2개를 SVG로 만들어 끼워 넣는 스크립트.

저장소 맨 위 폴더에서 실행한다: python3 code/poster_figs.py
check.py와 같은 전처리·모델을 사용하므로 그래프의 값은 check.py 수치와 같다.
포스터를 만들 때만 사용하는 도구라 matplotlib·segno가 필요하다(앱 실행에는 필요 없음).
  pip install matplotlib segno

poster.html의 <!-- FIG:이름 --> 과 <!-- /FIG:이름 --> 사이를 새 SVG로 바꾼다.
  FIG:hist     2. 데이터  분포 그래프
  FIG:bars     4. 결과    평가지표 비교 막대 그래프(k별 실루엣 + 방법별 실루엣, 선택 모델만 강조 색)
  FIG:result   4. 결과    결과 그래프(군집: 홈런율-도루율 산점도, 색 = 유형)
  FIG:qr_repo  제목띠     저장소 QR
  FIG:qr_app   발치       앱 QR

[다른 샘플에서 이 파일을 쓸 때]
  · 공통 부분(설정·svg_of·qr_svg·inject)은 그대로 둔다.
  · "이 샘플의 데이터와 그래프" 부분만 바꾼다. 그래프 이름(FIG:이름) 다섯 개는 그대로 사용한다.
  · 그래프 폭은 COL_W(단 폭)로 고정한다. 폭을 맞춰야 그래프 글자 20pt가 포스터에서도 20pt로 보인다.
"""
import io
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 없는 글꼴 경고 끄기
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
import segno

# ───────────── 공통 설정(다섯 샘플이 같다) ─────────────
ROOT = Path(__file__).resolve().parent.parent
POSTER = ROOT / "poster.html"
ACCENT = "#1f3d7a"      # poster.html의 --accent와 같은 남색
GRAY = "#9aa0aa"        # 선택하지 않은 모델·나머지 점
DARK = "#4a4f57"        # 기준선
# 단 폭(인치) = (594mm - 좌우 여백 2×26mm - 단 사이 22mm) / 2 = 260mm. poster.html의 --pad-x·--gap을 바꾸면 다시 계산한다
COL_W = 260 / 25.4

plt.rcParams.update({
    "svg.fonttype": "none",          # 글자를 글자로 남겨 브라우저 글꼴로 그린다
    "font.family": ["Apple SD Gothic Neo", "Noto Sans KR", "sans-serif"],
    "font.size": 20,                 # 포스터의 표·그림 설명 20pt와 맞춘다
    "axes.edgecolor": "#000", "axes.labelcolor": "#000",
    "xtick.color": "#000", "ytick.color": "#000",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "svg.hashsalt": "poster",        # 다시 만들어도 같은 SVG가 나오게 한다
})


def svg_of(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]                       # XML 머리말·DOCTYPE 제거
    s = re.sub(r'\swidth="[^"]+"\sheight="[^"]+"', "", s, count=1)  # 크기는 CSS가 정한다
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    return s


def qr_svg(url):
    qr = segno.make(url, error="m")
    n = qr.symbol_size(scale=1, border=2)[0]
    buf = io.BytesIO()
    qr.save(buf, kind="svg", xmldecl=False, svgns=True, scale=1, border=2, dark="#000")
    s = buf.getvalue().decode()
    s = re.sub(r'\swidth="\d+"\sheight="\d+"', f' viewBox="0 0 {n} {n}" shape-rendering="crispEdges"', s, count=1)
    return s.replace("<svg ", f'<svg role="img" aria-label="{url}" ', 1)


def inject(figs):
    html = POSTER.read_text(encoding="utf-8")
    for name, svg in figs.items():
        pat = re.compile(rf"(<!-- FIG:{name} -->).*?(<!-- /FIG:{name} -->)", re.S)
        if not pat.search(html):
            raise SystemExit(f"poster.html에 FIG:{name} 표시가 없습니다")
        html = pat.sub(lambda m: m.group(1) + svg + m.group(2), html)
    POSTER.write_text(html, encoding="utf-8")


def bar_chart(names, values, pick, base, fmt, xlabel, height=4.6):
    """평가지표 비교 가로 막대. pick=선택 모델 이름, base=기준선 이름."""
    fig, ax = plt.subplots(figsize=(COL_W, height))
    y = np.arange(len(names))[::-1]
    colors = [ACCENT if n == pick else DARK if n == base else GRAY for n in names]
    ax.barh(y, values, color=colors, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for t, n in zip(ax.get_yticklabels(), names):
        if n == pick:
            t.set_color(ACCENT)
            t.set_fontweight("bold")
    vmax = max(values)
    for yi, v, n in zip(y, values, names):
        ax.text(v + vmax * 0.012, yi, fmt(v), va="center", fontsize=20,
                color=ACCENT if n == pick else "#000", fontweight="bold" if n == pick else "normal")
    ax.set_xlim(0, vmax * 1.16)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:,.0f}" if t >= 1 or t == 0 else f"{t:.2f}"))
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return svg_of(fig)


# ───────────── 이 샘플의 데이터와 그래프(샘플마다 바꾼다) ─────────────
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

DATA = ROOT / "data" / "kbo_hitters_2026_full.csv"
REPO_URL = "https://github.com/greatsong/ds-sample-3-clustering"
APP_URL = "https://greatsong.github.io/ds-sample-3-clustering/"
LIGHT = "#7f9bd1"       # 셋째 유형 색(남색을 옅게)
RATE = ["홈런율", "도루율", "볼넷율", "삼진율", "2루타율"]
COUNT = ["홈런", "도루", "볼넷", "삼진", "2루타"]

# check.py와 같은 데이터 준비: 타석 100 이상, 타석당 비율 다섯 개, 표준화
raw = pd.read_csv(DATA, encoding="utf-8")
d = raw[raw["타석"] >= 100].copy().reset_index(drop=True)
for r, c in zip(RATE, COUNT):
    d[r] = d[c] / d["타석"]
Xs = StandardScaler().fit_transform(d[RATE])

# 그래프 1(FIG:hist): 타석 분포(364명)와 기준 100타석
fig, ax = plt.subplots(figsize=(COL_W, 2.3))
ax.hist(raw["타석"], bins=np.arange(0, 701, 20), color=ACCENT, edgecolor="#fff", linewidth=0.6)
top = ax.get_ylim()[1]
ax.axvline(100, color="#000", linestyle=(0, (4, 3)), linewidth=1.4)
n100 = int((raw["타석"] >= 100).sum())
ax.annotate(f"기준 100타석: 이상 {n100}명 사용", xy=(100, top * 0.7), xytext=(250, top * 0.7), va="center", fontsize=20,
            arrowprops=dict(arrowstyle="-", color="#000", linewidth=1.0, shrinkA=4, shrinkB=0))
ax.set_xlabel("타석")
ax.set_ylabel("선수 수(명)")
ax.set_xlim(0, 700)
svg_hist = svg_of(fig)

# 그래프 2(FIG:bars): k별 실루엣(K-평균, k=2~6)과 같은 k=3에서 방법별 실루엣(check.py 4절·8절과 같은 모델)
names, sils = [], []
for k in range(2, 7):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
    names.append(f"K-평균 k={k}" + ("(선택)" if k == 3 else ""))
    sils.append(silhouette_score(Xs, lab))
others = [("계층적 군집 3묶음", AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Xs)),
          ("가우시안 혼합 3묶음", GaussianMixture(n_components=3, random_state=0).fit_predict(Xs))]
for name, lab in others:
    names.append(name)
    sils.append(silhouette_score(Xs, lab))
db = DBSCAN(eps=1.5, min_samples=10).fit_predict(Xs)
n_db = len(set(db) - {-1})
names.append("DBSCAN")
sils.append(None)
pick = "K-평균 k=3(선택)"

fig, ax = plt.subplots(figsize=(COL_W, 4.4))
y = np.arange(len(names))[::-1].astype(float)
y[5:] -= 0.5                                       # K-평균 k별 막대와 방법별 막대 사이를 띄운다
for yi, v, n in zip(y, sils, names):
    if v is None:
        ax.text(0.003, yi, f"계산 불가(묶음 {n_db}개, 잡음 {int((db == -1).sum())}명)", va="center", fontsize=20)
        continue
    ax.barh(yi, v, color=ACCENT if n == pick else GRAY, height=0.66)
    ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=20,
            color=ACCENT if n == pick else "#000", fontweight="bold" if n == pick else "normal")
ax.set_yticks(y)
ax.set_yticklabels(names)
for t, n in zip(ax.get_yticklabels(), names):
    if n == pick:
        t.set_color(ACCENT)
        t.set_fontweight("bold")
ax.set_xlim(0, 0.25)
ax.set_ylim(y.min() - 0.6, y.max() + 0.5)
ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:.2f}"))
ax.set_xlabel("실루엣(1에 가까울수록 묶음이 뚜렷함)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
svg_bars = svg_of(fig)

# 그래프 3(FIG:result): 홈런율 대 도루율 산점도, 색 = K-평균 k=3 유형(check.py 유형_이름과 같은 규칙)
lab = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xs)
d["군집"] = lab
mean = d.groupby("군집")[RATE].mean()
name = {mean["홈런율"].idxmax(): "거포형"}
rest = mean.drop(index=list(name))
name[rest["도루율"].idxmax()] = "준족형"
for i in mean.index:
    name.setdefault(i, "교타형")
d["유형"] = d["군집"].map(name)
style = {"거포형": (ACCENT, "o"), "교타형": (LIGHT, "s"), "준족형": (GRAY, "^")}
fig, ax = plt.subplots(figsize=(COL_W, 4.5))
for t in ["준족형", "거포형", "교타형"]:
    g = d[d["유형"] == t]
    c, m = style[t]
    ax.scatter(g["홈런율"], g["도루율"], s=80, color=c, marker=m, edgecolor="none", label=f"{t} {len(g)}명")
ax.set_xlabel("홈런율(홈런 ÷ 타석)")
ax.set_ylabel("도루율(도루 ÷ 타석)")
ax.set_xlim(-0.003, 0.08)
ax.set_ylim(-0.004, 0.115)
ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:.2f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:.2f}"))
ax.legend(loc="upper right", frameon=False, fontsize=20, handletextpad=0.2, borderaxespad=0.1)
svg_result = svg_of(fig)

inject({"hist": svg_hist, "bars": svg_bars, "result": svg_result,
        "qr_repo": qr_svg(REPO_URL), "qr_app": qr_svg(APP_URL)})
print("실루엣:", ", ".join(f"{n} {'계산 불가' if v is None else f'{v:.3f}'}" for n, v in zip(names, sils)))
print("유형별 인원:", d["유형"].value_counts().to_dict(), f"· 분석 대상 {len(d)}명 · DBSCAN 묶음 {n_db}개")
