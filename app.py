"""
RnE 연구 도우미 AI  —  공공데이터 AI 활용 분석 대회 출품작
Render Free 배포 최적화 버전 (sentence-transformers 제거, TF-IDF 사용)

주요 변경:
- sentence-transformers / torch / transformers 완전 제거 (512MB 메모리 초과 방지)
- TF-IDF + cosine_similarity 로 유사도 계산 (scikit-learn 만 사용)
- Groq 클라이언트 지연 초기화 (startup crash 방지)
- 방어적 CSV 로딩 (파일/컬럼 없어도 앱 작동)
- 전국 기관 동적 파생 (rne_collab.csv 에서 추출)
"""

import os
import re
import time
import logging
import tempfile
import requests

import numpy as np
import pandas as pd
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# 로깅 설정
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 환경 변수
# ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "당신은 과학영재 R&E(Research and Education) 프로그램 전문 멘토입니다. "
    "고등학생이 연구 주제를 가져오면 선행연구 기반으로 주제의 실현 가능성, "
    "참신성, 개선 방향을 전문적이고 구체적으로 피드백합니다. "
    "존댓말을 사용하고, 전문 용어는 쉽게 설명을 덧붙여 주세요."
)

# ─────────────────────────────────────────────
# 데이터 경로
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ─────────────────────────────────────────────
# 방어적 CSV 로더
# ─────────────────────────────────────────────
def _load(fname: str) -> pd.DataFrame:
    """파일 없거나 오류나도 빈 DataFrame 반환"""
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        log.warning("파일 없음: %s", path)
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            df = pd.read_csv(path, encoding=enc)
            log.info("로드 완료: %s (%d행)", fname, len(df))
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            log.error("CSV 로드 오류 %s: %s", fname, e)
            return pd.DataFrame()
    log.error("인코딩 실패: %s", fname)
    return pd.DataFrame()


def _col(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    """컬럼 없으면 기본값 시리즈 반환"""
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series([default] * len(df), dtype=str)


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
log.info("데이터 로드 시작...")

df_full   = _load("rne_full.csv")      # 전국 RnE 전체 (유사 검색용)
df_collab = _load("rne_collab.csv")    # 대학 협업 RnE (기관 스코어링용)
df_equip  = _load("dgb_equipment.csv") # 연구장비
df_rnd    = _load("dgb_research.csv")  # 연구재단 보고서

log.info(
    "로드 완료 — 전체RnE:%d 협업:%d 장비:%d 연구재단:%d",
    len(df_full), len(df_collab), len(df_equip), len(df_rnd),
)

# ─────────────────────────────────────────────
# 전국 기관 목록 동적 파생
# (rne_collab.csv 의 협력대학기관 컬럼에서 추출)
# ─────────────────────────────────────────────
INST_COL = "협력대학기관"

if not df_collab.empty and INST_COL in df_collab.columns:
    ALL_INSTITUTIONS = (
        df_collab[INST_COL]
        .dropna()
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
else:
    # 협업 데이터 없을 때 기본 목록 (fallback)
    ALL_INSTITUTIONS = [
        "POSTECH", "DGIST", "경북대학교", "영남대학교", "대구대학교",
        "계명대학교", "대구가톨릭대학교", "금오공과대학교", "안동대학교",
        "경일대학교", "서울대학교", "연세대학교", "KAIST", "GIST", "UNIST",
    ]

log.info("기관 목록 %d개 파생 완료", len(ALL_INSTITUTIONS))

# ─────────────────────────────────────────────
# 분야 추론 함수 (기관 스코어링 가산점용)
# ─────────────────────────────────────────────
_FIELD_INFER: list[tuple[str, list[str]]] = [
    ("생명과학", ["광합성", "세균", "유전자", "식물", "효소", "세포", "단백질", "dna", "바이러스",
                  "미생물", "생태", "진화", "면역", "호르몬", "신경", "뇌"]),
    ("물리",     ["힘", "속도", "전류", "광학", "파동", "역학", "전자기", "양자", "나노",
                  "열역학", "마찰", "중력", "전압", "자기", "레이저", "광자"]),
    ("수학",     ["함수", "확률", "통계", "행렬", "최적화", "방정식", "수열", "급수",
                  "위상", "정수론", "조합", "기하"]),
    ("정보",     ["ai", "ml", "머신러닝", "딥러닝", "알고리즘", "데이터", "분류",
                  "신경망", "강화학습", "자연어", "컴퓨터", "소프트웨어"]),
    ("화학",     ["분자", "반응", "합성", "촉매", "결합", "원소", "고분자", "소재",
                  "산화", "환원", "전해질", "유기", "무기"]),
    ("지구과학", ["대기", "해양", "지진", "기후", "천체", "우주", "행성", "지질",
                  "날씨", "화산", "태풍", "빙하"]),
    ("에너지",   ["태양전지", "연료전지", "배터리", "재생에너지", "태양광", "수소", "발전"]),
    ("융합",     ["융합", "복합", "다학제", "iot", "스마트"]),
]

def _infer_field_from_text(text: str) -> str:
    """텍스트에서 분야 추론 (규칙 기반, 경량)"""
    t = (text or "").lower()
    for field, kws in _FIELD_INFER:
        if sum(1 for k in kws if k in t) >= 2:
            return field
    for field, kws in _FIELD_INFER:
        if any(k in t for k in kws[:4]):
            return field
    return "미분류"

# ─────────────────────────────────────────────
# 연도별 트렌드 테이블 (정제된 CSV 그대로 사용)
# ─────────────────────────────────────────────
try:
    TREND_DF: pd.DataFrame = (
        df_full.groupby(["year", "subject"]).size()
        .reset_index(name="건수")
        .pivot(index="year", columns="subject", values="건수")
        .fillna(0).astype(int).reset_index()
        .rename(columns={"year": "연도"})
    ) if not df_full.empty else pd.DataFrame()
    TREND_DF.columns.name = None
    log.info("TREND_DF 생성 완료: %d행", len(TREND_DF))
except Exception as e:
    log.warning("트렌드 DF 생성 실패: %s", e)
    TREND_DF = pd.DataFrame()

# ─────────────────────────────────────────────
# 기관별 교수 TOP5 (소개용)
# ─────────────────────────────────────────────
PROF_BY_INST: dict[str, pd.DataFrame] = {}
if (
    not df_collab.empty
    and INST_COL in df_collab.columns
    and "지도교수" in df_collab.columns
):
    for inst, grp in df_collab.groupby(INST_COL):
        cnt = grp["지도교수"].dropna().value_counts().head(5).reset_index()
        cnt.columns = ["교수명", "협업횟수"]
        PROF_BY_INST[inst] = cnt


# ─────────────────────────────────────────────
# TF-IDF 벡터라이저 구성 (전역 1회)
# ─────────────────────────────────────────────
def _make_rne_text(row: pd.Series) -> str:
    """RnE 레코드를 검색용 텍스트로 변환 (키워드 3배 가중)"""
    parts: list[str] = []
    subj  = str(row.get("subject", "") or "").strip()
    title = str(row.get("title", "") or "").strip()
    kw    = str(row.get("keywords", "") or "").strip()
    ab    = str(row.get("abstract", "") or "").strip()
    if subj:  parts.append(subj)
    if title: parts += [title, title]        # 2배
    if kw:    parts += [kw, kw, kw]          # 3배 (키워드 강조)
    if ab:    parts.append(ab[:200])
    return " ".join(parts)


def _make_collab_text(row: pd.Series) -> str:
    """협업 RnE 레코드용 텍스트 (분야 앵커 포함)"""
    parts: list[str] = []
    subj  = str(row.get("분야", "") or "").strip()
    title = str(row.get("제목", "") or "").strip()
    kw    = str(row.get("주제어", "") or "").strip()
    ab    = str(row.get("연구요약", "") or "").strip()
    if subj:  parts.append(subj)
    if title: parts += [title, title]
    if kw:    parts += [kw, kw]
    if ab:    parts.append(ab[:200])
    return " ".join(parts)


# 전체 RnE TF-IDF
_tfidf_full:   TfidfVectorizer | None = None
_matrix_full:  object = None
_valid_full:   pd.DataFrame = pd.DataFrame()

# 협업 RnE TF-IDF
_tfidf_collab:  TfidfVectorizer | None = None
_matrix_collab: object = None
_valid_collab:  pd.DataFrame = pd.DataFrame()


def _init_tfidf() -> None:
    """TF-IDF 행렬 초기화 (최초 호출 시 1회)"""
    global _tfidf_full, _matrix_full, _valid_full
    global _tfidf_collab, _matrix_collab, _valid_collab

    if _tfidf_full is not None:
        return  # 이미 초기화됨

    log.info("TF-IDF 초기화 시작...")

    # 전체 RnE
    if not df_full.empty:
        df_full["_text"] = df_full.apply(_make_rne_text, axis=1)
        _valid_full = df_full[df_full["_text"].str.strip() != ""].reset_index(drop=True)
        _tfidf_full = TfidfVectorizer(max_features=8000, sublinear_tf=True)
        _matrix_full = _tfidf_full.fit_transform(_valid_full["_text"].tolist())
        log.info("전체 RnE TF-IDF: %d건", len(_valid_full))

    # 협업 RnE
    if not df_collab.empty:
        df_collab["_text"] = df_collab.apply(_make_collab_text, axis=1)
        _valid_collab = df_collab[df_collab["_text"].str.strip() != ""].reset_index(drop=True)
        _tfidf_collab = TfidfVectorizer(max_features=6000, sublinear_tf=True)
        _matrix_collab = _tfidf_collab.fit_transform(_valid_collab["_text"].tolist())
        log.info("협업 RnE TF-IDF: %d건", len(_valid_collab))

    log.info("TF-IDF 초기화 완료")


# ─────────────────────────────────────────────
# Groq 지연 초기화 & 호출
# ─────────────────────────────────────────────
def _call_groq(prompt: str, max_tokens: int = 1500, retries: int = 3) -> str:
    """Groq API 호출 (지연 초기화, 재시도 포함)"""
    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY가 설정되지 않았습니다. Render 환경 변수를 확인해주세요."
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        return f"❌ Groq 클라이언트 초기화 실패: {e}"

    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return res.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            log.warning("Groq 호출 실패 (시도 %d/%d): %s → %ds 대기", attempt + 1, retries, e, wait)
            if attempt < retries - 1:
                time.sleep(wait)

    return "❌ AI 응답 생성에 실패했습니다. 잠시 후 다시 시도해주세요."


def _translate_to_english(topic: str) -> str:
    """한국어 연구 주제 → 영어 키워드 (실패 시 원본 반환)"""
    if not GROQ_API_KEY:
        return topic
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content":
                f"다음 한국어 연구 주제를 학술 논문 검색에 적합한 영어 키워드 4~6개로 변환해줘. "
                f"쉼표로 구분된 키워드만 출력하고 설명은 절대 포함하지 마.\n\n주제: {topic}"
            }],
            max_tokens=100,
            temperature=0.1,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        log.warning("번역 실패: %s", e)
        return topic


# ─────────────────────────────────────────────
# 논문 검색 (Semantic Scholar + OpenAlex)
# ─────────────────────────────────────────────
_HEADERS = {"User-Agent": "RnE-AI-Research-Assistant/2.0"}


def _search_semantic_scholar(keyword: str, limit: int = 10) -> list[dict]:
    try:
        res = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": keyword, "limit": limit,
                    "fields": "title,authors,year,abstract,externalIds,paperId"},
            headers=_HEADERS, timeout=12,
        )
        if res.status_code != 200:
            return []
        out = []
        for p in res.json().get("data", []):
            abstract = (p.get("abstract") or "").strip()
            if not abstract:
                continue
            authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
            ids = p.get("externalIds") or {}
            url = (f"https://doi.org/{ids['DOI']}" if "DOI" in ids
                   else f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}")
            out.append({
                "source": "Semantic Scholar",
                "title":  (p.get("title") or "-").strip(),
                "authors": authors or "-",
                "year":   str(p.get("year") or "-"),
                "abstract": abstract[:280],
                "url":    url,
                "_search": f"{p.get('title', '')} {abstract}",
            })
        return out
    except Exception as e:
        log.warning("Semantic Scholar 오류: %s", e)
        return []


def _search_openalex(keyword: str, limit: int = 10) -> list[dict]:
    try:
        res = requests.get(
            "https://api.openalex.org/works",
            params={"search": keyword, "per-page": limit,
                    "filter": "has_abstract:true",
                    "select": "title,authorships,publication_year,abstract_inverted_index,doi",
                    "mailto": "rne-ai@research.kr"},
            headers=_HEADERS, timeout=12,
        )
        if res.status_code != 200:
            return []
        out = []
        for w in res.json().get("results", []):
            inv = w.get("abstract_inverted_index") or {}
            if not inv:
                continue
            word_pos = [(wd, pos) for wd, positions in inv.items() for pos in positions]
            abstract = " ".join(wd for wd, _ in sorted(word_pos, key=lambda x: x[1]))
            if not abstract.strip():
                continue
            authors = ", ".join(
                a.get("author", {}).get("display_name", "")
                for a in (w.get("authorships") or [])[:3]
            )
            title = (w.get("title") or "-").strip()
            doi   = w.get("doi") or ""
            url   = doi if doi.startswith("http") else (f"https://doi.org/{doi}" if doi else "")
            out.append({
                "source": "OpenAlex",
                "title":  title,
                "authors": authors or "-",
                "year":   str(w.get("publication_year") or "-"),
                "abstract": abstract[:280],
                "url":    url,
                "_search": f"{title} {abstract}",
            })
        return out
    except Exception as e:
        log.warning("OpenAlex 오류: %s", e)
        return []


def _tfidf_rerank(topic: str, papers: list[dict]) -> list[dict]:
    """TF-IDF 기반 논문 재순위화 (sentence-transformers 대체)"""
    if not papers or len(papers) < 2:
        return papers
    try:
        texts = [f"{topic}"] + [p["_search"] for p in papers]
        vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        mat = vec.fit_transform(texts)
        sims = cosine_similarity(mat[0:1], mat[1:])[0]
        ranked = sorted(zip(papers, sims.tolist()), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked]
    except Exception as e:
        log.warning("TF-IDF 재순위화 실패: %s", e)
        return papers


def search_papers(topic: str, total: int = 6) -> list[dict]:
    """논문 검색: 번역 → API → TF-IDF 재순위화"""
    en_kw = _translate_to_english(topic)
    log.info("논문 검색 키워드: %s", en_kw)

    raw  = _search_semantic_scholar(en_kw, limit=10)
    time.sleep(0.4)
    raw += _search_openalex(en_kw, limit=10)

    # 중복 제거
    seen: set[str] = set()
    unique: list[dict] = []
    for r in raw:
        key = r["title"].lower()[:40]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if not unique:
        return [{"source": "없음", "title": f"관련 논문 없음 (키워드: {en_kw})",
                 "authors": "-", "year": "-", "abstract": "잠시 후 다시 시도해주세요.", "url": ""}]

    ranked = _tfidf_rerank(topic, unique)
    top = ranked[:total]
    for r in top:
        r.pop("_search", None)
    return top


# ─────────────────────────────────────────────
# 유사 R&E 검색 (TF-IDF)
# ─────────────────────────────────────────────
def _extract_keywords_fast(topic: str) -> list[str]:
    """
    Groq 없이 규칙 기반 키워드 추출 (빠른 fallback)
    2글자 이상 한국어 명사 토큰 + 영어 단어 추출
    """
    ko_words = re.findall(r"[가-힣]{2,}", topic)
    en_words = re.findall(r"[a-zA-Z]{3,}", topic.lower())
    return list(dict.fromkeys(ko_words + en_words))  # 순서 유지 중복 제거


def _extract_keywords_llm(topic: str) -> list[str]:
    """Groq로 핵심 키워드 추출 (실패 시 규칙 기반 fallback)"""
    if not GROQ_API_KEY:
        return _extract_keywords_fast(topic)
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content":
                f"다음 연구 주제에서 핵심 키워드 5~12개를 추출해줘. "
                f"한국어 키워드를 쉼표로 구분해서만 출력하고 설명은 하지 마.\n\n주제: {topic}"
            }],
            max_tokens=120, temperature=0.1,
        )
        raw = res.choices[0].message.content.strip()
        kws = [k.strip() for k in re.split(r"[,，\n]", raw) if k.strip()]
        return kws if kws else _extract_keywords_fast(topic)
    except Exception:
        return _extract_keywords_fast(topic)


def _keyword_overlap_score(kws: list[str], text: str) -> float:
    """키워드 집합과 텍스트 간 겹침 비율 (0~1)"""
    if not kws or not text:
        return 0.0
    t = text.lower()
    hits = sum(1 for k in kws if k.lower() in t)
    return hits / len(kws)


def _field_match_score(topic_kws: list[str], row_subject: str) -> float:
    """주제 키워드와 데이터 분야 일치 여부"""
    if not row_subject or row_subject == "미분류":
        return 0.0
    inferred = _infer_field_from_text(" ".join(topic_kws))
    return 1.0 if inferred == row_subject else 0.0


def search_rne_similar(
    topic: str,
    field_filter: str = "전체",
    top_k: int = 5,
) -> pd.DataFrame:
    """
    전국 RnE 데이터베이스에서 키워드 가중 혼합 유사도 검색
    final_similarity =
        0.45 * keyword_overlap
      + 0.25 * tfidf_similarity
      + 0.20 * field_match
      + 0.10 * method_keyword_overlap
    """
    _init_tfidf()

    empty_cols = ["연도", "분야", "제목", "소속고등학교", "주제어", "유사도(%)"]
    if _tfidf_full is None or _valid_full.empty:
        return pd.DataFrame(columns=empty_cols)

    try:
        df = _valid_full.copy()

        # 분야 필터
        subj_col = "subject"
        if field_filter and field_filter != "전체":
            sub = df[df[subj_col] == field_filter]
            if not sub.empty:
                df = sub

        # ① TF-IDF 유사도
        if len(df) < len(_valid_full):
            # 서브셋 재계산
            vec2 = TfidfVectorizer(max_features=6000, sublinear_tf=True)
            mat2 = vec2.fit_transform(df["_text"].tolist())
            tv2  = vec2.transform([topic])
            tfidf_sims = cosine_similarity(tv2, mat2)[0]
        else:
            tv = _tfidf_full.transform([topic])
            tfidf_sims = cosine_similarity(tv, _matrix_full)[0]

        # ② 키워드 추출 (LLM → fallback)
        topic_kws = _extract_keywords_llm(topic)
        log.info("추출된 키워드: %s", topic_kws)

        # ③ 혼합 유사도 계산
        mixed_sims = np.zeros(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            row_text    = str(row.get("_text", ""))
            row_kw_text = str(row.get("keywords", "") or "")
            row_subj    = str(row.get(subj_col, "") or "")

            kw_score     = _keyword_overlap_score(topic_kws, row_text)
            field_score  = _field_match_score(topic_kws, row_subj)
            # 방법론/재료 키워드 (keywords 컬럼만 대상)
            method_score = _keyword_overlap_score(topic_kws, row_kw_text)

            mixed_sims[i] = (
                0.45 * kw_score
                + 0.25 * tfidf_sims[i]
                + 0.20 * field_score
                + 0.10 * method_score
            )

        # ④ 상위 top_k 선택
        top_idx = mixed_sims.argsort()[::-1][:top_k]
        result  = df.iloc[top_idx].copy()
        # 0~100 스케일로 정규화 (최댓값 기준)
        max_sim = mixed_sims[top_idx[0]] if mixed_sims[top_idx[0]] > 0 else 1.0
        result["유사도(%)"] = np.round(mixed_sims[top_idx] / max_sim * 100, 1)

        col_map = {
            "year": "연도", "subject": "분야", "title": "제목",
            "school": "소속고등학교", "keywords": "주제어",
        }
        result = result.rename(columns=col_map)
        show = [c for c in empty_cols if c in result.columns]
        return result[show].reset_index(drop=True)

    except Exception as e:
        log.error("RnE 유사 검색 오류: %s", e)
        return pd.DataFrame(columns=empty_cols)


# ─────────────────────────────────────────────
# 전국 기관 적합도 스코어링 (TF-IDF 기반)
# ─────────────────────────────────────────────
_FIELD_KW: dict[str, list[str]] = {
    "물리":    ["물리", "역학", "파동", "열", "전자기", "양자", "광학", "나노"],
    "화학":    ["화학", "분자", "반응", "합성", "촉매", "결합", "원소", "고분자", "소재"],
    "수학":    ["수학", "방정식", "함수", "미적분", "통계", "확률", "대수", "위상"],
    "정보":    ["ai", "ml", "머신러닝", "딥러닝", "알고리즘", "데이터", "컴퓨터", "인공지능"],
    "생명과학": ["생명", "세포", "유전", "단백질", "dna", "바이러스", "생물", "효소"],
    "지구과학": ["지구", "대기", "해양", "지진", "기후", "천체", "우주", "행성"],
    "융합":    ["융합", "복합"],
    "에너지":  ["에너지", "태양", "배터리", "연료전지", "전력", "재생", "태양전지"],
}


def _field_bonus(inst: str, topic: str, field_strength: dict[str, list[str]]) -> float:
    """분야 키워드 2개 이상 일치 시 보너스"""
    t = topic.lower()
    for field, kws in _FIELD_KW.items():
        if sum(1 for k in kws if k in t) >= 2:
            if inst in field_strength.get(field, []):
                return 0.10
    return 0.0


def _softmax(x: np.ndarray, temperature: float = 4.0) -> np.ndarray:
    x = np.array(x, dtype=float) / temperature
    e = np.exp(x - x.max())
    return e / e.sum()


def _rank_norm(series: pd.Series) -> pd.Series:
    return series.rank(method="average") / len(series)


def calculate_scores(topic: str) -> pd.DataFrame:
    """
    전국 기관 적합도 계산 (TF-IDF 기반, Rank 정규화)
    가중치: 연구분야 40% | RnE실적 25% | 연구재단 20% | 분야가산점 10% | 장비키워드 5%
    """
    _init_tfidf()

    empty = pd.DataFrame(columns=["기관명", "적합도(%)", "RnE실적수"])
    if _tfidf_collab is None or _valid_collab.empty:
        return empty

    try:
        # 분야별 강점 기관 동적 파생 (협업 데이터에서 상위 기관 추출)
        field_strength: dict[str, list[str]] = {}
        if INST_COL in _valid_collab.columns and "분야" in _valid_collab.columns:
            for field in _valid_collab["분야"].dropna().unique():
                top_insts = (
                    _valid_collab[_valid_collab["분야"] == field][INST_COL]
                    .value_counts().head(3).index.tolist()
                )
                field_strength[str(field)] = top_insts

        # 주제 TF-IDF 벡터
        tv = _tfidf_collab.transform([topic])

        rows = []
        for inst in ALL_INSTITUTIONS:
            mask = _valid_collab.get(INST_COL, pd.Series(dtype=str)) == inst
            ir   = _valid_collab[mask]

            # ① 연구분야 일치도 (Top-3 유사 논문 평균)
            if not ir.empty:
                idxs    = ir.index.tolist()
                sub_mat = _matrix_collab[idxs]
                sims    = cosine_similarity(tv, sub_mat)[0]
                top3    = float(np.sort(sims)[::-1][:3].mean())
            else:
                top3 = 0.0

            # ② RnE 실적
            rne_cnt = len(ir)

            # ③ 연구재단 역량
            if not df_rnd.empty and "주관기관명" in df_rnd.columns:
                kw = inst[:4]
                rnd_cnt = int(df_rnd["주관기관명"].str.contains(kw, na=False, regex=False).sum())
            else:
                rnd_cnt = 0

            # ④ 장비 키워드 매칭 (단순 키워드 겹침)
            equip_score = 0.0
            if not df_equip.empty and "기관명" in df_equip.columns:
                kw = inst[:4]
                eq_sub = df_equip[
                    df_equip["기관명"].str.contains(kw, na=False, regex=False)
                ]
                if not eq_sub.empty:
                    topic_words = set(re.findall(r"[가-힣]{2,}", topic))
                    eq_cols = []
                    for c in ["장비분류(중분류)", "장비분류(소분류)"]:
                        if c in eq_sub.columns:
                            eq_cols.append(eq_sub[c].fillna(""))
                    if eq_cols and topic_words:
                        eq_text = " ".join(" ".join(c.tolist()) for c in eq_cols)
                        eq_words = set(re.findall(r"[가-힣]{2,}", eq_text))
                        overlap = len(topic_words & eq_words)
                        equip_score = min(overlap / max(len(topic_words), 1), 1.0)

            rows.append({
                "기관명":    inst,
                "RnE실적수": rne_cnt,
                "_field":   top3,
                "_rne":     float(np.log1p(rne_cnt)),
                "_rnd":     float(np.log1p(rnd_cnt)),
                "_equip":   equip_score,
                "_bonus":   _field_bonus(inst, topic, field_strength),
            })

        df_s = pd.DataFrame(rows)

        # Rank 정규화
        for col in ["_field", "_rne", "_rnd", "_equip"]:
            df_s[col + "_r"] = _rank_norm(df_s[col])

        df_s["raw"] = (
            df_s["_field_r"] * 0.40
            + df_s["_rne_r"]   * 0.25
            + df_s["_rnd_r"]   * 0.20
            + df_s["_equip_r"] * 0.05
            + df_s["_bonus"]   * 0.10
        )

        sm = _softmax(df_s["raw"].to_numpy())
        df_s["적합도(%)"] = (sm * 100).round(1)

        return (
            df_s[["기관명", "적합도(%)", "RnE실적수"]]
            .sort_values("적합도(%)", ascending=False)
            .reset_index(drop=True)
        )

    except Exception as e:
        log.error("기관 스코어링 오류: %s", e)
        return empty


# ─────────────────────────────────────────────
# Groq AI 생성 함수
# ─────────────────────────────────────────────
def gen_feedback(topic: str, papers: list[dict], rne_sim: pd.DataFrame) -> str:
    paper_ctx = "\n".join(
        f"- [{p.get('year','-')}] {p.get('title','-')} ({p.get('authors','-')}) [{p.get('source','')}]"
        for p in papers[:5]
    )
    if rne_sim.empty:
        rne_ctx = "유사 R&E 연구 없음"
    else:
        rne_ctx = "\n".join(
            f"- [{r.get('연도','-')}] {r.get('제목','-')} / {r.get('소속고등학교','-')} (유사도 {r.get('유사도(%)','-')}%)"
            for _, r in rne_sim.iterrows()
        )
    return _call_groq(f"""
학생의 연구 주제: {topic}

[관련 선행연구]
{paper_ctx}

[국내 R&E 유사 연구 사례 (2020~2025 전국)]
{rne_ctx}

아래 4가지 항목을 분석해 주세요:
1. 주제의 학문적 의의와 참신성 (선행연구 대비)
2. 고등학생 R&E 수준에서의 실현 가능성
3. 주제 구체화를 위한 개선 방향 (2~3가지 구체적 제안)
4. 추천 연구 방법론 또는 접근법
""", max_tokens=1500)


def gen_guide(topic: str, top_inst: str) -> str:
    return _call_groq(f"""
연구 주제: {topic}
추천 협력 기관: {top_inst}

{top_inst}과 협력하여 R&E를 진행하려는 고등학생에게 아래 절차를 안내해 주세요:
1. R&E 신청 준비 단계 (연구계획서 핵심 요소 포함)
2. 지도교수 섭외 방법 (연락 채널, 이메일 팁)
3. 연구 진행 단계 (6개월 기준 월별 일정)
4. 결과물 정리 및 발표 준비
5. 핵심 주의사항 및 성공 팁
""", max_tokens=1200)


def gen_plan(topic: str, feedback: str, top_inst: str) -> str:
    return _call_groq(f"""
연구 주제: {topic}
추천 기관: {top_inst}
AI 피드백 핵심: {feedback[:500]}

R&E 신청용 연구 계획서 초안을 Markdown 형식으로 작성해 주세요:

## 연구 제목
## 연구 배경 및 필요성
## 연구 목적
- 목표 1
- 목표 2
## 연구 방법
## 기대 효과
## 연구 일정 (6개월)
| 기간 | 주요 활동 |
|------|---------|
""", max_tokens=1800)


# ─────────────────────────────────────────────
# 결과 다운로드 파일 생성
# ─────────────────────────────────────────────
def make_report(
    topic: str, papers: list[dict], rne_sim: pd.DataFrame,
    feedback: str, scores: pd.DataFrame, guide: str, plan: str,
) -> str:
    sep = "=" * 60
    lines = [sep, "RnE 연구 도우미 AI — 분석 결과 보고서", sep, f"연구 주제: {topic}", "",
             "[ 1. 선행 학술 논문 ]"]
    for p in papers:
        lines.append(f"  • [{p.get('year','-')}] {p.get('title','-')}")
        if p.get("url"):
            lines.append(f"    링크: {p['url']}")
    lines += ["", "[ 2. 유사 R&E 연구 ]"]
    if rne_sim.empty:
        lines.append("  유사 연구 없음")
    else:
        for _, r in rne_sim.iterrows():
            lines.append(
                f"  • [{r.get('연도','-')}] {r.get('제목','-')} (유사도 {r.get('유사도(%)','-')}%)"
            )
    lines += ["", "[ 3. AI 피드백 ]", feedback or "생성 실패",
              "", "[ 4. 기관 적합도 ]"]
    for _, r in scores.iterrows():
        lines.append(f"  {int(r.name)+1:2d}위. {r['기관명']:<15} {r['적합도(%)']}%  (RnE {r['RnE실적수']}건)")
    lines += ["", "[ 5. RnE 진행 가이드 ]", guide or "생성 실패",
              "", "[ 6. 연구 계획서 초안 ]", plan or "생성 실패"]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write("\n".join(lines))
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────
# 메인 분석 파이프라인
# ─────────────────────────────────────────────
def run_analysis(topic: str, field_filter: str) -> tuple:
    empty   = pd.DataFrame()
    err_msg = "⚠️ 연구 주제를 5자 이상 입력해주세요."
    if not topic or len(topic.strip()) < 5:
        return empty, empty, err_msg, empty, err_msg, err_msg, None, pd.DataFrame()

    topic = topic.strip()
    try:
        # ① 논문 검색
        papers    = search_papers(topic, total=6)
        df_papers = pd.DataFrame(papers)[["source", "title", "authors", "year", "abstract", "url"]]
        df_papers.columns = ["출처", "논문 제목", "저자", "연도", "초록(요약)", "원문 링크"]

        # ② 유사 RnE 검색
        rne_sim = search_rne_similar(topic, field_filter, top_k=5)

        # ③ AI 피드백
        feedback = gen_feedback(topic, papers, rne_sim)
        time.sleep(0.5)

        # ④ 기관 적합도
        scores   = calculate_scores(topic)
        top_inst = scores.iloc[0]["기관명"] if not scores.empty else "경북대학교"

        # ⑤ RnE 가이드
        guide = gen_guide(topic, top_inst)
        time.sleep(0.5)

        # ⑥ 연구 계획서
        plan = gen_plan(topic, feedback, top_inst)

        # ⑦ 교수 TOP5
        prof_df = PROF_BY_INST.get(top_inst, pd.DataFrame(columns=["교수명", "협업횟수"]))

        # ⑧ 결과 파일
        dl_path = make_report(topic, papers, rne_sim, feedback, scores, guide, plan)

        return df_papers, rne_sim, feedback, scores, guide, plan, dl_path, prof_df

    except Exception as e:
        log.error("run_analysis 오류: %s", e, exc_info=True)
        msg = f"❌ 분석 중 오류가 발생했습니다: {e}\n잠시 후 다시 시도해주세요."
        return empty, empty, msg, empty, msg, msg, None, pd.DataFrame()


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
_CSS = """
.gradio-container {
    font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}
.tab-nav button { font-size: 14px !important; font-weight: 600 !important; padding: 10px 16px !important; }
#submit-btn {
    background: #1d4ed8 !important; color: white !important;
    font-size: 16px !important; font-weight: 700 !important;
    height: 82px !important; border-radius: 10px !important;
}
#submit-btn:hover { background: #1e40af !important; }
#topic-box textarea { font-size: 15px !important; }
"""

_FIELD_OPTS = ["전체", "수학", "물리", "화학", "생명과학", "지구과학", "정보", "융합", "에너지"]

_EXAMPLES = [
    ["광합성 효율과 LED 파장의 관계 분석"],
    ["머신러닝을 활용한 대기오염 농도 예측 모델 개발"],
    ["페로브스카이트 태양전지의 광전변환 효율 향상 연구"],
    ["생분해성 플라스틱 대체 소재 개발 및 물성 분석"],
    ["딥러닝 기반 천체 스펙트럼 자동 분류 시스템"],
]

with gr.Blocks(css=_CSS, title="RnE 연구 도우미 AI") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:28px 0 16px;border-bottom:2px solid #e2e8f0;margin-bottom:20px">
      <h1 style="font-size:2.2rem;font-weight:900;color:#1e3a8a;margin:0">🔬 RnE 연구 도우미 AI</h1>
      <p style="color:#64748b;margin:10px 0 0;font-size:1rem">
        연구 주제 입력 → 선행연구 탐색 → AI 피드백 → 전국 기관 적합도 → 연구 계획서 자동 생성
      </p>
      <div style="display:inline-flex;gap:8px;margin-top:10px;flex-wrap:wrap;justify-content:center">
        <span style="padding:3px 12px;background:#dbeafe;border-radius:20px;font-size:0.82rem;color:#1e40af">Groq Llama 3.3 70B</span>
        <span style="padding:3px 12px;background:#dcfce7;border-radius:20px;font-size:0.82rem;color:#166534">Semantic Scholar · OpenAlex</span>
        <span style="padding:3px 12px;background:#fef9c3;border-radius:20px;font-size:0.82rem;color:#713f12">공공데이터 기반</span>
      </div>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            topic_box = gr.Textbox(
                label="연구 주제를 입력하세요",
                placeholder="예) 광합성 효율과 LED 파장의 관계 분석\n예) 머신러닝을 활용한 대기오염 예측 모델 개발",
                lines=3, elem_id="topic-box",
            )
            field_sel = gr.Dropdown(
                choices=_FIELD_OPTS, value="전체",
                label="🔎 분야 필터 (유사 R&E 검색 범위 설정)",
            )
        submit_btn = gr.Button("🚀 분석\n시작", variant="primary", elem_id="submit-btn", scale=1)

    gr.Examples(label="📌 예시 주제 클릭 → 자동 입력", examples=_EXAMPLES, inputs=topic_box)

    with gr.Tabs():

        with gr.Tab("📖 RnE란?"):
            gr.HTML("""
            <div style="padding:8px 4px">
            <h2 style="color:#1e3a8a;margin-top:0">R&E 프로그램이란?</h2>
            <p style="font-size:1rem;line-height:1.9;color:#334155">
              <b>R&E(Research and Education)</b>는 과학고·영재학교 학생이
              대학교수 또는 연구기관 연구원의 지도 아래 실제 학술 연구를 수행하는
              <b>과학영재 창의연구 프로그램</b>입니다.
            </p>
            <hr style="margin:16px 0;border-color:#e2e8f0">
            <h3 style="color:#1e3a8a">🗺 진행 단계</h3>
            <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px">
              <div style="flex:1;min-width:130px;background:#f0f9ff;border-radius:8px;padding:12px;border-left:4px solid #3b82f6">
                <b>1️⃣ 주제 선정</b><br><span style="font-size:0.9rem;color:#475569">지도교사와 연구 주제 구체화</span></div>
              <div style="flex:1;min-width:130px;background:#f0f9ff;border-radius:8px;padding:12px;border-left:4px solid #3b82f6">
                <b>2️⃣ 교수 섭외</b><br><span style="font-size:0.9rem;color:#475569">관련 분야 교수에게 이메일 연락</span></div>
              <div style="flex:1;min-width:130px;background:#f0f9ff;border-radius:8px;padding:12px;border-left:4px solid #3b82f6">
                <b>3️⃣ 계획서 제출</b><br><span style="font-size:0.9rem;color:#475569">연구 계획서 작성 및 승인</span></div>
              <div style="flex:1;min-width:130px;background:#f0f9ff;border-radius:8px;padding:12px;border-left:4px solid #3b82f6">
                <b>4️⃣ 연구 수행</b><br><span style="font-size:0.9rem;color:#475569">6개월~1년간 실험·분석</span></div>
              <div style="flex:1;min-width:130px;background:#f0f9ff;border-radius:8px;padding:12px;border-left:4px solid #3b82f6">
                <b>5️⃣ 결과 발표</b><br><span style="font-size:0.9rem;color:#475569">성과자료집 등재 · 발표</span></div>
            </div>
            <hr style="margin:16px 0;border-color:#e2e8f0">
            <h3 style="color:#1e3a8a">🤖 이 AI가 도와드리는 것</h3>
            <ul style="font-size:1rem;line-height:2.2;color:#334155;padding-left:20px">
              <li>📚 <b>선행연구 탐색</b> — 전 세계 2억+ 논문 검색 + 원문 링크</li>
              <li>🔍 <b>유사 R&E 검색</b> — 전국 R&E 1,357건 데이터베이스 분야 필터 탐색</li>
              <li>🤖 <b>AI 피드백</b> — 참신성 · 실현 가능성 · 개선 방향 분석</li>
              <li>🏛 <b>기관 적합도</b> — <b>전국</b> 대학·연구기관 % 스코어 + 교수 TOP5</li>
              <li>📝 <b>연구 계획서</b> — R&E 신청용 초안 자동 생성</li>
              <li>💾 <b>결과 다운로드</b> — 전체 분석 결과 txt 저장</li>
            </ul>
            </div>
            """)
            gr.Markdown("### 📊 연도별 R&E 연구 분야 분포 (2020~2025)")
            gr.Dataframe(value=TREND_DF, wrap=True)

        with gr.Tab("📚 1단계 · 선행연구 탐색"):
            gr.Markdown("### 🔍 학술 논문 검색 결과")
            gr.Markdown("_Semantic Scholar + OpenAlex — 2억+ 논문 · TF-IDF 재순위화 · API 키 불필요_")
            out_papers = gr.Dataframe(
                headers=["출처", "논문 제목", "저자", "연도", "초록(요약)", "원문 링크"],
                wrap=True, row_count=6,
            )
            gr.Markdown("---")
            gr.Markdown("### 📂 기존 R&E 유사 연구 사례")
            gr.Markdown("_전국 R&E 전체 1,357건 · 분야 필터 가능_")
            out_rne = gr.Dataframe(
                headers=["연도", "분야", "제목", "소속고등학교", "주제어", "유사도(%)"],
                wrap=True, row_count=5,
            )

        with gr.Tab("🤖 2단계 · AI 피드백"):
            gr.Markdown("### Groq AI (Llama 3.3 70B)의 연구 주제 분석")
            out_feedback = gr.Markdown()

        with gr.Tab("🏛 3단계 · 기관 적합도"):
            gr.Markdown("### 전국 연구기관 적합도 순위")
            gr.Markdown("_가중치: 연구분야 40% + RnE실적 25% + 연구재단역량 20% + 분야가산점 10% + 장비키워드 5%_")
            out_scores = gr.Dataframe(headers=["기관명", "적합도(%)", "RnE실적수"], wrap=True, row_count=15)
            gr.Markdown("### 🎓 1위 기관 협업 교수 TOP 5")
out_prof = gr.Dataframe(value=pd.DataFrame(), headers=["교수명", "협업횟수"], wrap=True, row_count=5)
        with gr.Tab("📋 4단계 · RnE 진행 가이드"):
            gr.Markdown("### 적합도 1위 기관 기준 단계별 RnE 진행 절차")
            out_guide = gr.Markdown()

        with gr.Tab("📝 5단계 · 연구 계획서 초안"):
            gr.Markdown("### R&E 신청용 연구 계획서 자동 생성")
            out_plan = gr.Markdown()

    gr.Markdown("---")
    with gr.Row():
        gr.Markdown("### 💾 전체 분석 결과 다운로드")
        out_dl = gr.File(label="결과 저장 (txt)")

    gr.HTML("""
    <div style="text-align:center;padding:16px 0 8px;margin-top:8px;
                border-top:1px solid #e2e8f0;color:#94a3b8;font-size:0.80rem;line-height:1.9">
      데이터 출처: 한국연구재단 R&E 성과자료집(2020~2025) · Semantic Scholar · OpenAlex ·
      대학알리미 · 중소기업기술정보진흥원 · 한국연구재단<br>
      <b>공공데이터 AI 활용 분석 대회 출품작</b>
    </div>
    """)

    _outputs = [out_papers, out_rne, out_feedback, out_scores,
                out_guide, out_plan, out_dl, out_prof]
    submit_btn.click(fn=run_analysis, inputs=[topic_box, field_sel], outputs=_outputs)
    topic_box.submit(fn=run_analysis, inputs=[topic_box, field_sel], outputs=_outputs)


# ─────────────────────────────────────────────
# TF-IDF 시작 시 예열 (선택적)
# ─────────────────────────────────────────────
try:
    _init_tfidf()
except Exception as e:
    log.warning("TF-IDF 사전 초기화 실패 (첫 요청 시 초기화됨): %s", e)


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
