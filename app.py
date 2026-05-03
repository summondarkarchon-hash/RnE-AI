"""
RnE 연구 도우미 AI — 공공데이터 AI 활용 분석 대회 출품작
Author : RnE AI Team
Version: 2.0 (Final)

수정 이력:
  - 기관 적합도 스코어링 전면 재설계 (rank 기반 정규화로 POSTECH 독점 현상 해결)
  - 기관명 한국어 별칭 매핑 (POSTECH→포항공과, DGIST→대구경북과학기술원)
  - 논문 검색 재순위화 개선 (한/영 이중 임베딩 평균)
  - 논문 관련도 임계값 필터 (cosine < 0.20 논문 제외)
  - 검색 텍스트 가중치 개선 (제목·키워드 x2 반복)
  - Groq API 재시도 로직 추가
  - 전체 예외 처리 강화
"""

import os, re, time, requests, tempfile, logging
import pandas as pd
import numpy as np
import gradio as gr
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
# 1. API 설정
# ──────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
GROQ_MODEL   = 'llama-3.3-70b-versatile'

SYSTEM_PROMPT = (
    "당신은 과학영재 R&E(Research and Education) 프로그램 전문 멘토입니다. "
    "고등학생이 연구 주제를 가져오면 선행연구 기반으로 주제의 실현 가능성, "
    "참신성, 개선 방향을 전문적이고 구체적으로 피드백합니다. "
    "존댓말을 사용하고, 전문 용어는 쉽게 설명을 덧붙여 주세요."
)

# ──────────────────────────────────────────────────────
# 2. 데이터 로드
# ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def _load(fname: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        log.warning(f'파일 없음: {path}')
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except Exception as e:
        log.error(f'CSV 로드 실패 {fname}: {e}')
        return pd.DataFrame()

log.info('데이터 로드 시작...')
df_full   = _load('rne_full.csv')           # 전체 RnE 1,357건 — 유사 주제 검색
df_collab = _load('rne_collab.csv')   # 대학 협업 RnE 171건 — 기관 스코어링
df_equip  = _load('dgb_equipment.csv')
df_rnd    = _load('dgb_research.csv')
log.info(f'로드 완료 — 전체RnE:{len(df_full)} 협업:{len(df_collab)} 장비:{len(df_equip)} 연구재단:{len(df_rnd)}')

# 연도별 분야 트렌드 (소개 탭용, 로드 실패해도 빈 DF 반환)
try:
    TREND_DF = (
        df_full.groupby(['year', 'subject']).size()
        .reset_index(name='건수')
        .pivot(index='year', columns='subject', values='건수')
        .fillna(0).astype(int).reset_index()
        .rename(columns={'year': '연도'})
    ) if not df_full.empty else pd.DataFrame()
    TREND_DF.columns.name = None
except Exception as e:
    log.warning(f'트렌드 DF 생성 실패: {e}')
    TREND_DF = pd.DataFrame()

# 기관별 자주 협업한 교수 집계 (협업 데이터 기반)
PROF_BY_INST: dict[str, pd.DataFrame] = {}
if (not df_collab.empty
        and '협력대학기관' in df_collab.columns
        and '지도교수' in df_collab.columns):
    for inst, grp in df_collab.groupby('협력대학기관'):
        counts = grp['지도교수'].dropna().value_counts().head(5).reset_index()
        counts.columns = ['교수명', '협업횟수']
        PROF_BY_INST[inst] = counts

# ──────────────────────────────────────────────────────
# 3. 임베딩 모델 & 벡터 DB
# ──────────────────────────────────────────────────────
log.info('임베딩 모델 로드 중...')
embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
log.info('모델 로드 완료')

def _make_text_full(row) -> str:
    """전체 RnE 검색 텍스트: 제목·키워드 2배 가중, 초록 앞 200자"""
    parts = []
    title = str(row.get('title', '') or '').strip()
    kw    = str(row.get('keywords', '') or '').strip()
    ab    = str(row.get('abstract', '') or '').strip()
    if title: parts += [title, title]          # 제목 2배
    if kw:    parts += [kw, kw]               # 키워드 2배
    if ab:    parts.append(ab[:200])
    return ' '.join(parts)

def _make_text_collab(row) -> str:
    """협업 RnE 검색 텍스트: 분야·제목·키워드 2배, 요약 앞 200자"""
    parts = []
    subj  = str(row.get('분야', '') or '').strip()
    title = str(row.get('제목', '') or '').strip()
    kw    = str(row.get('주제어', '') or '').strip()
    ab    = str(row.get('연구요약', '') or '').strip()
    if subj:  parts.append(subj)              # 분야명으로 주제 앵커
    if title: parts += [title, title]
    if kw:    parts += [kw, kw]
    if ab:    parts.append(ab[:200])
    return ' '.join(parts)

log.info('전체 RnE 임베딩 시작...')
df_full['_text']   = df_full.apply(_make_text_full, axis=1)
emb_full = embedder.encode(
    df_full['_text'].tolist(), batch_size=32,
    show_progress_bar=True, normalize_embeddings=True
) if not df_full.empty else np.zeros((0, 768))

log.info('협업 RnE 임베딩 시작...')
df_collab['_text'] = df_collab.apply(_make_text_collab, axis=1)
emb_collab = embedder.encode(
    df_collab['_text'].tolist(), batch_size=32,
    show_progress_bar=True, normalize_embeddings=True
) if not df_collab.empty else np.zeros((0, 768))

log.info('임베딩 완료!')

# ──────────────────────────────────────────────────────
# 4. 기관 정보 매핑
# ──────────────────────────────────────────────────────
# 대구·경북 소재 13개 주요 협력 가능 기관
DGB_INSTITUTIONS = [
    'POSTECH', 'DGIST', '경북대학교', '영남대학교', '대구대학교',
    '계명대학교', '대구가톨릭대학교', '금오공과대학교', '안동대학교',
    '경일대학교', '대구한의대학교', '위덕대학교', '한동대학교',
]

# 연구재단 데이터 검색용 한국어 기관명 매핑
INST_RND_KW: dict[str, str] = {
    'POSTECH':      '포항공과',
    'DGIST':        '대구경북과학기술원',
    'KAIST':        '한국과학기술원',
    'GIST':         '광주과학기술원',
    'UNIST':        '울산과학기술원',
}

# 연구장비 데이터 검색용 한국어 기관명 매핑
INST_EQUIP_KW: dict[str, str] = {
    'POSTECH':  '포항공과',           # 포항공과대학교 산학협력단
    'DGIST':    '대구경북과학기술원',
    'KAIST':    '한국과학기술원',
    'GIST':     '광주과학기술원',
    'UNIST':    '울산과학기술원',
}

# 분야별 강점 기관 (한국 과학계 특성 반영)
FIELD_KEYWORDS: dict[str, list[str]] = {
    '물리':    ['물리', '역학', '파동', '열', '전자기', '양자', '광학', '나노'],
    '화학':    ['화학', '분자', '반응', '합성', '촉매', '결합', '원소', '고분자', '소재'],
    '수학':    ['수학', '방정식', '함수', '미적분', '통계', '확률', '대수', '위상', '수론'],
    '정보':    ['ai', 'ml', '머신러닝', '딥러닝', '알고리즘', '데이터', '컴퓨터', '인공지능', '신경망', '모델'],
    '생명과학': ['생명', '세포', '유전', '단백질', 'dna', '바이러스', '생물', '효소', '미생물', '식물'],
    '지구과학': ['지구', '대기', '해양', '지진', '기후', '천체', '우주', '행성', '천문', '지질'],
    '융합':    ['융합', '복합', '다학제', '학제간'],
    '에너지':  ['에너지', '태양', '배터리', '연료전지', '전력', '재생', '광합성', '태양전지'],
}

FIELD_STRENGTH: dict[str, list[str]] = {
    '물리':    ['POSTECH', 'DGIST', '경북대학교'],
    '화학':    ['경북대학교', '영남대학교', '대구대학교'],
    '수학':    ['경북대학교', '영남대학교', '금오공과대학교'],
    '정보':    ['DGIST', 'POSTECH', '금오공과대학교', '경일대학교'],
    '생명과학': ['경북대학교', '계명대학교', '대구가톨릭대학교'],
    '지구과학': ['경북대학교', '영남대학교'],
    '융합':    ['DGIST', 'POSTECH', '경북대학교'],
    '에너지':  ['POSTECH', 'DGIST', '경북대학교', '영남대학교'],
}

# ──────────────────────────────────────────────────────
# 5. 논문 검색 함수
# ──────────────────────────────────────────────────────
_HEADERS = {'User-Agent': 'RnE-AI-Research-Assistant/2.0 (academic-project)'}
_PAPER_RELEVANCE_THRESHOLD = 0.20   # 이 값 미만이면 무관련 논문으로 제외

def _call_groq(prompt: str, max_tokens: int = 1500, retries: int = 3) -> str:
    """Groq API 호출 (재시도 로직 포함)"""
    if not groq_client:
        return '❌ GROQ_API_KEY가 설정되지 않았습니다. Space Settings → Secrets를 확인해주세요.'
    for attempt in range(retries):
        try:
            res = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': prompt},
                ],
                max_tokens=max_tokens, temperature=0.7,
            )
            return res.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f'Groq 호출 실패 (시도 {attempt+1}/{retries}): {e} → {wait}s 대기')
            if attempt < retries - 1:
                time.sleep(wait)
    return '❌ AI 응답을 생성할 수 없습니다. 잠시 후 다시 시도해주세요.'


def _translate_to_english(topic: str) -> str:
    """한국어 연구 주제 → 영어 학술 키워드 (Groq 사용, 실패 시 원본 반환)"""
    if not groq_client:
        return topic
    try:
        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content':
                f'다음 한국어 연구 주제를 학술 논문 검색에 적합한 영어 키워드 4~6개로 변환해줘. '
                f'쉼표로 구분된 키워드만 출력하고 설명, 서론, 마침표는 절대 포함하지 마.\n\n주제: {topic}'
            }],
            max_tokens=100, temperature=0.1,
        )
        result = res.choices[0].message.content.strip()
        # 번역 결과 기본 검증 (한글이 섞여있으면 재시도 안 하고 그냥 사용)
        return result if result else topic
    except Exception as e:
        log.warning(f'번역 실패: {e}')
        return topic


def _search_semantic_scholar(keyword: str, limit: int = 12) -> list[dict]:
    """Semantic Scholar API — 키 불필요, 최대 12건 검색"""
    try:
        res = requests.get(
            'https://api.semanticscholar.org/graph/v1/paper/search',
            params={
                'query':  keyword,
                'limit':  limit,
                'fields': 'title,authors,year,abstract,externalIds,paperId',
            },
            headers=_HEADERS, timeout=15,
        )
        if res.status_code != 200:
            log.warning(f'S2 HTTP {res.status_code}')
            return []
        out = []
        for p in res.json().get('data', []):
            abstract = (p.get('abstract') or '').strip()
            if not abstract:
                continue
            authors = ', '.join(a.get('name', '') for a in (p.get('authors') or [])[:3])
            ids     = p.get('externalIds') or {}
            if 'DOI' in ids:
                url = f"https://doi.org/{ids['DOI']}"
            else:
                url = f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
            out.append({
                'source':     'Semantic Scholar',
                'title':      (p.get('title') or '-').strip(),
                'authors':    authors or '-',
                'year':       str(p.get('year') or '-'),
                'abstract':   abstract[:300],
                'url':        url,
                '_full_text': f"{p.get('title','')} {abstract}",
            })
        return out
    except Exception as e:
        log.warning(f'Semantic Scholar 오류: {e}')
        return []


def _search_openalex(keyword: str, limit: int = 12) -> list[dict]:
    """OpenAlex API — 키 불필요, 초록 있는 것만"""
    try:
        res = requests.get(
            'https://api.openalex.org/works',
            params={
                'search':   keyword,
                'per-page': limit,
                'filter':   'has_abstract:true',
                'select':   'title,authorships,publication_year,abstract_inverted_index,doi',
                'mailto':   'rne-ai-assistant@research.kr',
            },
            headers=_HEADERS, timeout=15,
        )
        if res.status_code != 200:
            log.warning(f'OpenAlex HTTP {res.status_code}')
            return []
        out = []
        for w in res.json().get('results', []):
            inv = w.get('abstract_inverted_index') or {}
            if not inv:
                continue
            word_pos = [(wd, pos) for wd, positions in inv.items() for pos in positions]
            abstract = ' '.join(wd for wd, _ in sorted(word_pos, key=lambda x: x[1]))
            if not abstract.strip():
                continue
            authors = ', '.join(
                a.get('author', {}).get('display_name', '')
                for a in (w.get('authorships') or [])[:3]
            )
            title = (w.get('title') or '-').strip()
            doi   = w.get('doi') or ''
            url   = doi if doi.startswith('http') else (f'https://doi.org/{doi}' if doi else '')
            out.append({
                'source':     'OpenAlex',
                'title':      title,
                'authors':    authors or '-',
                'year':       str(w.get('publication_year') or '-'),
                'abstract':   abstract[:300],
                'url':        url,
                '_full_text': f"{title} {abstract}",
            })
        return out
    except Exception as e:
        log.warning(f'OpenAlex 오류: {e}')
        return []


def search_papers(topic: str, total: int = 6) -> list[dict]:
    """
    논문 검색 파이프라인:
    ① 한국어 → 영어 번역
    ② Semantic Scholar + OpenAlex API 검색 (각 12건)
    ③ 중복 제거
    ④ 한·영 이중 임베딩 평균으로 재순위화
    ⑤ 관련도 임계값 (0.20) 미만 제거
    ⑥ 상위 total건 반환
    """
    en_kw = _translate_to_english(topic)
    log.info(f'논문 검색 키워드: {en_kw}')

    raw  = _search_semantic_scholar(en_kw, limit=12)
    time.sleep(0.4)
    raw += _search_openalex(en_kw, limit=12)

    # 중복 제거 (제목 앞 40자 기준)
    seen: set[str] = set()
    unique: list[dict] = []
    for r in raw:
        key = r['title'].lower()[:40]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if not unique:
        return [{
            'source':   '검색 없음', 'title': f'관련 논문을 찾지 못했습니다 (키워드: {en_kw})',
            'authors':  '-', 'year': '-',
            'abstract': '다른 표현으로 주제를 입력해보거나 잠시 후 다시 시도해주세요.', 'url': '',
        }]

    # 한·영 이중 임베딩으로 재순위화
    tv_ko = embedder.encode([topic],   normalize_embeddings=True)
    tv_en = embedder.encode([en_kw],   normalize_embeddings=True)
    tv    = (tv_ko + tv_en) / 2
    norm  = np.linalg.norm(tv)
    if norm > 0:
        tv /= norm

    texts  = [r['_full_text'] for r in unique]
    pvecs  = embedder.encode(texts, normalize_embeddings=True)
    sims   = cosine_similarity(tv, pvecs)[0]

    ranked = sorted(zip(unique, sims.tolist()), key=lambda x: x[1], reverse=True)

    # 관련도 임계값 필터
    filtered = [(r, s) for r, s in ranked if s >= _PAPER_RELEVANCE_THRESHOLD]
    if not filtered:
        # 임계값 초과 없으면 상위 total건 그냥 반환 (검색 결과 자체가 없는 극단 상황)
        filtered = ranked

    top = [r for r, _ in filtered[:total]]
    for r in top:
        r.pop('_full_text', None)
    return top


def search_rne_similar(
    topic: str,
    field_filter: str = '전체',
    top_k: int = 5,
) -> pd.DataFrame:
    """
    전체 RnE 1,357건에서 유사 주제 검색
    - field_filter: '전체' 또는 분야명
    - 빈 텍스트 항목 제외
    - 결과: 한글 컬럼명 DataFrame
    """
    if df_full.empty or len(emb_full) == 0:
        return pd.DataFrame(columns=['연도', '분야', '제목', '소속고등학교', '주제어', '유사도(%)'])

    df = df_full.copy()
    if field_filter and field_filter != '전체':
        sub = df[df['subject'] == field_filter]
        if not sub.empty:
            df = sub

    # 빈 텍스트 행 사전 제거
    nonempty_mask = df['_text'].str.strip() != ''
    df = df[nonempty_mask]
    if df.empty:
        return pd.DataFrame(columns=['연도', '분야', '제목', '소속고등학교', '주제어', '유사도(%)'])

    idxs     = df.index.tolist()          # 원본 0-based 인덱스 (numpy 행 선택에 사용)
    sub_emb  = emb_full[idxs]            # 해당 행들의 임베딩
    tv       = embedder.encode([topic], normalize_embeddings=True)
    sims     = cosine_similarity(tv, sub_emb)[0]

    # 상위 top_k 선택 (df.iloc 기준 위치 인덱스)
    top_pos  = sims.argsort()[::-1][:top_k]
    result   = df.iloc[top_pos].copy()
    result['유사도(%)'] = np.round(sims[top_pos] * 100, 1)

    col_map = {
        'year': '연도', 'subject': '분야', 'title': '제목',
        'school': '소속고등학교', 'keywords': '주제어',
    }
    result = result.rename(columns=col_map)
    show   = [c for c in [*col_map.values(), '유사도(%)'] if c in result.columns]
    return result[show].reset_index(drop=True)

# ──────────────────────────────────────────────────────
# 6. 기관 적합도 스코어링
# ──────────────────────────────────────────────────────

# 시작 시 기관별 장비 텍스트 임베딩 사전 계산 (분석 시 재계산 방지)
log.info('기관별 연구장비 임베딩 사전 계산...')
EQUIP_VECS: dict[str, np.ndarray] = {}
if not df_equip.empty:
    for _inst in DGB_INSTITUTIONS:
        _kw  = INST_EQUIP_KW.get(_inst, _inst[:4])
        _sub = df_equip[
            df_equip['기관명'].str.contains(_kw, na=False, regex=False)
            & (df_equip['유휴불용'] == '활용')
        ]
        if _sub.empty:
            continue
        _txt = ' '.join(
            _sub['장비분류(중분류)'].fillna('').tolist()
            + _sub['장비분류(소분류)'].fillna('').tolist()
        )
        if _txt.strip():
            EQUIP_VECS[_inst] = embedder.encode([_txt], normalize_embeddings=True)
log.info(f'장비 임베딩 완료: {len(EQUIP_VECS)}개 기관')


def _field_bonus(inst: str, topic: str) -> float:
    """
    분야 가산점: 주제에서 2개 이상 키워드 일치 시 0.10 부여
    (1개 일치로 낮추면 너무 많은 기관에 적용되므로 2개로 상향)
    """
    t = topic.lower()
    for field, kws in FIELD_KEYWORDS.items():
        matched = sum(1 for k in kws if k in t)
        if matched >= 2 and inst in FIELD_STRENGTH.get(field, []):
            return 0.10
    return 0.0


def _equip_score(topic_vec: np.ndarray, inst: str) -> float:
    """장비 임베딩 유사도 — 사전 계산된 EQUIP_VECS 사용"""
    if inst not in EQUIP_VECS:
        return 0.0
    return float(np.clip(cosine_similarity(topic_vec, EQUIP_VECS[inst])[0][0], 0.0, 1.0))


def _rnd_count(inst: str) -> int:
    """연구재단 과제 건수 — 한국어 기관명으로 검색"""
    if df_rnd.empty or '주관기관명' not in df_rnd.columns:
        return 0
    search_kw = INST_RND_KW.get(inst, inst[:4])
    return int(df_rnd['주관기관명'].str.contains(search_kw, na=False, regex=False).sum())


def _rank_normalize(series: pd.Series) -> pd.Series:
    """Rank 백분위 정규화 (0→최하위, 1→최상위). 동점은 평균 rank 부여."""
    return series.rank(method='average') / len(series)


def calculate_scores(topic: str) -> pd.DataFrame:
    """
    기관별 적합도 계산 — rank 기반 정규화로 POSTECH 독점 현상 해결
    가중치:
      연구분야 일치도  40%  — 기관 RnE Top-3 유사도 평균, rank 정규화
      RnE 실적        25%  — log(건수+1), rank 정규화
      기자재 매칭      20%  — 장비 텍스트 임베딩 유사도, rank 정규화
      연구재단 역량    10%  — log(과제수+1), rank 정규화
      분야 가산점       5%  — 2개+ 키워드 일치 시 0.10 (rank 정규화 없이 직접 가산)
    → Softmax (temperature=4.0) → %
    """
    tv = embedder.encode([topic], normalize_embeddings=True)

    rows = []
    for inst in DGB_INSTITUTIONS:
        ir = df_collab[df_collab['협력대학기관'] == inst] if not df_collab.empty else pd.DataFrame()

        # ① 연구분야 일치도 — Top-3 유사 논문 평균
        if not ir.empty and len(emb_collab) > 0:
            idxs    = ir.index.tolist()
            ivecs   = emb_collab[idxs]
            sim_arr = cosine_similarity(tv, ivecs)[0]
            top3    = float(np.sort(sim_arr)[::-1][:3].mean())
        else:
            top3 = 0.0

        rows.append({
            '기관명':    inst,
            'RnE실적수': len(ir),
            '_field':   top3,
            '_rne':     float(np.log1p(len(ir))),
            '_equip':   _equip_score(tv, inst),
            '_rnd':     float(np.log1p(_rnd_count(inst))),
            '_bonus':   _field_bonus(inst, topic),
        })

    df_s = pd.DataFrame(rows)

    # Rank 백분위 정규화 (각 항목 독립)
    for col in ['_field', '_rne', '_equip', '_rnd']:
        df_s[col + '_r'] = _rank_normalize(df_s[col])

    # 가중 합산
    df_s['raw'] = (
        df_s['_field_r'] * 0.40
        + df_s['_rne_r']   * 0.25
        + df_s['_equip_r'] * 0.20
        + df_s['_rnd_r']   * 0.10
        + df_s['_bonus']   * 0.05   # 가산점은 rank 없이 직접 적용
    )

    # Softmax (temperature=4.0 → 완만한 분포)
    raw_arr = df_s['raw'].to_numpy(dtype=float)
    raw_arr = raw_arr / 4.0
    exp_arr = np.exp(raw_arr - raw_arr.max())
    df_s['적합도(%)'] = (exp_arr / exp_arr.sum() * 100).round(1)

    return (
        df_s[['기관명', '적합도(%)', 'RnE실적수']]
        .sort_values('적합도(%)', ascending=False)
        .reset_index(drop=True)
    )

# ──────────────────────────────────────────────────────
# 7. Groq AI 생성 함수
# ──────────────────────────────────────────────────────

def gen_feedback(topic: str, papers: list[dict], rne_sim: pd.DataFrame) -> str:
    paper_ctx = '\n'.join(
        f"- [{p.get('year','-')}] {p.get('title','-')} ({p.get('authors','-')}) [{p.get('source','')}]"
        for p in papers[:5]
    )
    if rne_sim.empty:
        rne_ctx = '유사 R&E 연구 없음'
    else:
        rne_ctx = '\n'.join(
            f"- [{r.get('연도','-')}] {r.get('제목','-')} / {r.get('소속고등학교','-')} (유사도 {r.get('유사도(%)','-')}%)"
            for _, r in rne_sim.iterrows()
        )
    return _call_groq(f"""
학생의 연구 주제: {topic}

[관련 선행연구 (Semantic Scholar + OpenAlex)]
{paper_ctx}

[국내 R&E 유사 연구 사례 (2020~2025 전국)]
{rne_ctx}

위 자료를 바탕으로 아래 4가지 항목을 분석해 주세요:
1. 주제의 학문적 의의와 참신성 (선행연구 대비)
2. 고등학생 R&E 수준에서의 실현 가능성
3. 주제 구체화를 위한 개선 방향 (2~3가지 구체적 제안)
4. 추천 연구 방법론 또는 접근법 (실험 설계 포함)
""", max_tokens=1500)


def gen_guide(topic: str, top_inst: str) -> str:
    return _call_groq(f"""
연구 주제: {topic}
추천 협력 기관: {top_inst}

{top_inst}과 협력하여 R&E를 진행하려는 고등학생에게 아래 절차를 단계별로 안내해 주세요.
각 단계는 구체적인 행동 지침과 주의사항을 포함해야 합니다:

1. R&E 신청 준비 단계 (연구계획서 핵심 요소 포함)
2. 지도교수 섭외 방법 (연락 채널, 이메일 작성 팁 포함)
3. 연구 진행 단계 (6개월 기준 월별 일정 예시)
4. 결과물 정리 및 발표 준비 (성과자료집 제출 방법)
5. 핵심 주의사항 및 성공 팁
""", max_tokens=1200)


def gen_plan(topic: str, feedback: str, top_inst: str) -> str:
    return _call_groq(f"""
연구 주제: {topic}
추천 기관: {top_inst}
AI 피드백 핵심: {feedback[:600]}

위 정보를 바탕으로 R&E 신청용 연구 계획서 초안을 작성해 주세요.
반드시 아래 Markdown 형식을 정확히 따르세요:

## 연구 제목
(주제를 학술적으로 구체화한 제목)

## 연구 배경 및 필요성
(이 연구가 필요한 이유를 3~4문장으로 서술)

## 연구 목적
- (목표 1)
- (목표 2)
- (목표 3)

## 연구 방법
(실험/분석 방법, 필요 장비, 데이터 수집 방법을 구체적으로)

## 기대 효과 및 활용 방안
(연구 결과의 학술적·사회적 의의)

## 연구 일정 (6개월)
| 기간 | 주요 활동 |
|------|---------|
| 1개월차 | ... |
| 2개월차 | ... |
| 3개월차 | ... |
| 4개월차 | ... |
| 5개월차 | ... |
| 6개월차 | ... |
""", max_tokens=1800)

# ──────────────────────────────────────────────────────
# 8. 결과 다운로드 파일 생성
# ──────────────────────────────────────────────────────

def make_report(
    topic: str,
    papers: list[dict],
    rne_sim: pd.DataFrame,
    feedback: str,
    scores: pd.DataFrame,
    guide: str,
    plan: str,
) -> str:
    """전체 분석 결과를 txt 파일로 저장하고 경로 반환"""
    sep = '=' * 60
    lines = [
        sep,
        'RnE 연구 도우미 AI — 분석 결과 보고서',
        sep,
        f'연구 주제: {topic}',
        '',
        '[ 1. 선행 학술 논문 ]',
    ]
    for p in papers:
        lines.append(f"  • [{p.get('year','-')}] {p.get('title','-')}")
        lines.append(f"    저자: {p.get('authors','-')}")
        if p.get('url'):
            lines.append(f"    링크: {p['url']}")

    lines += ['', '[ 2. 유사 R&E 연구 ]']
    if rne_sim.empty:
        lines.append('  유사 연구 없음')
    else:
        for _, r in rne_sim.iterrows():
            lines.append(
                f"  • [{r.get('연도','-')}] {r.get('제목','-')} "
                f"/ {r.get('소속고등학교','-')} (유사도 {r.get('유사도(%)','-')}%)"
            )

    lines += ['', '[ 3. AI 피드백 ]', feedback or '생성 실패']
    lines += ['', '[ 4. 기관 적합도 순위 ]']
    for _, r in scores.iterrows():
        lines.append(f"  {int(r.name)+1:2d}위. {r['기관명']:<15} {r['적합도(%)']:>5}%   (RnE실적 {r['RnE실적수']}건)")

    lines += ['', '[ 5. RnE 진행 가이드 ]', guide or '생성 실패']
    lines += ['', '[ 6. 연구 계획서 초안 ]', plan or '생성 실패']

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, encoding='utf-8'
    )
    tmp.write('\n'.join(lines))
    tmp.close()
    return tmp.name

# ──────────────────────────────────────────────────────
# 9. 메인 분석 파이프라인
# ──────────────────────────────────────────────────────

def run_analysis(topic: str, field_filter: str):
    topic = (topic or '').strip()
    if len(topic) < 5:
        empty = pd.DataFrame()
        msg   = '⚠️ 연구 주제를 5자 이상 구체적으로 입력해주세요.'
        return empty, empty, msg, empty, msg, msg, None, pd.DataFrame()

    log.info(f'분석 시작: {topic[:50]}')

    # ① 논문 검색 (번역 → API → 임베딩 재순위화)
    papers    = search_papers(topic, total=6)
    df_papers = pd.DataFrame(papers)[['source', 'title', 'authors', 'year', 'abstract', 'url']]
    df_papers.columns = ['출처', '논문 제목', '저자', '연도', '초록(요약)', '원문 링크']

    # ② 유사 RnE 검색 (전체 1,357건 DB)
    rne_sim = search_rne_similar(topic, field_filter, top_k=5)

    # ③ AI 피드백 (Groq)
    feedback = gen_feedback(topic, papers, rne_sim)
    time.sleep(1)    # Groq rate limit 방지

    # ④ 기관 적합도 (rank 정규화 + softmax)
    scores   = calculate_scores(topic)

    # ⑤ RnE 진행 가이드
    top_inst = scores.iloc[0]['기관명'] if not scores.empty else '경북대학교'
    guide    = gen_guide(topic, top_inst)
    time.sleep(1)

    # ⑥ 연구 계획서 초안
    plan     = gen_plan(topic, feedback, top_inst)

    # ⑦ 교수 TOP5
    prof_df  = PROF_BY_INST.get(top_inst, pd.DataFrame(columns=['교수명', '협업횟수']))

    # ⑧ 결과 txt 파일
    dl_path  = make_report(topic, papers, rne_sim, feedback, scores, guide, plan)

    log.info('분석 완료')
    return df_papers, rne_sim, feedback, scores, guide, plan, dl_path, prof_df

# ──────────────────────────────────────────────────────
# 10. Gradio UI
# ──────────────────────────────────────────────────────
_CSS = """
.gradio-container {
    font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}
.tab-nav button {
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 16px !important;
}
#submit-btn {
    background: #1d4ed8 !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    height: 82px !important;
    border-radius: 10px !important;
    transition: background 0.2s;
}
#submit-btn:hover { background: #1e40af !important; }
#topic-box textarea { font-size: 15px !important; }
.stat-card {
    background: #f0f9ff;
    border-radius: 10px;
    padding: 14px 18px;
    flex: 1;
    min-width: 140px;
    border-left: 4px solid #3b82f6;
}
"""

_EXAMPLES = [
    ['광합성 효율과 LED 파장의 관계 분석'],
    ['머신러닝을 활용한 대기오염 농도 예측 모델 개발'],
    ['페로브스카이트 태양전지의 광전변환 효율 향상 연구'],
    ['생분해성 플라스틱 대체 소재 개발 및 물성 분석'],
    ['딥러닝 기반 천체 스펙트럼 자동 분류 시스템'],
]

_FIELD_OPTS = ['전체', '수학', '물리', '화학', '생명과학', '지구과학', '정보', '융합', '에너지']

with gr.Blocks(css=_CSS, title='RnE 연구 도우미 AI') as demo:

    # 헤더
    gr.HTML("""
    <div style="text-align:center;padding:28px 0 16px;
                border-bottom:2px solid #e2e8f0;margin-bottom:20px">
      <h1 style="font-size:2.2rem;font-weight:900;color:#1e3a8a;margin:0;letter-spacing:-0.5px">
        🔬 RnE 연구 도우미 AI
      </h1>
      <p style="color:#64748b;margin:10px 0 0;font-size:1.05rem;line-height:1.6">
        연구 주제 입력 → 선행연구 탐색 → AI 피드백 → 대구·경북 기관 적합도 → 연구 계획서 자동 생성
      </p>
      <div style="display:inline-flex;gap:8px;margin-top:12px;flex-wrap:wrap;justify-content:center">
        <span style="padding:4px 12px;background:#dbeafe;border-radius:20px;font-size:0.82rem;color:#1e40af">
          Groq Llama 3.3 70B
        </span>
        <span style="padding:4px 12px;background:#dcfce7;border-radius:20px;font-size:0.82rem;color:#166534">
          Semantic Scholar · OpenAlex
        </span>
        <span style="padding:4px 12px;background:#fef9c3;border-radius:20px;font-size:0.82rem;color:#713f12">
          공공데이터 기반
        </span>
      </div>
    </div>
    """)

    # 입력 영역
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            topic_box = gr.Textbox(
                label='연구 주제를 입력하세요',
                placeholder=(
                    '예) 광합성 효율과 LED 파장의 관계 분석\n'
                    '예) 머신러닝을 활용한 대기오염 예측 모델 개발'
                ),
                lines=3, elem_id='topic-box',
            )
            field_sel = gr.Dropdown(
                choices=_FIELD_OPTS, value='전체',
                label='🔎 분야 필터 (유사 R&E 검색 범위 설정)',
            )
        submit_btn = gr.Button('🚀 분석\n시작', variant='primary',
                               elem_id='submit-btn', scale=1)

    gr.Examples(
        label='📌 예시 주제 클릭 → 자동 입력',
        examples=_EXAMPLES, inputs=topic_box,
    )

    # 탭
    with gr.Tabs():

        # ── 탭 0: RnE 소개 ──────────────────────
        with gr.Tab('📖 RnE란?'):
            gr.HTML("""
            <div style="padding:8px 4px">
            <h2 style="color:#1e3a8a;margin-top:0">R&E 프로그램이란?</h2>
            <p style="font-size:1rem;line-height:1.9;color:#334155">
              <b>R&E(Research and Education)</b>는 과학고·영재학교 학생이
              대학교수 또는 연구기관 연구원의 지도 아래 실제 학술 연구를 수행하는
              <b>과학영재 창의연구 프로그램</b>입니다.<br>
              단순 실험 실습이 아닌 연구 설계·수행·발표의 전 과정을 직접 경험하며,
              논문 등재와 학술대회 발표 기회를 얻을 수 있습니다.
            </p>
            <hr style="margin:18px 0;border-color:#e2e8f0">
            <h3 style="color:#1e3a8a">🗺 진행 단계</h3>
            <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px">
              <div class="stat-card"><b>1️⃣ 주제 선정</b><br><span style="font-size:0.9rem;color:#475569">지도교사와 함께 연구 주제 구체화</span></div>
              <div class="stat-card"><b>2️⃣ 교수 섭외</b><br><span style="font-size:0.9rem;color:#475569">관련 분야 교수에게 이메일 연락</span></div>
              <div class="stat-card"><b>3️⃣ 계획서 제출</b><br><span style="font-size:0.9rem;color:#475569">연구 계획서 작성 및 학교 승인</span></div>
              <div class="stat-card"><b>4️⃣ 연구 수행</b><br><span style="font-size:0.9rem;color:#475569">6개월~1년간 실험·분석 진행</span></div>
              <div class="stat-card"><b>5️⃣ 결과 발표</b><br><span style="font-size:0.9rem;color:#475569">성과자료집 등재 · 학술대회 발표</span></div>
            </div>
            <hr style="margin:18px 0;border-color:#e2e8f0">
            <h3 style="color:#1e3a8a">🤖 이 AI가 도와드리는 것</h3>
            <ul style="font-size:1rem;line-height:2.2;color:#334155;padding-left:20px">
              <li>📚 <b>선행연구 탐색</b> — 전 세계 2억+ 논문 실시간 검색 + 원문 링크 제공</li>
              <li>🔍 <b>유사 R&E 검색</b> — 전국 R&E 1,357건 데이터베이스에서 분야 필터 정밀 탐색</li>
              <li>🤖 <b>AI 피드백</b> — 주제 참신성·실현 가능성·개선 방향 전문 분석</li>
              <li>🏛 <b>기관 적합도</b> — 대구·경북 13개 대학 % 스코어 + 협업 교수 TOP 5</li>
              <li>📝 <b>연구 계획서</b> — R&E 신청용 계획서 초안(목적·방법·일정) 자동 생성</li>
              <li>💾 <b>결과 다운로드</b> — 전체 분석 결과를 txt 파일로 저장</li>
            </ul>
            </div>
            """)
            gr.Markdown('### 📊 연도별 R&E 연구 분야 분포 (2020~2025)')
            gr.Dataframe(value=TREND_DF, wrap=True)

        # ── 탭 1: 선행연구 ───────────────────────
        with gr.Tab('📚 1단계 · 선행연구 탐색'):
            gr.Markdown('### 🔍 학술 논문 검색 결과')
            gr.Markdown(
                '_Semantic Scholar + OpenAlex — 2억+ 논문 실시간 검색 '
                '· 한·영 이중 임베딩 재순위화 · API 키 불필요_'
            )
            out_papers = gr.Dataframe(
                headers=['출처', '논문 제목', '저자', '연도', '초록(요약)', '원문 링크'],
                wrap=True, row_count=6,
            )
            gr.Markdown('---')
            gr.Markdown('### 📂 기존 R&E 유사 연구 사례')
            gr.Markdown(
                '_전국 R&E 전체 1,357건 데이터베이스 검색 '
                '(대학 협업 여부 무관) · 분야 필터 적용 가능_'
            )
            out_rne = gr.Dataframe(
                headers=['연도', '분야', '제목', '소속고등학교', '주제어', '유사도(%)'],
                wrap=True, row_count=5,
            )

        # ── 탭 2: AI 피드백 ──────────────────────
        with gr.Tab('🤖 2단계 · AI 피드백'):
            gr.Markdown('### Groq AI (Llama 3.3 70B)의 연구 주제 분석')
            gr.Markdown(
                '_선행연구와 R&E 사례를 바탕으로 주제의 참신성 · 실현 가능성 · 개선 방향을 분석합니다._'
            )
            out_feedback = gr.Markdown()

        # ── 탭 3: 기관 적합도 ────────────────────
        with gr.Tab('🏛 3단계 · 기관 적합도'):
            gr.Markdown('### 대구·경북 연구기관 적합도 순위')
            gr.Markdown(
                '_가중치: 연구분야 40% + RnE실적 25% + 기자재 20% + 연구재단역량 10% + 분야가산점 5%_  \n'
                '_Rank 백분위 정규화 + Softmax(T=4.0) → 균형 잡힌 점수 분포_'
            )
            out_scores = gr.Dataframe(
                headers=['기관명', '적합도(%)', 'RnE실적수'],
                wrap=True, row_count=13,
            )
            gr.Markdown('### 🎓 1위 기관 자주 협업한 교수 TOP 5')
            gr.Markdown('_실제 R&E 협업 이력 기반 — 연락 가능 교수 참고용_')
            out_prof = gr.Dataframe(
                headers=['교수명', '협업횟수'],
                wrap=True, row_count=5,
            )

        # ── 탭 4: 진행 가이드 ────────────────────
        with gr.Tab('📋 4단계 · RnE 진행 가이드'):
            gr.Markdown('### 적합도 1위 기관 기준 단계별 RnE 진행 절차')
            gr.Markdown('_신청 준비 → 교수 섭외 → 연구 수행 → 결과 발표까지 단계별로 안내합니다._')
            out_guide = gr.Markdown()

        # ── 탭 5: 연구 계획서 ────────────────────
        with gr.Tab('📝 5단계 · 연구 계획서 초안'):
            gr.Markdown('### R&E 신청용 연구 계획서 자동 생성')
            gr.Markdown(
                '_선행연구 · AI피드백 · 추천 기관 정보를 종합해 '
                'R&E 신청에 바로 활용 가능한 계획서 초안을 작성합니다._'
            )
            out_plan = gr.Markdown()

    # 다운로드
    gr.Markdown('---')
    with gr.Row():
        gr.Markdown('### 💾 전체 분석 결과 다운로드')
        out_dl = gr.File(label='결과 저장 (txt 파일)')

    # 푸터
    gr.HTML("""
    <div style="text-align:center;padding:18px 0 10px;margin-top:12px;
                border-top:1px solid #e2e8f0;color:#94a3b8;font-size:0.80rem;line-height:1.9">
      데이터 출처: 한국연구재단 R&E 성과자료집(2020~2025) · Semantic Scholar · OpenAlex ·
      대학알리미 · 중소기업기술정보진흥원 · 한국연구재단<br>
      <b>공공데이터 AI 활용 분석 대회 출품작</b>
    </div>
    """)

    # 이벤트 연결
    _outputs = [out_papers, out_rne, out_feedback, out_scores,
                out_guide, out_plan, out_dl, out_prof]
    submit_btn.click(fn=run_analysis, inputs=[topic_box, field_sel], outputs=_outputs)
    topic_box.submit(fn=run_analysis, inputs=[topic_box, field_sel], outputs=_outputs)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    demo.launch(server_name='0.0.0.0', server_port=port)
