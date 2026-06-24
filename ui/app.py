"""
Prompt 注意力诊断器 — Streamlit Web UI

双标签页交互界面：
1. Prompt 分析页 — 熵变率热力图 + Token 高亮
2. 长文本扫描页 — 块级范数扫描 + 弱信号标注

启动方式：
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompt_linter import PromptLinter
from prompt_linter.prompt_linter import InputEmptyError, InputTooLongError


# ── 颜色常量 ──────────────────────────────────────────────────────

RISK_COLOR_HIGH = "#ff4444"
RISK_COLOR_MEDIUM = "#ffaa00"
RISK_COLOR_LOW = "#44aa44"
RISK_COLOR_WEAK = "#aaaaaa"
BG_COLOR_HIGH = "#ffe0e0"
BG_COLOR_MEDIUM = "#fff3cd"


# ── 页面配置 ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prompt 注意力诊断器",
    page_icon="🔬",
    layout="wide",
)

# ── 侧边栏 ────────────────────────────────────────────────────────

st.sidebar.title("🔬 Prompt 注意力诊断器")
st.sidebar.markdown("**v0.1 Demo** — 基于 Qwen2.5-0.5B 的静态结构预检工具")

st.sidebar.markdown("---")
st.sidebar.markdown("### 使用说明")
st.sidebar.markdown(
    """
1. 在 **Prompt 分析** 页输入文本
2. 点击 **分析** 按钮
3. 查看 Token 级热力图和风险高亮
4. 切换到 **长文本扫描** 处理 RAG 拼接文本
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 模型信息")
model_info = st.sidebar.empty()
model_info.info("模型未加载，点击分析后自动加载")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ 技术边界")
st.sidebar.warning(
    """
1. **0.5B ≠ 70B** — 高熵不代表大模型会犯错
2. **RAG 归因限制** — 检测表征洼地，非检索漏召
3. **仅限静态分析** — 不判断事实正确性
"""
)


# ── Linter 实例（缓存） ───────────────────────────────────────────

@st.cache_resource
def get_linter():
    return PromptLinter()


# ── 公共分析流程 ──────────────────────────────────────────────────

def _update_model_info(metadata: dict):
    """更新侧边栏模型信息。"""
    model_info.info(
        f"**模型:** {metadata['model_name']}  "
        f"**Tokens:** {metadata['total_tokens']}  "
        f"**耗时:** {metadata['analysis_time_ms']:.0f}ms"
    )


def _display_analysis_results(
    result: dict,
    result_type: str,
    chart_fn,
    extra_display=None,
):
    """公共分析结果展示流程。

    Args:
        result: linter.analyze() 返回的结果 dict
        result_type: "分析" 或 "扫描"（用于成功消息）
        chart_fn: 接收 result 并返回 plotly figure 的函数
        extra_display: 可选，接收 result 的额外展示函数
    """
    meta = result["metadata"]
    _update_model_info(meta)
    st.success(f"{result_type}完成！共 {meta['total_tokens']} Token，耗时 {meta['analysis_time_ms']:.0f}ms")

    st.plotly_chart(chart_fn(result), use_container_width=True)

    if extra_display:
        extra_display(result)


def _safe_analyze(linter, text: str, **kwargs):
    """执行分析并统一处理错误。"""
    try:
        result = linter.analyze(text, **kwargs)
        return result
    except InputTooLongError as e:
        st.error(f"输入过长: {e}")
    except Exception as e:
        st.error(f"分析出错: {e}")
        st.info("首次运行需下载 Qwen2.5-0.5B（约 1GB），请确保网络正常")
    return None


# ── 绘图函数 ──────────────────────────────────────────────────────

def _risk_color(risk_level: str) -> str:
    return {
        "high": RISK_COLOR_HIGH,
        "medium": RISK_COLOR_MEDIUM,
    }.get(risk_level, RISK_COLOR_LOW)


def build_entropy_figure(token_risks: list[dict]) -> go.Figure:
    """构建 Token 级熵变率柱状图。"""
    tokens = [r["token"] for r in token_risks]
    deltas = [r["entropy_delta"] for r in token_risks]
    colors = [_risk_color(r["risk_level"]) for r in token_risks]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(tokens))),
        y=deltas,
        marker_color=colors,
        text=[f"{d:.2f}" for d in deltas],
        textposition="outside",
        hovertemplate="Token: %{customdata}<br>Delta: %{y:.4f}<br>Risk: %{marker.color}",
        customdata=tokens,
    ))
    fig.add_hline(y=2.0, line_dash="dash", line_color=RISK_COLOR_HIGH, annotation_text="高风险 (2.0)")
    fig.add_hline(y=1.5, line_dash="dash", line_color=RISK_COLOR_MEDIUM, annotation_text="中风险 (1.5)")
    fig.update_layout(
        title="Token 级熵变率 (Attention Entropy Delta)",
        xaxis_title="Token 序号",
        yaxis_title="熵差值",
        height=350, hovermode="x", showlegend=False,
    )
    return fig


def highlight_tokens(token_risks: list[dict]) -> str:
    """将高风险 Token 用红色加粗标记，中风险用黄色。"""
    parts = []
    for r in token_risks:
        token = r["token"].replace("▁", " ").replace("<|endoftext|>", "⋯")
        if r["risk_level"] == "high":
            parts.append(
                f'<span style="color:{RISK_COLOR_HIGH};font-weight:bold;background:{BG_COLOR_HIGH}">{token}</span>'
            )
        elif r["risk_level"] == "medium":
            parts.append(
                f'<span style="color:#cc8800;background:{BG_COLOR_MEDIUM}">{token}</span>'
            )
        else:
            parts.append(token)
    return "".join(parts)


def build_norm_figure(chunk_risks: list[dict]) -> go.Figure:
    """构建块级范数扫描图。"""
    scores = [c["norm_score"] for c in chunk_risks]
    colors = [RISK_COLOR_WEAK if c["is_weak"] else RISK_COLOR_LOW for c in chunk_risks]
    threshold = float(np.percentile(scores, 15)) if scores else 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Chunk {c['chunk_index']}" for c in chunk_risks],
        y=scores,
        marker_color=colors,
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        hovertemplate="Chunk %{x}<br>Norm: %{y:.4f}<br>%{customdata}",
        customdata=["⚠️ 弱信号" if c["is_weak"] else "正常" for c in chunk_risks],
    ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color=RISK_COLOR_WEAK,
        annotation_text=f"15% 分位数 ({threshold:.4f})",
    )
    fig.update_layout(
        title="文本块信号强度 (Hidden State L2 Norm)",
        xaxis_title="文本块",
        yaxis_title="平均 L2 范数",
        height=350, hovermode="x", showlegend=False,
    )
    return fig


# ── 各标签页的数据展示 ────────────────────────────────────────────

def _show_entropy_results(result: dict):
    """展示 Prompt 分析页的风险统计 + Token 高亮。"""
    token_risks = result["token_risks"]
    high = sum(1 for r in token_risks if r["risk_level"] == "high")
    medium = sum(1 for r in token_risks if r["risk_level"] == "medium")
    if high:
        st.error(f"🔴 {high} 个高风险 Token")
    if medium:
        st.warning(f"🟡 {medium} 个中风险 Token")
    if not high and not medium:
        st.info("✅ 未发现明显风险")
    st.subheader("风险 Token 高亮")
    st.markdown(highlight_tokens(token_risks), unsafe_allow_html=True)


def _show_norm_results(result: dict):
    """展示长文本扫描页的弱信号检测详情。"""
    chunk_risks = result["chunk_risks"]
    weak = sum(1 for c in chunk_risks if c["is_weak"])
    if weak:
        st.warning(f"⚠️ 发现 {weak}/{len(chunk_risks)} 个弱信号块，可考虑截断或前置")
    else:
        st.info("✅ 未发现明显表征洼地")
    for c in chunk_risks:
        if c["is_weak"]:
            with st.expander(
                f"Chunk {c['chunk_index']} "
                f"(Token {c['start_token']}-{c['end_token']}, "
                f"Norm: {c['norm_score']:.4f})"
            ):
                st.markdown(f"**预览:** {c['text_snippet']}")
                st.markdown("**建议:** 此段信号弱，可考虑删除或前置")


# ── 主页面 ────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["📝 Prompt 分析", "📄 长文本扫描"])


# ═══════════════════════════════════════════════════════════════════
# Tab 1: Prompt 分析
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.header("Prompt 结构健康度分析")
    st.caption("检测注意力熵变率异常位置——句法层面的「逻辑死结」")

    prompt_text = st.text_area("输入 Prompt", placeholder="粘贴你的 Prompt 文本...", height=180)

    if st.button("🔍 分析", type="primary"):
        if not prompt_text.strip():
            st.error("请输入待分析的文本")
        else:
            with st.spinner("加载模型并分析…"):
                result = _safe_analyze(get_linter(), prompt_text)
                if result:
                    _display_analysis_results(
                        result,
                        result_type="分析",
                        chart_fn=lambda r: build_entropy_figure(r["token_risks"]),
                        extra_display=_show_entropy_results,
                    )


# ═══════════════════════════════════════════════════════════════════
# Tab 2: 长文本扫描
# ═══════════════════════════════════════════════════════════════════

with tab2:
    st.header("长文本表征洼地扫描")
    st.caption("检测 hidden state 范数异常区域——模型内部被「压缩坍塌」的语义洼地")

    long_text = st.text_area("输入长文本", placeholder="粘贴 RAG 拼接结果或长文本...", height=180)

    col1, col2 = st.columns([1, 4])
    with col1:
        scan_btn = st.button("📡 扫描", type="primary", use_container_width=True)
    with col2:
        cs = st.number_input("块大小 (Token)", min_value=16, max_value=512, value=128, step=16)

    if scan_btn:
        if not long_text.strip():
            st.error("请输入待扫描的文本")
        else:
            with st.spinner("加载模型并扫描…"):
                result = _safe_analyze(get_linter(), long_text, chunk_size=int(cs))
                if result:
                    _display_analysis_results(
                        result,
                        result_type="扫描",
                        chart_fn=lambda r: build_norm_figure(r["chunk_risks"]),
                        extra_display=_show_norm_results,
                    )
