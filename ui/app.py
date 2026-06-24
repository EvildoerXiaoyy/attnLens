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


# ── 绘图函数 ──────────────────────────────────────────────────────

def build_entropy_figure(token_risks: list[dict]) -> go.Figure:
    """构建 Token 级熵变率柱状图。"""
    tokens = [r["token"] for r in token_risks]
    deltas = [r["entropy_delta"] for r in token_risks]
    colors = []
    for r in token_risks:
        if r["risk_level"] == "high":
            colors.append("#ff4444")
        elif r["risk_level"] == "medium":
            colors.append("#ffaa00")
        else:
            colors.append("#44aa44")

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
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="高风险 (2.0)")
    fig.add_hline(y=1.5, line_dash="dash", line_color="orange", annotation_text="中风险 (1.5)")
    fig.update_layout(
        title="Token 级熵变率 (Attention Entropy Delta)",
        xaxis_title="Token 序号",
        yaxis_title="熵差值",
        height=350,
        hovermode="x",
        showlegend=False,
    )
    return fig


def highlight_tokens(token_risks: list[dict]) -> str:
    """将高风险 Token 用红色加粗标记，中风险用黄色。"""
    parts = []
    for r in token_risks:
        token = r["token"].replace("▁", " ").replace("<|endoftext|>", "⋯")
        if r["risk_level"] == "high":
            parts.append(
                f'<span style="color:red;font-weight:bold;background:#ffe0e0">{token}</span>'
            )
        elif r["risk_level"] == "medium":
            parts.append(
                f'<span style="color:#cc8800;background:#fff3cd">{token}</span>'
            )
        else:
            parts.append(token)
    return "".join(parts)


def build_norm_figure(chunk_risks: list[dict]) -> go.Figure:
    """构建块级范数扫描图。"""
    scores = [c["norm_score"] for c in chunk_risks]
    colors = ["#aaaaaa" if c["is_weak"] else "#44aa44" for c in chunk_risks]
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
        y=threshold, line_dash="dash", line_color="gray",
        annotation_text=f"15% 分位数 ({threshold:.4f})",
    )
    fig.update_layout(
        title="文本块信号强度 (Hidden State L2 Norm)",
        xaxis_title="文本块",
        yaxis_title="平均 L2 范数",
        height=350,
        hovermode="x",
        showlegend=False,
    )
    return fig


# ── 主页面 ────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["📝 Prompt 分析", "📄 长文本扫描"])


# ═══════════════════════════════════════════════════════════════════
# Tab 1: Prompt 分析
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.header("Prompt 结构健康度分析")
    st.caption("检测注意力熵变率异常位置——句法层面的「逻辑死结」")

    prompt_text = st.text_area(
        "输入 Prompt",
        placeholder="粘贴你的 Prompt 文本...",
        height=180,
    )

    if st.button("🔍 分析", type="primary"):
        if not prompt_text.strip():
            st.error("请输入待分析的文本")
        else:
            with st.spinner("加载模型并分析…"):
                try:
                    linter = get_linter()
                    result = linter.analyze(prompt_text)
                    meta = result["metadata"]

                    model_info.info(
                        f"**模型:** {meta['model_name']}  "
                        f"**Tokens:** {meta['total_tokens']}  "
                        f"**耗时:** {meta['analysis_time_ms']:.0f}ms"
                    )
                    st.success(f"分析完成！共 {meta['total_tokens']} Token，耗时 {meta['analysis_time_ms']:.0f}ms")

                    # 风险统计
                    token_risks = result["token_risks"]
                    high = sum(1 for r in token_risks if r["risk_level"] == "high")
                    medium = sum(1 for r in token_risks if r["risk_level"] == "medium")
                    if high:
                        st.error(f"🔴 {high} 个高风险 Token")
                    if medium:
                        st.warning(f"🟡 {medium} 个中风险 Token")
                    if not high and not medium:
                        st.info("✅ 未发现明显风险")

                    # 热力图
                    st.plotly_chart(build_entropy_figure(token_risks), use_container_width=True)

                    # 原文高亮
                    st.subheader("风险 Token 高亮")
                    st.markdown(highlight_tokens(token_risks), unsafe_allow_html=True)

                except InputTooLongError as e:
                    st.error(f"输入过长: {e}")
                except Exception as e:
                    st.error(f"分析出错: {e}")
                    st.info("首次运行需下载 Qwen2.5-0.5B（约 1GB），请确保网络正常")


# ═══════════════════════════════════════════════════════════════════
# Tab 2: 长文本扫描
# ═══════════════════════════════════════════════════════════════════

with tab2:
    st.header("长文本表征洼地扫描")
    st.caption("检测 hidden state 范数异常区域——模型内部被「压缩坍塌」的语义洼地")

    long_text = st.text_area(
        "输入长文本",
        placeholder="粘贴 RAG 拼接结果或长文本...",
        height=180,
    )

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
                try:
                    linter = get_linter()
                    result = linter.analyze(long_text, chunk_size=int(cs))
                    meta = result["metadata"]

                    model_info.info(
                        f"**模型:** {meta['model_name']}  "
                        f"**Tokens:** {meta['total_tokens']}  "
                        f"**耗时:** {meta['analysis_time_ms']:.0f}ms"
                    )
                    st.success(f"扫描完成！共 {meta['total_tokens']} Token，{len(result['chunk_risks'])} 个块")

                    chunk_risks = result["chunk_risks"]
                    weak = sum(1 for c in chunk_risks if c["is_weak"])
                    if weak:
                        st.warning(f"⚠️ 发现 {weak}/{len(chunk_risks)} 个弱信号块，可考虑截断或前置")
                    else:
                        st.info("✅ 未发现明显表征洼地")

                    st.plotly_chart(build_norm_figure(chunk_risks), use_container_width=True)

                    # 弱信号块详情
                    for c in chunk_risks:
                        if c["is_weak"]:
                            with st.expander(
                                f"Chunk {c['chunk_index']} "
                                f"(Token {c['start_token']}-{c['end_token']}, "
                                f"Norm: {c['norm_score']:.4f})"
                            ):
                                st.markdown(f"**预览:** {c['text_snippet']}")
                                st.markdown("**建议:** 此段信号弱，可考虑删除或前置")

                except Exception as e:
                    st.error(f"扫描出错: {e}")
                    st.info("首次运行需下载 Qwen2.5-0.5B（约 1GB），请确保网络正常")
