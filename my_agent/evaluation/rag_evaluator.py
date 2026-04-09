"""
RAG 系统自动评估台 (LLM-as-a-Judge)

针对 RAG 系统的检索（Retrieval）与生成（Generation）环节做全方位体检：
- 检索指标：命中率 (Hit Rate)、平均倒数排名 (MRR)
- 生成指标：忠实度 (Faithfulness) / 幻觉检测
- 端到端指标：相关性 (Relevancy)、正确率 (Correctness)

利用大模型作为评估器，对 RAG 系统进行自动化打分。
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from loguru import logger
import asyncio

from my_agent.services.embedding_service import DashScopeEmbeddingService
from my_agent.services.rerank_service import DashScopeRerankService
from my_agent.core.llm_service import LLMService
from my_agent.core.milvus_manager import MilvusManager

# ===== 评测数据结构 =====
@dataclass
class EvalCase:
    query: str
    expected_answer: str
    relevant_chunk_ids: List[str]
    intent: str

@dataclass
class ScoreResult:
    score: int     # 1-5分
    label: str     # 分类标签 (例如: faithful, partially_faithful 等)
    reason: str    # 评分理由

@dataclass
class EvalResult:
    eval_case: EvalCase
    retrieved_chunk_ids: List[str]
    actual_answer: str
    hit: bool
    reciprocal_rank: float
    recall: float = 0.0
    precision: float = 0.0
    faithfulness: Optional[ScoreResult] = None
    relevancy: Optional[ScoreResult] = None
    correctness: Optional[ScoreResult] = None

class RAGEvaluator:
    def __init__(self, override_judge_model: str = "qwen-plus"):
        # 评测使用 阿里百炼 qwen-plus 作为稳健裁判
        self.llm = LLMService()
        self.llm.model_name = override_judge_model

    # ========== LLM 测评员引擎 ==========
    async def _llm_score(self, prompt: str) -> ScoreResult:
        """调用大模型作为高冷法官进行强格式化打分"""
        messages = [{"role": "user", "content": prompt}]
        # 严格设定极低温度，防范主观随意评判
        result = self.llm.evaluate_tools(messages, available_tools=[], temperature=0.1)
        
        if result["type"] != "text":
            logger.error("评分模型失控触发了工具！这在评测系统中是绝对灾难。")
            return ScoreResult(0, "error", "模型异常动作发散")
            
        content = result["content"].strip()
        # 清除大模型带出的高亮 Markdown 框残渣
        clean_content = content.replace("```json", "").replace("```", "").strip()
        
        try:
            # 高鲁棒性切片：只抓取首尾 {} 之间的内容
            start = clean_content.find("{")
            end = clean_content.rfind("}") + 1
            if start >= 0 and end > start:
                clean_content = clean_content[start:end]
                
            data = json.loads(clean_content)
            return ScoreResult(
                score=int(data.get("score", 0)),
                label=data.get("label", "unknown"),
                reason=data.get("reason", "no reason provided")
            )
        except Exception as e:
            logger.error(f"🚨 [评测台] JSON 解析因模型大喘气产生崩塌: {e}. 原始内容切片: {content[:100]}...")
            return ScoreResult(0, "parse_error", "大模型输出格式错乱")

    # ========== 三大生死维度打分项 (忠实、相关、正确) ==========
    async def score_faithfulness(self, chunks: str, answer: str) -> ScoreResult:
        """【生成维度】 忠实度（抓捕最恶劣的大模型发散造谣与幻觉）"""
        prompt = (
            "你是一个专业的 RAG 系统评估员。你的任务是评估模型的回答是否忠实于给定的参考文档内容。\n\n"
            "评分标准：\n"
            "- 5 分：回答完全基于参考文档，没有添加任何文档中没有的信息\n"
            "- 4 分：回答基本基于参考文档，有极少量合理推断但不影响准确性\n"
            "- 3 分：回答部分基于参考文档，但添加了一些文档中没有的信息\n"
            "- 2 分：回答包含较多文档中没有的信息，存在明显编造\n"
            "- 1 分：回答与参考文档内容严重不符或大量编造\n\n"
            f"参考文档内容：\n{chunks}\n\n"
            f"模型的回答：\n{answer}\n\n"
            "请按以下 JSON 格式输出评分结果，绝对不要输出任何无关紧要的内容：\n"
            '{"score": <1-5的整数>, "label": "<faithful/partially_faithful/unfaithful>", '
            '"reason": "<简要说明评分理由>"}'
        )
        return await self._llm_score(prompt)

    async def score_relevancy(self, query: str, answer: str) -> ScoreResult:
        """【生成维度】 相关性（抓捕敷衍与答非所问的情况）"""
        prompt = (
            "你是一个专业的 RAG 系统评估员。你的任务是评估模型的回答是否回答了用户的问题。\n\n"
            "评分标准：\n"
            "- 5 分：直接、完整地回答了用户的问题\n"
            "- 4 分：回答了用户的问题，但不够完整或包含了多余信息\n"
            "- 3 分：部分回答了用户的问题，但遗漏了关键信息\n"
            "- 2 分：回答与用户的问题有关，但没有真正回答问题\n"
            "- 1 分：回答与用户的问题完全无关\n\n"
            f"用户问题：\n{query}\n\n"
            f"模型的回答：\n{answer}\n\n"
            "请按以下 JSON 格式输出评分结果，不要输出其他前导后缀：\n"
            '{"score": <1-5的整数>, "label": "<relevant/partially_relevant/irrelevant>", '
            '"reason": "<简要说明评分理由>"}'
        )
        return await self._llm_score(prompt)

    async def score_correctness(self, query: str, expected_answer: str, actual_answer: str) -> ScoreResult:
        """【端到端维度】 正确率（决定 RAG 系统能不能上线大考的终极分）"""
        prompt = (
            "你是一个专业的 RAG 系统评估员。你的任务是评估模型的回答是否正确。\n\n"
            "评分标准：\n"
            "- 5 分：回答与标准答案的含义完全一致\n"
            "- 4 分：回答与标准答案基本一致，核心信息正确，细节略有差异\n"
            "- 3 分：回答部分正确，但遗漏或错误了一些重要信息\n"
            "- 2 分：回答包含正确信息，但主要结论有误\n"
            "- 1 分：回答与标准答案完全不一致\n\n"
            f"用户问题：\n{query}\n\n"
            f"标准答案：\n{expected_answer}\n\n"
            f"模型的回答：\n{actual_answer}\n\n"
            "请按以下 JSON 格式严格输出评分结果，不可存在其他字符：\n"
            '{"score": <1-5的整数>, "label": "<correct/partially_correct/incorrect>", '
            '"reason": "<简要说明评分理由>"}'
        )
        return await self._llm_score(prompt)

    # ========== 检索硬核算数指标 ==========
    @staticmethod
    def calculate_hit(retrieved_ids: List[str], relevant_ids: List[str]) -> bool:
        """计算命中率(Hit Rate)：Top-K 里有没有在千军万马里捞到哪怕一块真正的关键碎片"""
        if not relevant_ids:
            return False 
        return any(rid in retrieved_ids for rid in relevant_ids)

    @staticmethod
    def calculate_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """计算平均倒数排名(MRR)：正确的金块到底藏得有多深，排名越靠前越好"""
        if not relevant_ids:
            return 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """计算召回率(Recall)：所有的标准答案里，你捞到了百分之几"""
        if not relevant_ids:
            # 如果这题压根没答案（比如库外提问），不纳入指标计算惩罚
            return 0.0
        hit_count = sum(1 for rid in relevant_ids if rid in retrieved_ids)
        return hit_count / len(relevant_ids)

    @staticmethod
    def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """计算精确率(Precision)：你捞出来的金矿里，有多少是真金"""
        if not retrieved_ids:
            return 0.0
        hit_count = sum(1 for rid in retrieved_ids if rid in relevant_ids)
        return hit_count / len(retrieved_ids)

    # ========== 实验室病历准备区 ==========
    @staticmethod
    def build_eval_dataset() -> List[EvalCase]:
        return [
            EvalCase("iPhone 16 Pro 的退货政策是什么？", "iPhone 16 Pro 支持 7 天无理由退货，需保持商品完好、配件齐全、包装完整。退货运费由买家承担，质量问题由卖家承担运费。", ["chunk_12", "chunk_13"], "knowledge"),
            EvalCase("AirPods Pro 的保修期是多久？", "AirPods Pro 保修期为 1 年，自购买之日起计算。保修范围包括硬件故障和制造缺陷，不包括人为损坏和进水。", ["chunk_21"], "knowledge"),
            EvalCase("退货运费谁承担？", "正常退货运费由买家承担。如果是商品质量问题导致的退货，运费由卖家承担。", ["chunk_13"], "knowledge"),
            EvalCase("跨境商品能退货吗？", "跨境商品支持退货，但需要在签收后 7 天内提出。退货运费由买家承担，且需要自行办理退货物流。部分商品可能不支持退货，以商品详情页说明为准。", ["chunk_35", "chunk_36"], "knowledge"),
            EvalCase("质量问题怎么换货？", "质量问题换货流程：1. 在订单详情页提交换货申请并上传质量问题照片 2. 等待客服审核（1-2 个工作日）3. 审核通过后寄回商品，运费由卖家承担 4. 收到商品后 3 个工作日内寄出新商品。", ["chunk_08", "chunk_09"], "knowledge"),
            EvalCase("Apple Watch Ultra 的防水等级是多少？", "抱歉，当前知识库中没有找到 Apple Watch Ultra 防水等级的相关信息。建议您查看商品详情页或联系人工客服获取准确信息。", [], "knowledge")
        ]

    # ========== Sandbox 模拟区 ==========
    @staticmethod
    def simulate_retrieval() -> Dict[str, List[str]]:
        """截断模拟：冒充向量数据库干出来的召回流水"""
        return {
            "iPhone 16 Pro 的退货政策是什么？": ["chunk_12", "chunk_05", "chunk_33"],
            "AirPods Pro 的保修期是多久？": ["chunk_21", "chunk_07", "chunk_44"],
            "退货运费谁承担？": ["chunk_18", "chunk_29", "chunk_55"], # 验证致命漏判: 没捞到13
            "跨境商品能退货吗？": ["chunk_35", "chunk_03", "chunk_67"], # 只碰壁命中了35
            "质量问题怎么换货？": ["chunk_08", "chunk_15", "chunk_22"], # 命中了08
            "Apple Watch Ultra 的防水等级是多少？": ["chunk_50", "chunk_51", "chunk_52"] # 库里本身没相关知识
        }

    @staticmethod
    def simulate_generation() -> Dict[str, str]:
        """截断模拟：冒充最终的大模型端出盘子的生成结果"""
        return {
            "iPhone 16 Pro 的退货政策是什么？": "iPhone 16 Pro 支持 7 天无理由退货，需要保持商品完好、配件齐全。退货运费由买家承担，质量问题运费由卖家承担。",
            "AirPods Pro 的保修期是多久？": "AirPods Pro 的保修期为 1 年，从购买日期开始计算。保修覆盖硬件故障和制造缺陷，人为损坏和进水不在保修范围内。",
            "退货运费谁承担？": "一般情况下退货运费由买家自行承担。不过如果是因为商品本身的质量问题需要退货，运费会由卖家来承担。",
            "跨境商品能退货吗？": "跨境商品可以退货，需要在签收后 7 天内申请。退货运费由买家承担。需要注意的是，跨境退货支持全球免费上门取件服务。", # 测试严重幻觉事故：生造"全球免费上门"
            "质量问题怎么换货？": "质量问题换货步骤：1. 提交换货申请并上传照片 2. 等待审核 1-2 个工作日 3. 寄回商品（运费卖家承担）4. 收到后 3 个工作日寄出新商品。",
            "Apple Watch Ultra 的防水等级是多少？": "抱歉，目前没有找到 Apple Watch Ultra 防水等级的相关信息，建议您查看商品详情页或联系人工客服确认。"
        }
    
    # ========== 核心评分发卷区 ==========
    async def evaluate_mock_run(self) -> List[EvalResult]:
        """在 Sandbox 中跑一遍这套题库，并让无情法官出结果"""
        dataset = self.build_eval_dataset()
        retrieval_mock = self.simulate_retrieval()
        generation_mock = self.simulate_generation()
        
        results = []
        for i, case in enumerate(dataset):
            logger.info(f"== 正在对第 {i+1} 题上法庭审判进行三围体检: {case.query[:12]}... ==")
            retrieved_ids = retrieval_mock.get(case.query, [])
            actual_ans = generation_mock.get(case.query, "")
            
            hit = self.calculate_hit(retrieved_ids, case.relevant_chunk_ids)
            rr = self.calculate_reciprocal_rank(retrieved_ids, case.relevant_chunk_ids)
            recall = self.calculate_recall(retrieved_ids, case.relevant_chunk_ids)
            precision = self.calculate_precision(retrieved_ids, case.relevant_chunk_ids)
            
            # 制造假体碎片文本文本，因为不直连真向量库因此硬塞一个假文案代替 chunks文本
            dummy_chunks_str = f"提取的知识块内容ID矩阵 {retrieved_ids}，包含对应的条款内容。"
            
            faith_score = await self.score_faithfulness(dummy_chunks_str, actual_ans)
            rel_score = await self.score_relevancy(case.query, actual_ans)
            corr_score = await self.score_correctness(case.query, case.expected_answer, actual_ans)
            
            results.append(EvalResult(
                eval_case=case, retrieved_chunk_ids=retrieved_ids, actual_answer=actual_ans,
                hit=hit, reciprocal_rank=rr, recall=recall, precision=precision,
                faithfulness=faith_score, relevancy=rel_score, correctness=corr_score
            ))
            
        return results

    # ========== 终端打印仪印发报告 ==========
    @staticmethod
    def print_eval_report(results: List[EvalResult]):
        print("=" * 70)
        print("                 RAG 企业级系统评估质检大报 (Python端)")
        print("=" * 70)

        # 打通检索统计
        retrieval_cases = [r for r in results if r.eval_case.relevant_chunk_ids]
        hit_count = sum(1 for r in retrieval_cases if r.hit)
        hit_rate = hit_count / len(retrieval_cases) if retrieval_cases else 0.0
        mrr = sum(r.reciprocal_rank for r in retrieval_cases) / len(retrieval_cases) if retrieval_cases else 0.0
        avg_recall = sum(r.recall for r in retrieval_cases) / len(retrieval_cases) if retrieval_cases else 0.0
        avg_precision = sum(r.precision for r in retrieval_cases) / len(retrieval_cases) if retrieval_cases else 0.0

        print("\n【检索阶能力指标 (Retrieval)】")
        print(f"  命中率（Hit Rate）：{hit_rate * 100:.1f}%（{hit_count} / {len(retrieval_cases)}）")
        print(f"  MRR（平均排名）：{mrr:.3f} (金矿藏得多深)")
        print(f"  召回率（Recall）：{avg_recall * 100:.1f}% (库里本身有多少好矿，捞全了没有)")
        print(f"  精确率（Precision）：{avg_precision * 100:.1f}% (捞出的这一堆盲盒里，真金子的浓度)")

        # 打通生成统计
        avg_faith = sum(r.faithfulness.score for r in results if r.faithfulness) / len(results) if results else 0
        avg_rel = sum(r.relevancy.score for r in results if r.relevancy) / len(results) if results else 0
        hallucination_count = sum(1 for r in results if r.faithfulness and r.faithfulness.score <= 2)
        hallucination_rate = hallucination_count / len(results) if results else 0

        print("\n【生成阶文采与保真指标 (Generation)】")
        print(f"  忠实度平均分：{avg_faith:.2f} / 5.0")
        print(f"  相关性平均分：{avg_rel:.2f} / 5.0")
        print(f"  恶性幻觉病发率：{hallucination_rate * 100:.1f}%（系统查出 {hallucination_count} / {len(results)} 条存在发散胡诌）")

        # 端到端交付大考
        avg_corr = sum(r.correctness.score for r in results if r.correctness) / len(results) if results else 0
        correct_count = sum(1 for r in results if r.correctness and r.correctness.score >= 4)
        correct_rate = correct_count / len(results) if results else 0
        
        fallback_keywords = ["抱歉", "找不到", "没有找到"]
        fallback_count = sum(1 for r in results if any(k in r.actual_answer for k in fallback_keywords))
        fallback_rate = fallback_count / len(results) if results else 0

        print("\n【端到端验收总览表 (End-to-End)】")
        print(f"  综合正确率均值：{avg_corr:.2f} / 5.0")
        print(f"  成品妥投率（正确评分 ≥4 分）：{correct_rate * 100:.1f}%（{correct_count} / {len(results)}）")
        print(f"  安全兜底告急率：{fallback_rate * 100:.1f}%（{fallback_count} / {len(results)}）")

        # 病灶彻查追责模块 (Bad Case RCA)
        print("\n【病危追踪清单 (Bad Case，成绩 < 4分)】")
        print("-" * 70)
        has_bad_case = False
        for r in results:
            if r.correctness and r.correctness.score < 4:
                has_bad_case = True
                print(f"  患者(问题)：{r.eval_case.query}")
                print(f"  期望病历：{r.eval_case.expected_answer}")
                print(f"  医嘱(生成)：{r.actual_answer}")
                
                f_score = r.faithfulness.score if r.faithfulness else 0
                r_score = r.relevancy.score if r.relevancy else 0
                c_score = r.correctness.score
                print(f"  检索命中：{'是' if r.hit else '否'} | 忠实度得分：{f_score} | 相关度得分：{r_score} | 终盘正确率：{c_score}")
                
                # 开始追责归因
                if not r.hit:
                    print("  → 刑侦归因：【第一嫌疑人】底层向量库检索失败导致前线断炊缺粮！没有给它任何正确线索！")
                elif r.faithfulness and r.faithfulness.score <= 3:
                    print("  → 刑侦归因：【第二嫌疑人】素材库给准了，但大模型犯了造谣罪或发散罪（甚至违背原意凭空捏造）！")
                else:
                    print("  → 刑侦归因：【第三嫌疑人】素材全命中，模型也没造假，但底库里的知识本身大概率是缺失、过时或者有坑的旧规则。")
                print("-" * 70)
                
        if not has_bad_case:
            print("  🏆 完美收官！未查出任何 Bad Case，所有防浪测试中正确率均 ≥ 4分！")

        print("\n" + "=" * 70)

async def run_eval_mock():
    # 模拟入口
    evaluator = RAGEvaluator()
    reports = await evaluator.evaluate_mock_run()
    evaluator.print_eval_report(reports)

if __name__ == "__main__":
    asyncio.run(run_eval_mock())
