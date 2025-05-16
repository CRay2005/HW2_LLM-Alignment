import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any
from collections import Counter

def analyze_dpo_results(file_path: str):
    """分析DPO评估结果文件并生成可视化分析"""
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Total samples loaded: {len(results)}")
    
    # 提取得分数据
    base_scores = [item['base_score'] for item in results]
    dpo_scores = [item['dpo_score'] for item in results]
    score_diff = [dpo - base for dpo, base in zip(dpo_scores, base_scores)]
    
    # 创建数据框
    df = pd.DataFrame({
        'base_score': base_scores,
        'dpo_score': dpo_scores,
        'score_diff': score_diff,
        'prompts': [item['prompt'] for item in results],
        'base_response': [item['base_output'] for item in results],
        'dpo_response': [item['dpo_output'] for item in results]
    })
    
    # 1. 得分差异性分析
    print("\n===== Score Statistical Analysis =====")
    print(f"Base Model Average Score: {np.mean(base_scores):.4f} ± {np.std(base_scores):.4f}")
    print(f"DPO Model Average Score: {np.mean(dpo_scores):.4f} ± {np.std(dpo_scores):.4f}")
    print(f"Average Score Difference: {np.mean(score_diff):.4f}")
    print(f"Percentage of samples where DPO outperforms base model: {(np.array(score_diff) > 0).mean():.2%}")
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 10))
    
    # 创建2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 1.1 两个模型得分的散点图比较
    axs[0, 0].scatter(base_scores, dpo_scores, alpha=0.6)
    axs[0, 0].plot([min(base_scores), max(base_scores)], 
                   [min(base_scores), max(base_scores)], 'r--')
    axs[0, 0].set_xlabel('Base Model Score')
    axs[0, 0].set_ylabel('DPO Model Score')
    axs[0, 0].set_title('Model Score Comparison Scatter Plot')
    
    # 1.2 得分差异直方图
    axs[0, 1].hist(score_diff, bins=30, alpha=0.7, color='skyblue')
    axs[0, 1].axvline(x=0, color='red', linestyle='--')
    axs[0, 1].set_xlabel('Score Difference (DPO - Base)')
    axs[0, 1].set_ylabel('Sample Count')
    axs[0, 1].set_title('Score Difference Distribution')
    
    # 1.3 累积分布函数(CDF)对比
    axs[1, 0].hist(base_scores, bins=30, alpha=0.5, color='blue', 
                   density=True, cumulative=True, label='Base Model')
    axs[1, 0].hist(dpo_scores, bins=30, alpha=0.5, color='orange', 
                   density=True, cumulative=True, label='DPO Model')
    axs[1, 0].set_xlabel('Score')
    axs[1, 0].set_ylabel('Cumulative Probability')
    axs[1, 0].set_title('Score Cumulative Distribution Function')
    axs[1, 0].legend()
    
    # 1.4 箱线图比较
    box_data = [base_scores, dpo_scores]
    axs[1, 1].boxplot(box_data, labels=['Base Model', 'DPO Model'])
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].set_title('Model Score Boxplot')
    
    # 添加统计信息到第四个图上
    stats_text = (
        f"Base Model Avg: {np.mean(base_scores):.4f} ± {np.std(base_scores):.4f}\n"
        f"DPO Model Avg: {np.mean(dpo_scores):.4f} ± {np.std(dpo_scores):.4f}\n"
        f"Avg Difference: {np.mean(score_diff):.4f}\n"
        f"DPO Win Rate: {(np.array(score_diff) > 0).mean():.2%}"
    )
    # 将文本放在两根箱线图中间的空白处靠上的位置，文字左对齐
    axs[1, 1].text(0.35, 0.85, stats_text, transform=axs[1, 1].transAxes, 
                   ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('dpo_score_comparison.png', dpi=300)
    
    # 2. 性能分析 - 找出DPO模型表现更好和更差的案例
    print("\n===== Performance Analysis =====")
    
    # 按得分差异排序
    df_sorted = df.sort_values(by='score_diff', ascending=False).reset_index(drop=True)
    
    # 2.1 DPO模型表现最好的案例
    print("\n----- Top 5 Cases where DPO Outperforms Base Model -----")
    best_cases = df_sorted.head(5)
    for i, row in best_cases.iterrows():
        print(f"Sample {i+1}:")
        print(f"Score Difference: {row['score_diff']:.4f}")
        print(f"Base Model Score: {row['base_score']:.4f}")
        print(f"DPO Model Score: {row['dpo_score']:.4f}")
        print(f"Prompt: {row['prompts'][:100]}...")
        print(f"Base Model Response: {row['base_response'][:100]}...")
        print(f"DPO Model Response: {row['dpo_response'][:100]}...")
        print("-" * 50)
    
    # 2.2 DPO模型表现最差的案例
    print("\n----- Bottom 5 Cases where DPO Underperforms Base Model -----")
    worst_cases = df_sorted.tail(5).iloc[::-1]
    for i, row in worst_cases.iterrows():
        print(f"Sample {i+1}:")
        print(f"Score Difference: {row['score_diff']:.4f}")
        print(f"Base Model Score: {row['base_score']:.4f}")
        print(f"DPO Model Score: {row['dpo_score']:.4f}")
        print(f"Prompt: {row['prompts'][:100]}...")
        print(f"Base Model Response: {row['base_response'][:100]}...")
        print(f"DPO Model Response: {row['dpo_response'][:100]}...")
        print("-" * 50)
    
    # 3. 响应长度分析
    print("\n===== Response Length Analysis =====")
    df['base_len'] = df['base_response'].apply(len)
    df['dpo_len'] = df['dpo_response'].apply(len)
    df['len_diff'] = df['dpo_len'] - df['base_len']
    
    print(f"Base Model Average Response Length: {df['base_len'].mean():.2f} characters")
    print(f"DPO Model Average Response Length: {df['dpo_len'].mean():.2f} characters")
    print(f"Average Length Difference: {df['len_diff'].mean():.2f} characters")
    
    # 长度与分数的相关性
    base_len_score_corr = np.corrcoef(df['base_len'], df['base_score'])[0, 1]
    dpo_len_score_corr = np.corrcoef(df['dpo_len'], df['dpo_score'])[0, 1]
    
    print(f"Base Model Length-Score Correlation: {base_len_score_corr:.4f}")
    print(f"DPO Model Length-Score Correlation: {dpo_len_score_corr:.4f}")
    
    # 绘制长度与分数的散点图
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['base_len'], df['base_score'], alpha=0.6, label='Base Model')
    plt.xlabel('Response Length')
    plt.ylabel('Score')
    plt.title('Base Model: Response Length vs Score')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['dpo_len'], df['dpo_score'], alpha=0.6, color='orange', label='DPO Model')
    plt.xlabel('Response Length')
    plt.ylabel('Score')
    plt.title('DPO Model: Response Length vs Score')
    
    plt.tight_layout()
    plt.savefig('response_length_analysis.png', dpi=300)
    
    # 4. 内容分析 - 找出模型改进的方面
    print("\n===== Content Analysis =====")
    
    # 分析提高分数的案例中，DPO模型的主要改进点
    improved_cases = df[df['score_diff'] > 0]
    print(f"Number of cases where DPO outperforms base model: {len(improved_cases)}")
    
    # 简单的文本分析，统计关键词出现频率
    # 这里只是一个简单示例，实际应用中可能需要更复杂的NLP分析
    def extract_keywords(text, min_length=2):
        """提取文本中的关键词"""
        words = text.lower().split()
        return [word for word in words if len(word) >= min_length]
    
    base_keywords = []
    dpo_keywords = []
    
    for _, row in improved_cases.iterrows():
        base_keywords.extend(extract_keywords(row['base_response']))
        dpo_keywords.extend(extract_keywords(row['dpo_response']))
    
    base_word_counts = Counter(base_keywords).most_common(20)
    dpo_word_counts = Counter(dpo_keywords).most_common(20)
    
    print("\nBase Model Common Keywords:")
    for word, count in base_word_counts:
        print(f"{word}: {count}")
    
    print("\nDPO Model Common Keywords:")
    for word, count in dpo_word_counts:
        print(f"{word}: {count}")
    
    # 5. 保存分析结果
    df.to_csv('dpo_analysis_results.csv', index=False)
    print("\nAnalysis results saved to dpo_analysis_results.csv")
    print("Visualization charts saved to dpo_score_comparison.png and response_length_analysis.png")

if __name__ == "__main__":
    file_path = "result/dpo_eval_results.json"
    analyze_dpo_results(file_path)