#!/usr/bin/env python3
"""
测试问题生成系统
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'practice_tools'))

from question_generator import QuestionGenerator

def test_question_generator():
    """测试问题生成器功能"""
    print("🧪 测试问题生成系统")
    print("=" * 50)
    
    # 初始化生成器
    generator = QuestionGenerator("/Users/hanguangshuai/Dropbox/Documents/Job/ML design")
    
    # 扫描可用问题
    print("📂 扫描可用问题...")
    questions = generator.scan_questions()
    
    print("📋 可用问题分类:")
    for category, question_list in questions.items():
        print(f"  {category}: {len(question_list)} 个问题")
        for question in question_list[:3]:  # 显示前3个
            print(f"    - {question}")
        if len(question_list) > 3:
            print(f"    ... 还有 {len(question_list) - 3} 个问题")
        print()
    
    # 测试随机问题生成
    print("🎲 测试随机问题生成...")
    for i in range(3):
        question = generator.get_random_question()
        if "error" not in question:
            print(f"\n第 {i+1} 题:")
            print(f"分类: {question['category']}")
            print(f"问题: {question['question_name']}")
            print(f"内容预览: {question['question_content'][:100]}...")
        else:
            print(f"错误: {question['error']}")
    
    # 测试特定分类
    print("\n🎯 测试特定分类 (basic_ml)...")
    question = generator.get_random_question(category="basic_ml")
    if "error" not in question:
        print(f"分类: {question['category']}")
        print(f"问题: {question['question_name']}")
        print(f"内容: {question['question_content']}")
        
        if question['answer_content']:
            print(f"\n答案预览: {question['answer_content'][:200]}...")
        else:
            print("\n❌ 没有找到对应答案")
    else:
        print(f"错误: {question['error']}")
    
    print("\n✅ 测试完成!")

if __name__ == "__main__":
    test_question_generator()
