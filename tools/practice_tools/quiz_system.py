#!/usr/bin/env python3
"""
AI/MLE 面试测验系统
提供更高级的测验功能，包括计时、评分和详细分析
"""

import os
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from question_generator import QuestionGenerator

class QuizSystem:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.generator = QuestionGenerator(project_root)
        self.results_file = self.project_root / "practice_tools" / "quiz_results.json"
        self.results = self._load_results()
        
    def _load_results(self) -> List[Dict]:
        """Load previous quiz results"""
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_results(self):
        """Save quiz results"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
    
    def start_timed_quiz(self, num_questions: int = 10, 
                        time_limit: int = 30) -> Dict:
        """Start a timed quiz session"""
        print(f"⏰ 开始计时测验")
        print(f"📝 题目数量: {num_questions}")
        print(f"⏱️  时间限制: {time_limit} 分钟")
        print("=" * 50)
        
        quiz_questions = self.generator.generate_quiz(num_questions)
        if not quiz_questions:
            return {"error": "No questions available"}
        
        start_time = time.time()
        answers = []
        correct_count = 0
        
        for i, question in enumerate(quiz_questions, 1):
            print(f"\n📝 第 {i}/{num_questions} 题")
            print(f"📚 分类: {question['category']}")
            print(f"📖 问题: {question['question_name']}")
            print("-" * 50)
            print(question['question_content'])
            
            # Get user answer
            user_answer = input("\n💭 你的答案 (输入 'skip' 跳过): ").strip()
            
            if user_answer.lower() == 'skip':
                answers.append({
                    "question": question['question_name'],
                    "category": question['category'],
                    "user_answer": "SKIPPED",
                    "correct": False,
                    "time_spent": 0
                })
                continue
            
            # Show correct answer
            print("\n💡 正确答案:")
            print("-" * 50)
            print(question['answer_content'])
            
            # Get user's self-assessment
            while True:
                self_assessment = input("\n🤔 你认为自己答对了吗? (y/n): ").strip().lower()
                if self_assessment in ['y', 'yes', 'n', 'no']:
                    break
                print("请输入 y 或 n")
            
            is_correct = self_assessment in ['y', 'yes']
            if is_correct:
                correct_count += 1
            
            answers.append({
                "question": question['question_name'],
                "category": question['category'],
                "user_answer": user_answer,
                "correct": is_correct,
                "time_spent": time.time() - start_time
            })
            
            # Check time limit
            elapsed_time = (time.time() - start_time) / 60
            if elapsed_time >= time_limit:
                print(f"\n⏰ 时间到！已用时 {elapsed_time:.1f} 分钟")
                break
        
        total_time = time.time() - start_time
        score = (correct_count / len(answers)) * 100 if answers else 0
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(answers),
            "correct_answers": correct_count,
            "score": score,
            "time_spent": total_time,
            "answers": answers
        }
        
        self.results.append(result)
        self._save_results()
        
        # Display results
        self._display_results(result)
        
        return result
    
    def adaptive_quiz(self, target_difficulty: str = "medium") -> Dict:
        """Adaptive quiz that adjusts difficulty based on performance"""
        print(f"🎯 自适应测验 - 目标难度: {target_difficulty}")
        print("=" * 50)
        
        # This is a simplified version - in practice, you'd implement
        # more sophisticated difficulty adjustment
        categories = ["basic_ml", "deep_learning", "statistics"]
        questions_per_category = 2
        
        all_questions = []
        for category in categories:
            questions = self.generator.generate_quiz(questions_per_category, [category])
            all_questions.extend(questions)
        
        random.shuffle(all_questions)
        
        answers = []
        correct_count = 0
        
        for i, question in enumerate(all_questions, 1):
            print(f"\n📝 第 {i}/{len(all_questions)} 题")
            print(f"📚 分类: {question['category']}")
            print(f"📖 问题: {question['question_name']}")
            print("-" * 50)
            print(question['question_content'])
            
            user_answer = input("\n💭 你的答案: ").strip()
            
            print("\n💡 正确答案:")
            print("-" * 50)
            print(question['answer_content'])
            
            while True:
                self_assessment = input("\n🤔 你认为自己答对了吗? (y/n): ").strip().lower()
                if self_assessment in ['y', 'yes', 'n', 'no']:
                    break
            
            is_correct = self_assessment in ['y', 'yes']
            if is_correct:
                correct_count += 1
            
            answers.append({
                "question": question['question_name'],
                "category": question['category'],
                "user_answer": user_answer,
                "correct": is_correct
            })
        
        score = (correct_count / len(answers)) * 100 if answers else 0
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "quiz_type": "adaptive",
            "target_difficulty": target_difficulty,
            "total_questions": len(answers),
            "correct_answers": correct_count,
            "score": score,
            "answers": answers
        }
        
        self.results.append(result)
        self._save_results()
        
        self._display_results(result)
        return result
    
    def category_focused_quiz(self, category: str, num_questions: int = 5) -> Dict:
        """Focus on a specific category"""
        print(f"🎯 {category} 专项测验")
        print(f"📝 题目数量: {num_questions}")
        print("=" * 50)
        
        questions = self.generator.generate_quiz(num_questions, [category])
        if not questions:
            return {"error": f"No questions found in category: {category}"}
        
        answers = []
        correct_count = 0
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 第 {i}/{len(questions)} 题")
            print(f"📖 问题: {question['question_name']}")
            print("-" * 50)
            print(question['question_content'])
            
            user_answer = input("\n💭 你的答案: ").strip()
            
            print("\n💡 正确答案:")
            print("-" * 50)
            print(question['answer_content'])
            
            while True:
                self_assessment = input("\n🤔 你认为自己答对了吗? (y/n): ").strip().lower()
                if self_assessment in ['y', 'yes', 'n', 'no']:
                    break
            
            is_correct = self_assessment in ['y', 'yes']
            if is_correct:
                correct_count += 1
            
            answers.append({
                "question": question['question_name'],
                "user_answer": user_answer,
                "correct": is_correct
            })
        
        score = (correct_count / len(answers)) * 100 if answers else 0
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "quiz_type": "category_focused",
            "category": category,
            "total_questions": len(answers),
            "correct_answers": correct_count,
            "score": score,
            "answers": answers
        }
        
        self.results.append(result)
        self._save_results()
        
        self._display_results(result)
        return result
    
    def _display_results(self, result: Dict):
        """Display quiz results"""
        print("\n" + "=" * 50)
        print("📊 测验结果")
        print("=" * 50)
        print(f"📝 总题数: {result['total_questions']}")
        print(f"✅ 正确数: {result['correct_answers']}")
        print(f"❌ 错误数: {result['total_questions'] - result['correct_answers']}")
        print(f"📈 得分: {result['score']:.1f}%")
        
        if 'time_spent' in result:
            print(f"⏱️  用时: {result['time_spent']/60:.1f} 分钟")
        
        # Category breakdown
        if 'answers' in result:
            category_stats = {}
            for answer in result['answers']:
                category = answer.get('category', 'unknown')
                if category not in category_stats:
                    category_stats[category] = {'total': 0, 'correct': 0}
                category_stats[category]['total'] += 1
                if answer['correct']:
                    category_stats[category]['correct'] += 1
            
            print("\n📂 分类表现:")
            for category, stats in category_stats.items():
                accuracy = (stats['correct'] / stats['total']) * 100
                print(f"  {category}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    def show_progress_analysis(self):
        """Show detailed progress analysis"""
        if not self.results:
            print("📊 暂无测验记录")
            return
        
        print("📊 学习进度分析")
        print("=" * 50)
        
        # Overall statistics
        total_quizzes = len(self.results)
        total_questions = sum(r['total_questions'] for r in self.results)
        total_correct = sum(r['correct_answers'] for r in self.results)
        overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        
        print(f"📈 总测验次数: {total_quizzes}")
        print(f"📝 总题目数: {total_questions}")
        print(f"✅ 总正确数: {total_correct}")
        print(f"📊 总体准确率: {overall_accuracy:.1f}%")
        
        # Recent performance (last 5 quizzes)
        recent_results = self.results[-5:]
        if len(recent_results) > 1:
            recent_accuracy = sum(r['score'] for r in recent_results) / len(recent_results)
            print(f"📈 最近5次平均得分: {recent_accuracy:.1f}%")
        
        # Category performance
        category_performance = {}
        for result in self.results:
            if 'answers' in result:
                for answer in result['answers']:
                    category = answer.get('category', 'unknown')
                    if category not in category_performance:
                        category_performance[category] = {'total': 0, 'correct': 0}
                    category_performance[category]['total'] += 1
                    if answer['correct']:
                        category_performance[category]['correct'] += 1
        
        print("\n📂 分类表现:")
        for category, stats in sorted(category_performance.items()):
            accuracy = (stats['correct'] / stats['total']) * 100
            print(f"  {category}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
        
        # Improvement suggestions
        print("\n💡 改进建议:")
        weak_categories = [cat for cat, stats in category_performance.items() 
                          if (stats['correct'] / stats['total']) < 0.7]
        if weak_categories:
            print(f"  - 重点复习: {', '.join(weak_categories)}")
        else:
            print("  - 表现良好，继续保持！")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AI/MLE 面试测验系统")
    parser.add_argument("--project-root", 
                       default="/Users/hanguangshuai/Dropbox/Documents/Job/ML design",
                       help="项目根目录路径")
    parser.add_argument("--mode", 
                       choices=["timed", "adaptive", "category", "analysis"],
                       default="timed",
                       help="测验模式")
    parser.add_argument("--category", help="指定分类")
    parser.add_argument("--num-questions", type=int, default=10,
                       help="题目数量")
    parser.add_argument("--time-limit", type=int, default=30,
                       help="时间限制(分钟)")
    
    args = parser.parse_args()
    
    quiz_system = QuizSystem(args.project_root)
    
    if args.mode == "timed":
        quiz_system.start_timed_quiz(args.num_questions, args.time_limit)
    elif args.mode == "adaptive":
        quiz_system.adaptive_quiz()
    elif args.mode == "category":
        if not args.category:
            print("❌ 请指定分类 --category")
            return
        quiz_system.category_focused_quiz(args.category, args.num_questions)
    elif args.mode == "analysis":
        quiz_system.show_progress_analysis()

if __name__ == "__main__":
    main()
