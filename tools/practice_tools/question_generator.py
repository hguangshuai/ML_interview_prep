#!/usr/bin/env python3
"""
AI/MLE 面试问题随机生成器
支持从已学习的问题中随机出题，帮助复习和练习
"""

import os
import random
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import markdown
from datetime import datetime

class QuestionGenerator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.questions_dir = self.project_root / "questions"
        self.answers_dir = self.project_root / "answers"
        self.progress_file = self.project_root / "practice_tools" / "progress.json"
        
        # Load progress tracking
        self.progress = self._load_progress()
        
    def _load_progress(self) -> Dict:
        """Load learning progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "total_questions": 0,
            "answered_correctly": 0,
            "categories": {},
            "last_session": None
        }
    
    def _save_progress(self):
        """Save learning progress to file"""
        self.progress["last_session"] = datetime.now().isoformat()
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)
    
    def scan_questions(self) -> Dict[str, List[str]]:
        """Scan all available questions by category"""
        questions = {}
        
        for category_dir in self.questions_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                questions[category_name] = []
                
                for question_file in category_dir.glob("*.md"):
                    questions[category_name].append(question_file.stem)
        
        return questions
    
    def get_random_question(self, category: Optional[str] = None, 
                          difficulty: Optional[str] = None) -> Dict:
        """Get a random question from specified category and difficulty"""
        available_questions = self.scan_questions()
        
        if category and category in available_questions:
            category_questions = available_questions[category]
        else:
            # Random category
            category = random.choice(list(available_questions.keys()))
            category_questions = available_questions[category]
        
        if not category_questions:
            return {"error": "No questions found in selected category"}
        
        # Select random question
        question_name = random.choice(category_questions)
        
        # Read question content
        question_file = self.questions_dir / category / f"{question_name}.md"
        answer_file = self.answers_dir / category / f"{question_name}.md"
        
        question_content = ""
        answer_content = ""
        
        if question_file.exists():
            with open(question_file, 'r', encoding='utf-8') as f:
                question_content = f.read()
        
        if answer_file.exists():
            with open(answer_file, 'r', encoding='utf-8') as f:
                answer_content = f.read()
        
        return {
            "category": category,
            "question_name": question_name,
            "question_content": question_content,
            "answer_content": answer_content,
            "question_file": str(question_file),
            "answer_file": str(answer_file)
        }
    
    def generate_quiz(self, num_questions: int = 5, 
                     categories: Optional[List[str]] = None) -> List[Dict]:
        """Generate a quiz with multiple questions"""
        quiz = []
        available_questions = self.scan_questions()
        
        # Filter categories if specified
        if categories:
            available_questions = {k: v for k, v in available_questions.items() 
                                 if k in categories}
        
        for _ in range(num_questions):
            question = self.get_random_question()
            if "error" not in question:
                quiz.append(question)
        
        return quiz
    
    def start_interactive_session(self):
        """Start an interactive question-answer session"""
        print("🎯 AI/MLE 面试问题练习系统")
        print("=" * 50)
        
        while True:
            print("\n选择练习模式:")
            print("1. 随机单题练习")
            print("2. 分类练习")
            print("3. 快速测验 (5题)")
            print("4. 查看学习进度")
            print("5. 退出")
            
            choice = input("\n请选择 (1-5): ").strip()
            
            if choice == "1":
                self._single_question_mode()
            elif choice == "2":
                self._category_mode()
            elif choice == "3":
                self._quiz_mode()
            elif choice == "4":
                self._show_progress()
            elif choice == "5":
                print("👋 练习结束，继续加油！")
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def _single_question_mode(self):
        """Single question practice mode"""
        question = self.get_random_question()
        if "error" in question:
            print("❌ 没有找到问题")
            return
        
        print(f"\n📚 分类: {question['category']}")
        print(f"📝 问题: {question['question_name']}")
        print("-" * 50)
        print(question['question_content'])
        
        input("\n按回车键查看答案...")
        print("\n💡 答案:")
        print("-" * 50)
        print(question['answer_content'])
        
        # Record progress
        self._record_answer(question['category'], True)
    
    def _category_mode(self):
        """Category-specific practice mode"""
        available_questions = self.scan_questions()
        
        print("\n📂 可用分类:")
        for i, category in enumerate(available_questions.keys(), 1):
            print(f"{i}. {category}")
        
        try:
            choice = int(input("\n选择分类 (输入数字): ")) - 1
            categories = list(available_questions.keys())
            if 0 <= choice < len(categories):
                category = categories[choice]
                question = self.get_random_question(category=category)
                
                print(f"\n📚 分类: {question['category']}")
                print(f"📝 问题: {question['question_name']}")
                print("-" * 50)
                print(question['question_content'])
                
                input("\n按回车键查看答案...")
                print("\n💡 答案:")
                print("-" * 50)
                print(question['answer_content'])
                
                self._record_answer(question['category'], True)
            else:
                print("❌ 无效选择")
        except ValueError:
            print("❌ 请输入有效数字")
    
    def _quiz_mode(self):
        """Quiz mode with multiple questions"""
        print("\n🎯 开始快速测验 (5题)")
        quiz = self.generate_quiz(num_questions=5)
        
        correct = 0
        for i, question in enumerate(quiz, 1):
            print(f"\n📝 第 {i} 题")
            print(f"📚 分类: {question['category']}")
            print(f"📖 问题: {question['question_name']}")
            print("-" * 50)
            print(question['question_content'])
            
            input("\n按回车键查看答案...")
            print("\n💡 答案:")
            print("-" * 50)
            print(question['answer_content'])
            
            # Simple scoring (always correct for now)
            correct += 1
            self._record_answer(question['category'], True)
        
        print(f"\n🎉 测验完成！得分: {correct}/{len(quiz)}")
    
    def _show_progress(self):
        """Show learning progress"""
        print("\n📊 学习进度")
        print("=" * 50)
        print(f"总问题数: {self.progress['total_questions']}")
        print(f"正确回答: {self.progress['answered_correctly']}")
        
        if self.progress['total_questions'] > 0:
            accuracy = (self.progress['answered_correctly'] / 
                       self.progress['total_questions']) * 100
            print(f"准确率: {accuracy:.1f}%")
        
        print("\n📂 分类统计:")
        for category, stats in self.progress['categories'].items():
            print(f"  {category}: {stats['answered']} 题")
    
    def _record_answer(self, category: str, correct: bool):
        """Record answer for progress tracking"""
        self.progress['total_questions'] += 1
        if correct:
            self.progress['answered_correctly'] += 1
        
        if category not in self.progress['categories']:
            self.progress['categories'][category] = {'answered': 0}
        
        self.progress['categories'][category]['answered'] += 1
        self._save_progress()

def main():
    parser = argparse.ArgumentParser(description="AI/MLE 面试问题生成器")
    parser.add_argument("--project-root", 
                       default="/Users/hanguangshuai/Dropbox/Documents/Job/ML design",
                       help="项目根目录路径")
    parser.add_argument("--mode", choices=["interactive", "single", "quiz"],
                       default="interactive",
                       help="运行模式")
    parser.add_argument("--category", help="指定问题分类")
    parser.add_argument("--num-questions", type=int, default=5,
                       help="测验题目数量")
    
    args = parser.parse_args()
    
    generator = QuestionGenerator(args.project_root)
    
    if args.mode == "interactive":
        generator.start_interactive_session()
    elif args.mode == "single":
        question = generator.get_random_question(category=args.category)
        if "error" not in question:
            print(f"分类: {question['category']}")
            print(f"问题: {question['question_name']}")
            print("-" * 50)
            print(question['question_content'])
        else:
            print("❌ 没有找到问题")
    elif args.mode == "quiz":
        quiz = generator.generate_quiz(num_questions=args.num_questions)
        for i, question in enumerate(quiz, 1):
            print(f"\n第 {i} 题 - {question['category']}")
            print("-" * 50)
            print(question['question_content'])

if __name__ == "__main__":
    main()
