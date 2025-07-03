#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Accuracy Calculator
Tính accuracy trực tiếp từ system Neo4j QA đang chạy

Usage:
    python system_accuracy_calculator.py
"""

import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer
import traceback
import json
import pandas as pd
from typing import List, Dict, Optional
import re
from difflib import SequenceMatcher
from datetime import datetime
import ast

from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import AzureChatOpenAI

import dotenv
import os

dotenv.load_dotenv()

class SystemAccuracyCalculator:
    def __init__(self):
        """Initialize system accuracy calculator"""
        # Reuse the same configuration as your main system
        self.neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # Initialize LLM with same config
        self.llm = AzureChatOpenAI(
            azure_deployment="chat-gpt4",
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )
        
        # Initialize Neo4j graph
        self.graph = Neo4jGraph(
            url=self.neo4j_url,
            username=self.neo4j_user,
            password=self.neo4j_password
        )
        
        # Setup QA chain - exact same as your system
        self.setup_qa_chain()
        
    def setup_qa_chain(self):
        """Setup QA chain - exactly same as your main system"""
        # Same cypher template as your system
        cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts Vietnamese to Cypher based on the Neo4j Schema provided, following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
3. Use only Nodes and relationships mentioned in the schema
4. Use exact matching with node labels and property names as shown in examples
5. Available node types: Disease, SubDisease, Symptom, Complication, Treatment, Description, TimeStage, Cause, Population, Duration, Type, Action, Goal, Topic, Advice, Mechanism, ConditionNote, Note, RiskFactor, Definition
6. Available relationships: HAS_SYMPTOM, HAS_CAUSE, HAS_TREATMENT, HAS_DESCRIPTION, OCCURS_AT, HAS_SUBTYPE, AFFECTS, HAS_COMPLICATION, HAS_TYPE, OCCURS_IN, LASTS_FOR, EXPECTED_BY_AGE, HAS_DEFINITION, RECOMMENDED_FOR
7. When searching for diseases or conditions, use exact name matching with property syntax

schema: {schema}

Examples:
Question: Các triệu chứng của Vô kinh thứ phát?
Answer: ```MATCH (sd:SubDisease {{name: "Vô kinh thứ phát"}})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name ```
Question: Các triệu chứng của Chứng kinh nguyệt ẩn?
Answer: ```MATCH (sd:SubDisease {{name: "Chứng kinh nguyệt ẩn"}})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name ```
Question: Các triệu chứng của Mãn kinh?
Answer: ```MATCH (d:Disease {{name: "Mãn kinh"}})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name ```
Question: Cách điều trị U xơ tử cung? 
Answer: ```MATCH (d:Disease {{name: "U xơ tử cung"}})-[:HAS_TREATMENT]->(t:Treatment)
RETURN t.name```
Question: Cách điều trị Viêm âm đạo?
Answer: ```MATCH (d:Disease {{name: "Viêm âm đạo"}})-[:HAS_TREATMENT]->(t:Treatment)
RETURN t.name```
Question: Cách điều trị Bệnh viêm vùng chậu?
Answer: ```MATCH (d:Disease {{name: "Bệnh viêm vùng chậu"}})-[:HAS_TREATMENT]->(t:Treatment)
RETURN t.name```
Question: Mãn kinh diễn ra ở người nào?
Answer: ```MATCH (d:Disease {{name: "Mãn kinh"}})-[:AFFECTS]->(p:Population)
RETURN p.name```
Question: Vô kinh có những loại nào?
Answer: ```MATCH (d:Disease {{name: "Vô kinh"}})-[:HAS_TYPE]->(t:Type)
RETURN t.name```
Question: Triệu chứng của vô kinh là gì?
Answer: ```MATCH (d:Disease {{name: "Vô kinh"}})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name```
Question: Nguyên nhân của vô kinh?
Answer: ```MATCH (d:Disease {{name: "Vô kinh"}})-[:HAS_CAUSE]->(c:Cause)
RETURN c.name```

Question: {question}
"""

        cypher_prompt = PromptTemplate(
            template=cypher_generation_template,
            input_variables=["schema", "question"]
        )

        # Same QA template as your system
        CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Final answer should be easily readable and structured.
The final answer should be in Vietnamese.

CRITICAL: If the context contains database results in list format, extract ONLY the values and present them as a clear list. Do NOT add your own knowledge.

For example:
- If context shows results about types or categories, list them clearly
- If context shows symptoms, treatments, or causes, present them in numbered format
- Always use the exact values from the database

Information:
{context}

Question: {question}
Helpful Answer:"""

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"], 
            template=CYPHER_QA_TEMPLATE
        )

        # Create QA chain - same as your system
        self.qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            return_intermediate_steps=True,
            cypher_prompt=cypher_prompt,
            qa_prompt=qa_prompt,
            allow_dangerous_requests=True
        )

    def query_graph(self, user_input):
        """Same query_graph function as your main system"""
        try:
            # Test graph connection
            test_query = self.graph.query("RETURN 1 as test")
            
            result = self.qa_chain(user_input)
            return result
        except Exception as e:
            print(f"Error in query_graph: {str(e)}")
            raise e

    def extract_database_results(self, db_results_str: str) -> List[str]:
        """Extract actual results from database"""
        if not db_results_str or db_results_str.strip() == "":
            return []
        
        try:
            # Parse the database results string
            results = ast.literal_eval(db_results_str)
            items = []
            
            for result in results:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, list):
                            items.extend(value)
                        elif isinstance(value, str):
                            items.append(value)
            
            return [self.clean_text(item) for item in items if item and item.strip()]
            
        except:
            # Fallback: extract using regex
            pattern = r"'([^']+)'"
            matches = re.findall(pattern, db_results_str)
            # Filter out key names like 's.name', 't.name', etc.
            items = [match for match in matches if not re.match(r'^[a-z]\.', match)]
            return [self.clean_text(item) for item in items if item and item.strip()]

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        text = text.lower().strip()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def extract_answer_items(self, answer: str) -> List[str]:
        """Extract items from model answer"""
        if not answer or "không biết" in answer.lower():
            return []
        
        items = []
        
        # Method 1: Numbered lists
        numbered_pattern = r'\d+\.\s*([^\n.]+?)(?=\s*\d+\.|$)'
        numbered_matches = re.findall(numbered_pattern, answer, re.DOTALL)
        
        if numbered_matches:
            return [self.clean_text(item) for item in numbered_matches]
        
        # Method 2: Bullet points
        bullet_pattern = r'[-*•]\s*([^\n]+)'
        bullet_matches = re.findall(bullet_pattern, answer)
        
        if bullet_matches:
            return [self.clean_text(item) for item in bullet_matches]
        
        # Method 3: Comma separated
        if ',' in answer:
            # Remove common prefixes
            cleaned = re.sub(r'^[^:]*:\s*', '', answer)  # Remove "Symptoms include:"
            cleaned = re.sub(r'^[^bao gồm]*bao gồm\s*', '', cleaned)  # Remove "...bao gồm"
            
            parts = cleaned.split(',')
            items = [self.clean_text(part) for part in parts if part.strip()]
            
            if len(items) > 1:
                return items
        
        # Method 4: Single item
        cleaned = self.clean_text(answer)
        return [cleaned] if cleaned else []

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        clean1 = self.clean_text(text1)
        clean2 = self.clean_text(text2)
        
        if not clean1 or not clean2:
            return 0.0
        
        # Exact match
        if clean1 == clean2:
            return 1.0
        
        # One contains the other
        if clean1 in clean2 or clean2 in clean1:
            return 0.85
        
        # Sequence similarity
        return SequenceMatcher(None, clean1, clean2).ratio()

    def evaluate_response(self, predicted_items: List[str], db_items: List[str], 
                         threshold: float = 0.7) -> Dict:
        """Evaluate model response against database results"""
        
        # If no database results, check if model correctly says "don't know"
        if not db_items:
            if not predicted_items:
                return {
                    "exact_match": True,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "count_accuracy": 1.0,
                    "matched_items": [],
                    "missed_items": [],
                    "extra_items": [],
                    "notes": "Correctly identified no results"
                }
            else:
                return {
                    "exact_match": False,
                    "precision": 0.0,
                    "recall": 1.0,
                    "f1": 0.0,
                    "count_accuracy": 0.0,
                    "matched_items": [],
                    "missed_items": [],
                    "extra_items": predicted_items,
                    "notes": "Incorrectly provided answers when none exist"
                }
        
        # If no predicted items but database has results
        if not predicted_items:
            return {
                "exact_match": False,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "count_accuracy": 0.0,
                "matched_items": [],
                "missed_items": db_items,
                "extra_items": [],
                "notes": "Failed to extract any answers"
            }
        
        # Match predicted items with database items
        matched_items = []
        missed_items = []
        extra_items = list(predicted_items)
        
        for db_item in db_items:
            best_match = None
            best_score = 0.0
            
            for pred_item in extra_items:
                score = self.calculate_text_similarity(db_item, pred_item)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = pred_item
            
            if best_match:
                matched_items.append({
                    "database": db_item,
                    "predicted": best_match,
                    "similarity": best_score
                })
                extra_items.remove(best_match)
            else:
                missed_items.append(db_item)
        
        # Calculate metrics
        true_positives = len(matched_items)
        false_positives = len(extra_items)
        false_negatives = len(missed_items)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        exact_match = (false_positives == 0 and false_negatives == 0)
        
        # Count accuracy
        predicted_count = len(predicted_items)
        actual_count = len(db_items)
        count_accuracy = 1.0 - abs(predicted_count - actual_count) / max(actual_count, 1)
        
        return {
            "exact_match": exact_match,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count_accuracy": count_accuracy,
            "matched_items": matched_items,
            "missed_items": missed_items,
            "extra_items": extra_items,
            "predicted_count": predicted_count,
            "actual_count": actual_count,
            "notes": f"Matched {true_positives}/{actual_count} items"
        }

    def get_test_questions(self):
        """Get standard test questions"""
        return [
            {
                "id": 1,
                "question": "Các triệu chứng của U xơ tử cung?",
                "category": "symptoms"
            },
            {
                "id": 2,
                "question": "Cách điều trị U xơ tử cung?",
                "category": "treatment"
            },
            {
                "id": 3,
                "question": "U xơ tử cung ảnh hưởng đến ai?",
                "category": "population"
            },
            {
                "id": 4,
                "question": "Vô kinh có những loại nào?",
                "category": "types"
            },
            {
                "id": 5,
                "question": "Các triệu chứng của Mãn kinh?",
                "category": "symptoms"
            },
            {
                "id": 6,
                "question": "Nguyên nhân của vô kinh?",
                "category": "causes"
            },
            {
                "id": 7,
                "question": "Cách điều trị Viêm âm đạo?",
                "category": "treatment"
            },
            {
                "id": 8,
                "question": "Các triệu chứng của Vô kinh thứ phát?",
                "category": "symptoms"
            },
            {
                "id": 9,
                "question": "Mãn kinh diễn ra ở người nào?",
                "category": "population"
            },
            {
                "id": 10,
                "question": "Biến chứng của U xơ tử cung?",
                "category": "complications"
            }
        ]

    def test_single_question(self, test_item: Dict) -> Dict:
        """Test a single question"""
        question = test_item["question"]
        print(f"Testing: {question}")
        
        try:
            # Get model response using same function as your system
            result = self.query_graph(question)
            
            # Extract components - same as your system
            answer = result.get("result", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            cypher_query = ""
            db_results_str = ""
            
            if intermediate_steps:
                if len(intermediate_steps) > 0:
                    cypher_query = intermediate_steps[0].get("query", "")
                if len(intermediate_steps) > 1:
                    db_results_str = str(intermediate_steps[1].get("context", ""))
            
            # Extract items
            predicted_items = self.extract_answer_items(answer)
            db_items = self.extract_database_results(db_results_str)
            
            # Evaluate
            evaluation = self.evaluate_response(predicted_items, db_items)
            
            print(f"  DB Items ({len(db_items)}): {db_items[:3]}{'...' if len(db_items) > 3 else ''}")
            print(f"  Predicted ({len(predicted_items)}): {predicted_items[:3]}{'...' if len(predicted_items) > 3 else ''}")
            print(f"  F1: {evaluation['f1']:.3f}, Precision: {evaluation['precision']:.3f}, Recall: {evaluation['recall']:.3f}")
            
            return {
                "id": test_item["id"],
                "question": question,
                "category": test_item.get("category", ""),
                "answer": answer,
                "predicted_items": predicted_items,
                "db_items": db_items,
                "cypher_query": cypher_query,
                "db_results_raw": db_results_str,
                "evaluation": evaluation,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {
                "id": test_item["id"],
                "question": question,
                "category": test_item.get("category", ""),
                "answer": "",
                "predicted_items": [],
                "db_items": [],
                "cypher_query": "",
                "db_results_raw": "",
                "evaluation": {
                    "exact_match": False,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "count_accuracy": 0.0
                },
                "success": False,
                "error": str(e)
            }

    def calculate_overall_metrics(self, results: List[Dict]) -> Dict:
        """Calculate overall metrics"""
        successful_tests = [r for r in results if r["success"]]
        
        if not successful_tests:
            return {
                "total_questions": len(results),
                "successful_tests": 0,
                "success_rate": 0.0,
                "average_precision": 0.0,
                "average_recall": 0.0,
                "average_f1": 0.0,
                "exact_match_rate": 0.0,
                "average_count_accuracy": 0.0
            }
        
        n = len(successful_tests)
        
        metrics = {
            "total_questions": len(results),
            "successful_tests": n,
            "success_rate": n / len(results),
            "average_precision": sum(r["evaluation"]["precision"] for r in successful_tests) / n,
            "average_recall": sum(r["evaluation"]["recall"] for r in successful_tests) / n,
            "average_f1": sum(r["evaluation"]["f1"] for r in successful_tests) / n,
            "exact_match_rate": sum(r["evaluation"]["exact_match"] for r in successful_tests) / n,
            "average_count_accuracy": sum(r["evaluation"]["count_accuracy"] for r in successful_tests) / n
        }
        
        return metrics

    def run_system_accuracy_test(self):
        """Run accuracy test on the live system"""
        print("🚀 Testing System Accuracy")
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        # Get test questions
        test_questions = self.get_test_questions()
        print(f"📝 Testing {len(test_questions)} questions on live system")
        print("-" * 60)
        
        # Run tests
        results = []
        for i, test_item in enumerate(test_questions, 1):
            print(f"\n[{i}/{len(test_questions)}] ", end="")
            result = self.test_single_question(test_item)
            results.append(result)
        
        # Calculate metrics
        overall_metrics = self.calculate_overall_metrics(results)
        
        # Create report
        report = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "type": "live_system_accuracy_test",
                "model_info": {
                    "llm": "Azure GPT-4",
                    "database": "Neo4j",
                    "framework": "LangChain"
                }
            },
            "overall_metrics": overall_metrics,
            "detailed_results": results
        }
        
        # Print summary
        self.print_summary(overall_metrics)
        
        # Save results
        output_file = f"system_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return report

    def print_summary(self, metrics: Dict):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("📊 LIVE SYSTEM ACCURACY TEST SUMMARY")
        print("=" * 60)
        print(f"Total Questions:      {metrics['total_questions']}")
        print(f"Successful Tests:     {metrics['successful_tests']}")
        print(f"Success Rate:         {metrics['success_rate']:.1%}")
        print(f"Exact Match Rate:     {metrics['exact_match_rate']:.1%}")
        print(f"Average F1 Score:     {metrics['average_f1']:.3f}")
        print(f"Average Precision:    {metrics['average_precision']:.3f}")
        print(f"Average Recall:       {metrics['average_recall']:.3f}")
        print(f"Count Accuracy:       {metrics['average_count_accuracy']:.3f}")
        print("=" * 60)
        
        # Performance interpretation
        f1_score = metrics['average_f1']
        if f1_score >= 0.8:
            print("🎉 EXCELLENT system performance!")
        elif f1_score >= 0.6:
            print("✅ GOOD system performance!")
        elif f1_score >= 0.4:
            print("⚠️  FAIR system performance - needs improvement")
        else:
            print("❌ POOR system performance - major improvements needed")
        
        print("=" * 60)

def main():
    """Main function to test system accuracy"""
    try:
        print("🔧 Initializing System Accuracy Calculator...")
        calculator = SystemAccuracyCalculator()
        
        print("✅ Calculator initialized successfully")
        
        # Run accuracy test
        report = calculator.run_system_accuracy_test()
        
        print(f"\n🎉 System accuracy test completed!")
        
    except Exception as e:
        print(f"\n❌ System accuracy test failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()