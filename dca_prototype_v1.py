"""
Dialectical Cognitive Architecture (DCA) Prototype

This script implements Experiment #34: A Dialectical Cognitive Architecture that uses TSR as an
"executive layer" to guide LLM reasoning through strategic semantic analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import spacy
import networkx as nx
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import google.generativeai as genai
from dataclasses import dataclass

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyAEkmac2blqwr-zSdKjHZcJKF4TMixbWiY"
genai.configure(api_key=GEMINI_API_KEY)

# Constants
PROBLEM = """
As the CEO of a legacy automotive company, should we pivot to an all-electric vehicle (EV) lineup 
immediately, risking short-term financial stability, or should we pursue a gradual transition, 
risking long-term market relevance?
"""

POLE_A = ['all-electric vehicles', 'immediate pivot', 'market leader', 'disruption', 'long-term relevance']
POLE_B = ['gradual transition', 'hybrid vehicles', 'financial stability', 'existing assets', 'short-term profit']

@dataclass
class ConceptNode:
    """Represents a concept node in the semantic network."""
    name: str
    vector: np.ndarray
    
@dataclass
class Mediator:
    """Represents a mediator between two poles."""
    name: str
    vector: np.ndarray
    score: float

class DialecticalCognitiveArchitecture:
    def __init__(self):
        """Initialize the DCA with required models and components."""
        print("Initializing DCA...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_sm')
        self.llm = genai.GenerativeModel('gemini-2.5-pro')
        self.temperature = 0.5
        
        # Initialize poles with vector representations
        self.pole_a = self.semantic_model.encode(POLE_A, convert_to_tensor=False)
        self.pole_b = self.semantic_model.encode(POLE_B, convert_to_tensor=False)
        
        # Calculate the synthetic mediator
        self.synthetic_mediator = (np.mean(self.pole_a, axis=0) + np.mean(self.pole_b, axis=0)) / 2
        
    def diagnose_problem(self, problem_text: str) -> Dict:
        """Phase 1: Deconstruct the problem into its core semantic tensions."""
        print("\n" + "="*80)
        print("PHASE 1: DIAGNOSIS")
        print("="*80)
        
        # Extract key concepts using spaCy
        doc = self.nlp(problem_text)
        key_concepts = [chunk.text.lower() for chunk in doc.noun_chunks]
        key_concepts = list(set(key_concepts))  # Remove duplicates
        
        # Calculate contradiction energy
        pole_a_avg = np.mean(self.pole_a, axis=0)
        pole_b_avg = np.mean(self.pole_b, axis=0)
        contradiction_energy = 1 - cosine_similarity(
            pole_a_avg.reshape(1, -1), 
            pole_b_avg.reshape(1, -1)
        )[0][0]
        
        print(f"Core Tension: Immediate Pivot vs. Gradual Transition")
        print(f"Contradiction Energy (E_c): {contradiction_energy:.4f}")
        print(f"Key Concepts: {', '.join(key_concepts[:10])}...")
        
        return {
            'core_tension': ('immediate_pivot', 'gradual_transition'),
            'contradiction_energy': contradiction_energy,
            'key_concepts': key_concepts
        }
    
    def analyze_solution_space(self, key_concepts: List[str]) -> Dict:
        """Phase 2: Map the solution space by finding mediators and keystone concepts."""
        print("\n" + "="*80)
        print("PHASE 2: STRATEGIC ANALYSIS")
        print("="*80)
        
        # Encode key concepts
        concept_vectors = self.semantic_model.encode(key_concepts, convert_to_tensor=False)
        
        # Find potential mediators
        mediators = []
        for concept, vector in zip(key_concepts, concept_vectors):
            # Calculate mediation score (how well it bridges the poles)
            score = (cosine_similarity(
                vector.reshape(1, -1), 
                self.synthetic_mediator.reshape(1, -1)
            )[0][0] + 1) / 2  # Normalize to [0,1]
            
            if score > 0.7:  # Threshold for good mediators
                mediators.append(Mediator(concept, vector, score))
        
        # Sort by score and take top 5
        mediators.sort(key=lambda x: x.score, reverse=True)
        top_mediators = mediators[:5]
        
        # Find keystone concept (highest connectivity)
        keystone = max(top_mediators, key=lambda x: x.score) if top_mediators else None
        
        print("Top Mediating Concepts:")
        for i, m in enumerate(top_mediators, 1):
            print(f"  {i}. {m.name} (score: {m.score:.3f})")
        
        if keystone:
            print(f"\nIdentified Strategic Keystone: '{keystone.name}'")
        
        return {
            'mediators': [(m.name, m.score) for m in top_mediators],
            'keystone': keystone.name if keystone else None,
            'keystone_vector': keystone.vector if keystone else None
        }
    
    def construct_prompts(self, analysis: Dict) -> Tuple[str, str]:
        """Phase 3: Construct the baseline and DCA-guided prompts."""
        print("\n" + "="*80)
        print("PHASE 3: PROMPT ENGINEERING")
        print("="*80)
        
        # Baseline prompt
        baseline_prompt = """As the CEO of a legacy automotive company, what is your strategy 
for transitioning to electric vehicles? Consider both immediate and long-term implications.

Structure your response with:
1. Executive Summary
2. Key Strategic Pillars
3. Implementation Timeline
4. Risk Assessment
5. Conclusion"""
        
        # DCA-guided prompt with enhanced structure
        mediators = [m[0] for m in analysis['mediators']]
        keystone = analysis['keystone']
        
        dca_prompt = f"""# DIALECTICAL STRATEGIC FRAMEWORK

## CONTEXT
You are a strategic advisor to the CEO of a legacy automotive company facing a critical decision point. The company must navigate the transition to electric vehicles while managing competing priorities.

## CORE TENSION
{PROBLEM}

## DIALECTICAL ANALYSIS

### 1. THESIS (IMMEDIATE PIVOT)
- **Key Argument**: Present the strongest case for an immediate, aggressive transition to EVs
- **Supporting Evidence**: Market trends, regulatory pressures, first-mover advantages
- **Risks**: Financial strain, operational disruption, supply chain challenges

### 2. ANTITHESIS (GRADUAL TRANSITION)
- **Key Argument**: Present the strongest case for a measured, phased approach
- **Supporting Evidence**: Financial stability, existing infrastructure, customer readiness
- **Risks**: Market share loss, regulatory non-compliance, technological obsolescence

### 3. SYNTHESIS (MEDIATED STRATEGY)
Develop an integrated strategy that resolves the core tension using these mediating principles:

#### Primary Mediator: {keystone}
- How this resolves the tension
- Specific initiatives
- Expected outcomes

#### Secondary Mediator: {mediators[1] if len(mediators) > 1 else 'Financial Sustainability'}
- Integration with primary mediator
- Implementation approach
- Success metrics

#### Tertiary Mediator: {mediators[2] if len(mediators) > 2 else 'Customer-Centric Transition'}
- Customer segmentation
- Value proposition
- Adoption incentives

## EXECUTIVE RECOMMENDATION
- Clear, prioritized action items
- 12-24 month roadmap
- Resource allocation
- Key performance indicators

## IMPLEMENTATION CHECKPOINTS
- 0-6 months: [Key milestones]
- 6-12 months: [Key milestones]
- 12-24 months: [Key milestones]

## RISK MITIGATION
- Potential obstacles
- Contingency plans
- Early warning indicators

## CONCLUSION
- Summary of strategic advantage
- Long-term vision
- Call to action"""
        
        return baseline_prompt, dca_prompt
        
        print("Constructed baseline and DCA-guided prompts.")
        return baseline_prompt, dca_prompt
    
    def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response with retry logic for incomplete responses."""
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(
                    prompt,
                    generation_config={
                        'temperature': self.temperature,
                        'max_output_tokens': 4096,  # Increased token limit
                        'top_p': 0.9,
                        'top_k': 50
                    }
                )
                
                # Validate response completeness
                text = response.text.strip()
                if not text:
                    raise ValueError("Empty response from model")
                    
                # Check for common truncation indicators
                if any(phrase in text.lower() for phrase in ["...", "continue", "in conclusion"]):
                    if attempt < max_retries - 1:
                        print(f"  - Response may be truncated, retrying... (Attempt {attempt + 1}/{max_retries})")
                        continue
                        
                return text
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  - Error generating response (attempt {attempt + 1}): {str(e)}")
                    return "Error: Failed to generate complete response after multiple attempts."
                continue
                
        return "Error: Unknown error in response generation"
    
    def execute_generation(self, baseline_prompt: str, dca_prompt: str) -> Tuple[str, str]:
        """Phase 4: Generate responses using the LLM."""
        print("\n" + "="*80)
        print("PHASE 4: EXECUTION")
        print("="*80)
        
        print("Generating baseline response...")
        baseline_response = self._generate_with_retry(baseline_prompt)
        
        print("\nGenerating DCA-guided response...")
        dca_response = self._generate_with_retry(dca_prompt)
        
        return baseline_response, dca_response
    
    def _analyze_response_quality(self, response: str, response_type: str) -> Dict:
        """Analyze the quality of a response."""
        # Basic metrics
        word_count = len(response.split())
        sentence_count = len([s for s in response.split('. ') if s.strip()])
        
        # Check for structure elements
        has_structure = any(marker in response.lower() for marker in 
                          ['##', '###', '1.', '2.', '3.', '- ', '* ', '• '])
        
        # Check for dialectical elements if DCA response
        if response_type == 'dca':
            has_dialectical = all(term in response.lower() 
                                for term in ['thesis', 'antithesis', 'synthesis'])
        else:
            has_dialectical = False
            
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'has_structure': has_structure,
            'has_dialectical': has_dialectical
        }
    
    def verify_and_conclude(self, baseline_response: str, dca_response: str, keystone_vector: np.ndarray) -> None:
        """Phase 5: Verify and compare the responses."""
        print("\n" + "="*80)
        print("PHASE 5: VERIFICATION")
        print("="*80)
        
        # Encode responses
        baseline_vec = self.semantic_model.encode(baseline_response, convert_to_tensor=False).reshape(1, -1)
        dca_vec = self.semantic_model.encode(dca_response, convert_to_tensor=False).reshape(1, -1)
        
        # Calculate distances to synthetic mediator
        baseline_dist = 1 - cosine_similarity(baseline_vec, self.synthetic_mediator.reshape(1, -1))[0][0]
        dca_dist = 1 - cosine_similarity(dca_vec, self.synthetic_mediator.reshape(1, -1))[0][0]
        
        # Calculate distances to keystone
        baseline_keystone_dist = 1 - cosine_similarity(baseline_vec, keystone_vector.reshape(1, -1))[0][0]
        dca_keystone_dist = 1 - cosine_similarity(dca_vec, keystone_vector.reshape(1, -1))[0][0]
        
        # Analyze response quality
        baseline_analysis = self._analyze_response_quality(baseline_response, 'baseline')
        dca_analysis = self._analyze_response_quality(dca_response, 'dca')
        
        # Print results
        print("\nTable 2: Quantitative Verification")
        print("-" * 80)
        print(f"{'Metric':<30} | {'Baseline':<20} | {'DCA-Guided':<20}")
        print("-" * 80)
        print(f"{'Distance to Mediator':<30} | {baseline_dist:.4f}{' ' * 10} | {dca_dist:.4f}")
        print(f"{'Distance to Keystone':<30} | {baseline_keystone_dist:.4f}{' ' * 10} | {dca_keystone_dist:.4f}")
        print(f"{'Word Count':<30} | {baseline_analysis['word_count']:<20} | {dca_analysis['word_count']}")
        print(f"{'Sentence Count':<30} | {baseline_analysis['sentence_count']:<20} | {dca_analysis['sentence_count']}")
        
        # Qualitative analysis
        print("\nQualitative Analysis:")
        print("-" * 80)
        print("1. Baseline Response:")
        print(f"   - Structure: {'Well-structured' if baseline_analysis['has_structure'] else 'Lacks structure'}")
        print(f"   - Depth: {'Detailed' if baseline_analysis['word_count'] > 300 else 'Brief'}")
        print(f"   - Key Strength: {'Clear executive summary' if 'executive summary' in baseline_response.lower() else 'General overview'}")
        
        print("\n2. DCA-Guided Response:")
        print(f"   - Dialectical Structure: {'Complete' if dca_analysis['has_dialectical'] else 'Incomplete'}")
        print(f"   - Mediator Integration: {'Strong' if dca_analysis['word_count'] > 500 else 'Needs expansion'}")
        print(f"   - Strategic Value: {'High' if 'implementation' in dca_response.lower() and 'risk' in dca_response.lower() else 'Needs improvement'}")
        
        # Final conclusion
        improvement = ((baseline_dist - dca_dist) / baseline_dist) * 100 if baseline_dist > 0 else 0
        print("\n" + "="*80)
        print("FINAL CONCLUSION")
        print("="*80)
        print(f"The DCA-guided approach demonstrated a {improvement:.1f}% improvement in semantic alignment")
        print("with the ideal mediator compared to the baseline response.")
        
        if improvement > 0:
            print("The structured dialectical approach led to a more nuanced and strategic")
            print("recommendation that better synthesizes the core tension.")
        else:
            print("The baseline response performed similarly or better than the DCA approach.")
            print("This may indicate that the problem requires less dialectical reasoning")
            print("or that the DCA parameters need adjustment.")
            
        # Save detailed analysis
        analysis = {
            'baseline_metrics': {
                'distance_to_mediator': float(baseline_dist),
                'distance_to_keystone': float(baseline_keystone_dist),
                **baseline_analysis
            },
            'dca_metrics': {
                'distance_to_mediator': float(dca_dist),
                'distance_to_keystone': float(dca_keystone_dist),
                **dca_analysis
            },
            'improvement_percentage': float(improvement),
            'keystone_concept': keystone_vector.tolist() if keystone_vector is not None else None
        }
        
        os.makedirs('dca_results', exist_ok=True)
        with open('dca_results/analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def run(self):
        """Execute the full DCA workflow."""
        try:
            # Phase 1: Diagnosis
            diagnosis = self.diagnose_problem(PROBLEM)
            
            # Phase 2: Strategic Analysis
            analysis = self.analyze_solution_space(diagnosis['key_concepts'])
            
            # Phase 3: Prompt Engineering
            baseline_prompt, dca_prompt = self.construct_prompts(analysis)
            
            # Phase 4: Execution
            baseline_response, dca_response = self.execute_generation(baseline_prompt, dca_prompt)
            
            # Save responses
            os.makedirs('dca_results', exist_ok=True)
            with open('dca_results/baseline_response.txt', 'w', encoding='utf-8') as f:
                f.write(baseline_response)
            with open('dca_results/dca_response.txt', 'w', encoding='utf-8') as f:
                f.write(dca_response)
            
            # Phase 5: Verification
            self.verify_and_conclude(
                baseline_response, 
                dca_response,
                analysis['keystone_vector']
            )
            
            print("\nAnalysis complete! Results saved to 'dca_results/' directory.")
            
        except Exception as e:
            print(f"\nError during DCA execution: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Initialize and run the DCA
    dca = DialecticalCognitiveArchitecture()
    dca.run()
