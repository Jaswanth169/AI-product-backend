from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)
api_key = os.environ.get('NVIDIA_API_KEY', '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')
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Load agents configuration from JSON file
def load_agents_config():
    try:
        with open('agents_config.json', 'r') as f:
            config = json.load(f)
        return config['domains']
    except FileNotFoundError:
        print("ERROR: agents_config.json not found!")
        return {}
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in agents_config.json: {e}")
        return {}

# Load configuration at startup
DOMAINS = load_agents_config()

# In-memory storage for projects
projects = {}

@app.route('/api/domains', methods=['GET'])
def get_domains():
    """Get all domains and their agents"""
    return jsonify({"domains": DOMAINS})

@app.route('/api/project/create', methods=['POST'])
def create_project():
    """Create a new project"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['product_name', 'product_description', 'product_features', 'target_price', 'budget', 'industry']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Generate project ID
        project_id = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store project data
        projects[project_id] = {
            "project_id": project_id,
            "created_at": datetime.now().isoformat(),
            "product_name": data['product_name'],
            "product_description": data['product_description'],
            "product_features": data['product_features'],
            "target_price": data['target_price'],
            "budget": data['budget'],
            "industry": data['industry'],
            "agent_results": {}
        }
        
        print(f"\nProject Created: {project_id}")
        print(f"   Product: {data['product_name']}")
        print(f"   Industry: {data['industry']}")
        
        return jsonify({
            "project_id": project_id,
            "message": "Project created successfully"
        })
        
    except Exception as e:
        print(f"ERROR: Error creating project: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/agent/run', methods=['POST'])
def run_agent():
    """Run a specific agent for a project"""
    try:
        data = request.json
        project_id = data.get('project_id')
        agent_id = data.get('agent_id')
        
        if not project_id or not agent_id:
            return jsonify({"error": "Missing project_id or agent_id"}), 400
        
        if project_id not in projects:
            return jsonify({"error": "Project not found"}), 404
        
        # Find the agent in domains
        agent = None
        for domain_key, domain_data in DOMAINS.items():
            for a in domain_data['agents']:
                if a['id'] == agent_id:
                    agent = a
                    break
            if agent:
                break
        
        if not agent:
            return jsonify({"error": "Agent not found"}), 404
        
        project = projects[project_id]
        
        print(f"\nRunning Agent: {agent['name']}")
        print(f"   Project: {project['product_name']}")
        print(f"   Agent ID: {agent_id}")
        
        # Prepare context from project data
        features_str = ', '.join(project['product_features']) if isinstance(project['product_features'], list) else project['product_features']
        
        # Replace placeholders in prompt
        prompt = agent['prompt']
        prompt = prompt.replace('{product_name}', project['product_name'])
        prompt = prompt.replace('{product_description}', project['product_description'])
        prompt = prompt.replace('{product_features}', features_str)
        prompt = prompt.replace('{target_price}', str(project['target_price']))
        prompt = prompt.replace('{budget}', str(project['budget']))
        prompt = prompt.replace('{industry}', project['industry'])
        
        context = f"""
Product Launch Analysis Request

Product Name: {project['product_name']}
Description: {project['product_description']}
Key Features: {features_str}
Target Price: ${project['target_price']}
Marketing Budget: ${project['budget']}
Industry: {project['industry']}

Task:
{prompt}
"""
        
        print(f"Sending request to NVIDIA API...")
        
        try:
            # Call NVIDIA API
            completion = client.chat.completions.create(
                model="qwen/qwen3-next-80b-a3b-thinking",
                messages=[{"role": "user", "content": context}],
                temperature=0.6,
                top_p=0.7,
                max_tokens=4096,
                stream=False
            )
            
            result = completion.choices[0].message.content
            
            # Handle None response
            if result is None:
                result = "No response generated. Please try again."
                print(f"WARNING: API returned None response")
            else:
                print(f"Agent completed successfully")
                print(f"   Response length: {len(result)} characters")
            
            # Store result
            projects[project_id]["agent_results"][agent_id] = {
                "agent_name": agent["name"],
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            return jsonify({
                "agent_id": agent_id,
                "agent_name": agent["name"],
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
            
        except Exception as api_error:
            print(f"NVIDIA API Error: {str(api_error)}")
            traceback.print_exc()
            return jsonify({
                "error": f"API Error: {str(api_error)}",
                "details": "Check if API key is valid and has quota remaining"
            }), 500
        
    except Exception as e:
        print(f"ERROR: Error running agent: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/project/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get project details and results"""
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    
    return jsonify(projects[project_id])

@app.route('/api/project/<project_id>/summary', methods=['GET'])
def get_project_summary(project_id):
    """Get AI-generated summary of all agent results"""
    try:
        if project_id not in projects:
            return jsonify({"error": "Project not found"}), 404
        
        project = projects[project_id]
        agent_results = project.get("agent_results", {})
        
        if not agent_results or len(agent_results) == 0:
            return jsonify({"error": "No agent results available. Please run at least one agent first."}), 400
        
        print(f"\nGenerating summary for project: {project_id}")
        print(f"   Agents completed: {len(agent_results)}")
        
        # Compile all results
        results_summary = "\n\n".join([
            f"{data['agent_name']}:\n{data['result']}"
            for data in agent_results.values()
        ])
        
        summary_prompt = f"""
Based on all the agent analyses below for {project['product_name']}, create a comprehensive executive summary that includes:

1. Key Recommendations (top 5)
2. Target Market & Customer Profile
3. Pricing Strategy
4. Go-to-Market Strategy
5. Expected Financial Outcomes
6. Critical Success Factors
7. Risks & Mitigation

Agent Results:
{results_summary}

Provide a clear, actionable executive summary.
"""
        
        print(f"Generating executive summary...")
        
        completion = client.chat.completions.create(
            model="qwen/qwen3-next-80b-a3b-thinking",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.6,
            top_p=0.7,
            max_tokens=4096,
            stream=False
        )
        
        summary = completion.choices[0].message.content
        
        print(f"Summary generated successfully")
        
        return jsonify({
            "project_id": project_id,
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"ERROR: Error generating summary: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/project/<project_id>/recommendations', methods=['GET'])
def get_agent_recommendations(project_id):
    """Get AI recommendations for which agents to run based on product details"""
    try:
        if project_id not in projects:
            return jsonify({"error": "Project not found"}), 404
        
        project = projects[project_id]
        
        print(f"\nGenerating agent recommendations for project: {project_id}")
        print(f"   Product: {project['product_name']}")
        
        features_str = ', '.join(project['product_features']) if isinstance(project['product_features'], list) else project['product_features']
        
        # Try multiple models with different prompts
        recommendations = None
        models_to_try = [
            ("meta/llama-3.1-70b-instruct", 2048),
            ("mistralai/mixtral-8x7b-instruct-v0.1", 2048),
            ("meta/llama-3.1-8b-instruct", 1024)
        ]
        
        for model_name, max_tokens in models_to_try:
            try:
                print(f"   Trying model: {model_name}")
                
                # Simpler, more direct prompt
                simple_prompt = f"""Analyze this product and recommend which AI agents to run first:

Product: {project['product_name']}
Description: {project['product_description']}
Features: {features_str}
Price: ${project['target_price']}
Budget: ${project['budget']}
Industry: {project['industry']}

Available agent domains:
1. Market Intelligence (4 agents) - Market research, sizing, competition
2. Pricing Strategy (4 agents) - Price optimization, revenue models
3. Advertising & Promotion (5 agents) - Marketing channels, campaigns
4. Customer Insights (4 agents) - Customer personas, journeys
5. Sales Forecasting (4 agents) - Demand forecasting, targets
6. Customer Acquisition (4 agents) - Acquisition channels, funnels
7. Competitive Strategy (4 agents) - Positioning, SWOT
8. Product Positioning (4 agents) - Value proposition, messaging
9. Creative Strategy (4 agents) - Creative concepts, content
10. Performance Analytics (4 agents) - KPIs, optimization

Recommend the TOP 5-8 most important agents to run first for THIS product. For each, explain:
- WHY it's critical for this specific product
- WHAT insights it will provide

Keep response under 1500 words, focused and actionable."""
                
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": simple_prompt}],
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=max_tokens,
                    stream=False
                )
                
                if hasattr(completion.choices[0].message, 'content'):
                    content = completion.choices[0].message.content
                    if content and len(content.strip()) > 100:  # Minimum viable response
                        recommendations = content
                        print(f"   ✓ Success with {model_name}")
                        print(f"   Response length: {len(recommendations)} characters")
                        break
                
            except Exception as model_error:
                print(f"   ✗ {model_name} failed: {str(model_error)}")
                continue
        
        # If all models fail, use smart template-based recommendations
        if not recommendations:
            print(f"   All models failed, using intelligent template")
            recommendations = generate_smart_recommendations(project, features_str)
        
        return jsonify({
            "project_id": project_id,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"ERROR: Error generating recommendations: {str(e)}")
        traceback.print_exc()
        # Return template recommendations on error
        project = projects.get(project_id, {})
        features_str = ', '.join(project.get('product_features', [])) if isinstance(project.get('product_features'), list) else project.get('product_features', '')
        return jsonify({
            "project_id": project_id,
            "recommendations": generate_smart_recommendations(project, features_str),
            "generated_at": datetime.now().isoformat()
        })

def generate_smart_recommendations(project, features_str):
    """Generate intelligent template-based recommendations"""
    product_name = project.get('product_name', 'your product')
    industry = project.get('industry', 'the market')
    target_price = project.get('target_price', 'N/A')
    budget = project.get('budget', 'N/A')
    
    return f"""# AI Agent Recommendations for {product_name}

## Analysis Summary
Based on your {industry} product at ${target_price} with a ${budget} marketing budget, here are the strategic agent recommendations:

## TOP PRIORITY AGENTS (Must Run First)

### 1. Market Size Estimator (Market Intelligence Domain)
**WHY CRITICAL:** Before investing ${budget}, you need to validate the market opportunity exists. This agent will determine if the addressable market is large enough to justify your investment and pricing strategy.

**WHAT YOU'LL GET:**
- Total Addressable Market (TAM) size for {product_name}
- Serviceable market estimates
- Growth rate projections for {industry}
- Market entry feasibility assessment

**ACTION:** Run this FIRST to validate the business case.

---

### 2. Competitive Landscape Analyzer (Market Intelligence Domain)  
**WHY CRITICAL:** At ${target_price}, you need to understand who you're competing against and where you can win. This is especially important in {industry} where competition is fierce.

**WHAT YOU'LL GET:**
- Direct and indirect competitor identification
- Competitive pricing analysis
- Market gaps and white space opportunities
- Differentiation requirements based on your features: {features_str}

**ACTION:** Run IMMEDIATELY after market sizing to understand competitive dynamics.

---

### 3. Customer Segmentation Analyst (Customer Insights Domain)
**WHY CRITICAL:** A ${budget} budget can be wasted quickly without knowing exactly who to target. This agent identifies your highest-value customer segments.

**WHAT YOU'LL GET:**
- 3-4 detailed customer segments
- Which segments to prioritize
- Segment size and value estimates
- Purchase behavior patterns

**ACTION:** Essential for targeting your marketing spend effectively.

---

### 4. Price Point Optimizer (Pricing Strategy Domain)
**WHY CRITICAL:** Your ${target_price} pricing needs validation against customer willingness to pay and competitive pricing. Wrong pricing can kill an otherwise great product.

**WHAT YOU'LL GET:**
- Optimal price point recommendations
- Price elasticity analysis
- Competitive pricing comparison
- Pricing tier suggestions

**ACTION:** Run early to validate or adjust pricing strategy.

---

### 5. Marketing Channel Strategist (Advertising & Promotion Domain)
**WHY CRITICAL:** With ${budget} to spend, you need to know which channels deliver the best ROI. This prevents budget waste on ineffective channels.

**WHAT YOU'LL GET:**
- Optimal marketing channel mix
- Budget allocation by channel
- Expected ROI per channel
- Channel-specific tactics

**ACTION:** Critical for planning your ${budget} allocation.

---

### 6. Value Proposition Architect (Product Positioning Domain)
**WHY CRITICAL:** Your features ({features_str}) need to be translated into compelling customer benefits. This agent creates the core messaging.

**WHAT YOU'LL GET:**
- Core value proposition statement
- Feature-to-benefit mapping
- Differentiation messaging
- Proof points for credibility

**ACTION:** Run before launching any marketing to ensure consistent messaging.

---

### 7. Customer Persona Developer (Customer Insights Domain)
**WHY CRITICAL:** Detailed personas help your team create targeted content and campaigns that resonate with real buyers.

**WHAT YOU'LL GET:**
- 3-4 detailed buyer personas
- Demographics, goals, pain points
- Buying behavior and decision criteria
- How to reach each persona

**ACTION:** Essential for content creation and campaign planning.

---

### 8. Demand Forecaster (Sales Forecasting Domain)
**WHY CRITICAL:** You need realistic sales projections to plan inventory, staffing, and cash flow. Prevents over/under-investment.

**WHAT YOU'LL GET:**
- First year sales forecast by quarter
- Adoption curve projections
- Best/base/worst case scenarios
- Key assumptions to track

**ACTION:** Run after market analysis to set realistic expectations.

---

## SECONDARY PRIORITY AGENTS (Run After Top 8)

### 9. Campaign Launch Planner
Creates comprehensive launch campaign timeline and strategy.

### 10. Competitive Positioning Strategist  
Develops unique competitive positioning and differentiation strategy.

### 11. Conversion Funnel Optimizer
Optimizes customer journey from awareness to purchase.

### 12. Content Marketing Strategist
Plans content strategy for first 90 days.

---

## STRATEGIC RATIONALE

**For {product_name} in {industry}:**

1. **Market Validation First** - Before spending ${budget}, validate that the market exists and is accessible
2. **Competition Understanding** - Know who you're fighting and where you can win
3. **Customer Clarity** - Identify exactly who will pay ${target_price} and why
4. **Pricing Confidence** - Ensure ${target_price} is optimized for market conditions
5. **Marketing Efficiency** - Plan ${budget} allocation for maximum ROI

**Key Success Factors:**
- Product features ({features_str}) must address real customer pain points
- Pricing must balance profitability with market competitiveness  
- Marketing must reach the right customers through the right channels
- Launch timing and messaging must be coordinated

---

## RECOMMENDED EXECUTION ORDER

**Week 1: Market Foundation**
1. Market Size Estimator
2. Competitive Landscape Analyzer
3. Customer Segmentation Analyst

**Week 2: Strategy Validation**  
4. Price Point Optimizer
5. Marketing Channel Strategist
6. Demand Forecaster

**Week 3: Positioning & Messaging**
7. Value Proposition Architect
8. Customer Persona Developer

**Week 4+: Campaign Planning**
9-12. Run secondary priority agents based on initial insights

---

## NEXT STEPS

1. **Start with the top 3 agents** to validate market opportunity
2. **Review insights** before proceeding to ensure viability
3. **Run pricing and marketing agents** to plan execution
4. **Generate Executive Summary** after running 8-10 agents
5. **Iterate based on insights** - some findings may change priorities

---

**💡 Pro Tip:** Don't run all 40 agents at once. Start with these 8, review insights, and let the findings guide which additional agents to run. Quality over quantity!

**⚠️ Important:** If market sizing or competitive analysis reveal major concerns, pause and reassess before investing ${budget} in marketing.
"""

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AgentLaunch AI - Backend Server Starting...")
    print("="*60)
    print(f"Server: http://localhost:5000")
    
    # Count total agents
    total_agents = sum(len(domain['agents']) for domain in DOMAINS.values())
    print(f"Domains: {len(DOMAINS)} domains loaded")
    print(f"Agents: {total_agents} specialized agents loaded")
    print(f"API: NVIDIA (Qwen 3)")
    print("="*60 + "\n")
    
    if not DOMAINS:
        print("WARNING: No domains loaded! Check agents_config.json file.")
    
    app.run(debug=True, port=5000)