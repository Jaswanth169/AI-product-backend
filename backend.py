from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
from datetime import datetime
import traceback

# ------------------------------
# Initialize Flask App
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Configure CORS for your frontend
# ------------------------------
CORS(app, resources={r"/api/*": {"origins": "https://zealous-pond-0fc6e090f.3.azurestaticapps.net"}})

# ------------------------------
# Initialize NVIDIA OpenAI Client
# ------------------------------
api_key = os.environ['NVIDIA_API_KEY']

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# ------------------------------
# Load agents configuration
# ------------------------------
def load_agents_config():
    try:
        with open('agents_config.json', 'r') as f:
            config = json.load(f)
        return config.get('domains', {})
    except FileNotFoundError:
        print("ERROR: agents_config.json not found!")
        return {}
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in agents_config.json: {e}")
        return {}

DOMAINS = load_agents_config()

# ------------------------------
# In-memory storage for projects
# ------------------------------
projects = {}

# ------------------------------
# API Endpoints
# ------------------------------

@app.route('/api/domains', methods=['GET'])
def get_domains():
    """Get all domains and their agents"""
    return jsonify({"domains": DOMAINS}), 200

@app.route('/api/project/create', methods=['POST'])
def create_project():
    """Create a new project"""
    try:
        data = request.json
        required_fields = [
            'product_name', 'product_description', 'product_features', 
            'target_price', 'budget', 'industry'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        project_id = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

        print(f"Project Created: {project_id} | Product: {data['product_name']} | Industry: {data['industry']}")
        return jsonify({"project_id": project_id, "message": "Project created successfully"}), 200

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

        # Find agent
        agent = None
        for domain_data in DOMAINS.values():
            for a in domain_data['agents']:
                if a['id'] == agent_id:
                    agent = a
                    break
            if agent:
                break

        if not agent:
            return jsonify({"error": "Agent not found"}), 404

        project = projects[project_id]
        features_str = ', '.join(project['product_features']) if isinstance(project['product_features'], list) else project['product_features']

        # Replace placeholders in agent prompt
        prompt = agent['prompt']
        for key, val in {
            "{product_name}": project['product_name'],
            "{product_description}": project['product_description'],
            "{product_features}": features_str,
            "{target_price}": str(project['target_price']),
            "{budget}": str(project['budget']),
            "{industry}": project['industry']
        }.items():
            prompt = prompt.replace(key, val)

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

        print(f"Sending request to NVIDIA API for agent {agent['name']}...")

        completion = client.chat.completions.create(
            model="qwen/qwen3-next-80b-a3b-thinking",
            messages=[{"role": "user", "content": context}],
            temperature=0.6,
            top_p=0.7,
            max_tokens=4096,
            stream=False
        )

        result = completion.choices[0].message.content or "No response generated."

        projects[project_id]["agent_results"][agent_id] = {
            "agent_name": agent["name"],
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }

        print(f"Agent {agent['name']} completed successfully")
        return jsonify({
            "agent_id": agent_id,
            "agent_name": agent["name"],
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }), 200

    except Exception as e:
        print(f"ERROR: Error running agent: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/project/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get project details and results"""
    if project_id not in projects:
        return jsonify({"error": "Project not found"}), 404
    return jsonify(projects[project_id]), 200

@app.route('/api/project/<project_id>/summary', methods=['GET'])
def get_project_summary(project_id):
    """Get AI-generated summary of all agent results"""
    try:
        if project_id not in projects:
            return jsonify({"error": "Project not found"}), 404

        project = projects[project_id]
        agent_results = project.get("agent_results", {})
        if not agent_results:
            return jsonify({"error": "No agent results available. Please run at least one agent first."}), 400

        results_summary = "\n\n".join([f"{data['agent_name']}:\n{data['result']}" for data in agent_results.values()])

        summary_prompt = f"""
Based on all the agent analyses below for {project['product_name']}, create a comprehensive executive summary including:

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

        completion = client.chat.completions.create(
            model="qwen/qwen3-next-80b-a3b-thinking",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.6,
            top_p=0.7,
            max_tokens=4096,
            stream=False
        )

        summary = completion.choices[0].message.content
        print(f"Executive summary generated successfully for {project_id}")

        return jsonify({
            "project_id": project_id,
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"ERROR: Error generating summary: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Run Flask App
# ------------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("AgentLaunch AI - Backend Server Starting...")
    print("="*60)
    print(f"Server: http://localhost:5000")
    total_agents = sum(len(domain['agents']) for domain in DOMAINS.values())
    print(f"Domains loaded: {len(DOMAINS)} | Total agents: {total_agents}")
    print(f"API: NVIDIA (Qwen 3)")
    print("="*60 + "\n")

    if not DOMAINS:
        print("WARNING: No domains loaded! Check agents_config.json file.")
    
    app.run(debug=True, port=5000)
