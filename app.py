from flask import Flask, render_template, request, jsonify, send_file
import base64
import os
import json
import re
import xml.etree.ElementTree as ET
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import time
import random

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: No Google API key found. Set GOOGLE_API_KEY in your .env file.")

app = Flask(__name__)

def initialize_database():
    """Initialize database file if it doesn't exist or is invalid"""
    try:
        # Check if data directory exists, create if not
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Check if database file exists, create if not
        if not os.path.exists('data/analysis_results.json'):
            with open('data/analysis_results.json', 'w') as f:
                json.dump([], f)
        else:
            # Verify the file contains valid JSON
            with open('data/analysis_results.json', 'r') as f:
                content = f.read().strip()
                if content:  # Only try to parse if there's content
                    try:
                        json.loads(content)
                    except json.JSONDecodeError:
                        # Backup corrupted file
                        backup_path = f'data/analysis_results_backup_{int(time.time())}.json'
                        os.rename('data/analysis_results.json', backup_path)
                        print(f"Corrupted database backed up to {backup_path}")
                        # Create new empty file
                        with open('data/analysis_results.json', 'w') as f:
                            json.dump([], f)
                else:
                    # File is empty, initialize with empty array
                    with open('data/analysis_results.json', 'w') as f:
                        json.dump([], f)
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        # Ensure a valid database exists no matter what
        with open('data/analysis_results.json', 'w') as f:
            json.dump([], f)

# Initialize database on startup
initialize_database()

def clean_recommendation(recommendation):
    """
    Remove any surrounding quotes or dictionary-like text
    """
    # Remove dictionary notation
    recommendation = re.sub(r"^[\{']*(recommendation[':]*)?\s*['\"]?", '', str(recommendation))
    recommendation = re.sub(r"['\"}]*$", '', recommendation)
    
    # Trim whitespace
    return recommendation.strip()

def extract_subcategory_score(text, keyword, overall_score):
    """
    Extract a subcategory score from text based on keyword mentions.
    If no clear score is found, estimate based on sentiment and overall score.
    """
    try:
        # Look for explicit scores in the text
        score_pattern = r'(\b' + keyword + r'.*?)\b([0-9]\.?[0-9]?|10|[0-9])/10\b'
        match = re.search(score_pattern, text.lower())
        if match:
            score = float(match.group(2))
            return score
            
        # If no explicit score, use keyword proximity analysis
        keyword_lower = keyword.lower()
        if keyword_lower in text.lower():
            # Check for positive/negative sentiment around the keyword
            # Extract the sentence containing the keyword
            sentences = re.split(r'[.!?]', text)
            relevant_sentences = [s for s in sentences if keyword_lower in s.lower()]
            
            if relevant_sentences:
                relevant_text = ' '.join(relevant_sentences)
                
                # Check for positive keywords
                positive_keywords = ['excellent', 'good', 'great', 'impressive', 'exceptional', 'effective']
                negative_keywords = ['poor', 'inadequate', 'failed', 'weak', 'lacking', 'insufficient']
                
                positive_count = sum(1 for word in positive_keywords if word in relevant_text.lower())
                negative_count = sum(1 for word in negative_keywords if word in relevant_text.lower())
                
                # Adjust score based on sentiment
                sentiment_modifier = (positive_count - negative_count) * 0.5
                # Derive from overall score but adjusted by sentiment
                adjusted_score = max(1, min(10, overall_score + sentiment_modifier))
                return round(adjusted_score, 1)
        
        # Default: estimate based on overall score with slight random variation
        # This ensures subcategories are related to but not identical to overall score
        variation = random.uniform(-0.7, 0.7)
        return round(max(1, min(10, overall_score + variation)), 1)
    
    except Exception:
        # Fallback to a reasonable default
        return round(max(1, min(10, overall_score - 0.2)), 1)

def parse_xml_to_json(xml_text):
    """
    Parse XML response and convert to JSON with detailed subcategories
    """
    try:
        # Remove any XML declaration or processing instructions
        xml_text = xml_text.split('<call_analysis>', 1)[-1].split('</call_analysis>', 1)[0]
        xml_text = f'<call_analysis>{xml_text}</call_analysis>'
        
        # Parse the XML
        root = ET.fromstring(xml_text)
        
        # Create a dictionary to store parsed data
        parsed_data = {}
        
        # Extract sections
        for section in root:
            tag = section.tag
            if len(section) > 0:
                # Handle nested elements
                subsections = {}
                for subsection in section:
                    subsections[subsection.tag] = str(subsection.text).strip() if subsection.text else ""
                parsed_data[tag] = subsections
            else:
                # Handle simple text elements
                parsed_data[tag] = str(section.text).strip() if section.text else ""
        
        # Clean recommendations
        if 'recommendations' in parsed_data:
            if isinstance(parsed_data['recommendations'], dict):
                parsed_data['recommendations'] = parsed_data['recommendations'].get('recommendation', '')
            parsed_data['recommendations'] = clean_recommendation(parsed_data['recommendations'])
        
        # Add subcategories for agent_evaluation with default values
        if 'agent_evaluation' in parsed_data and isinstance(parsed_data['agent_evaluation'], dict):
            # Extract overall score
            overall_score = float(parsed_data['agent_evaluation'].get('score', 0)) 
            
            # Add subcategories if not present
            subcategories = {
                'communication': extract_subcategory_score(parsed_data['agent_evaluation'].get('justification', ''), 'communication', overall_score),
                'problem_solving': extract_subcategory_score(parsed_data['agent_evaluation'].get('justification', ''), 'problem-solving', overall_score),
                'empathy': extract_subcategory_score(parsed_data['agent_evaluation'].get('justification', ''), 'empathy', overall_score),
                'adherence': extract_subcategory_score(parsed_data['agent_evaluation'].get('justification', ''), 'protocol', overall_score)
            }
            
            parsed_data['agent_evaluation']['subcategories'] = subcategories
        
        # Add subcategories for audit_and_compliance with default values
        if 'audit_and_compliance' in parsed_data and isinstance(parsed_data['audit_and_compliance'], dict):
            # Extract overall score
            overall_score = float(parsed_data['audit_and_compliance'].get('score', 0))
            
            # Add subcategories if not present
            subcategories = {
                'communication_protocols': extract_subcategory_score(parsed_data['audit_and_compliance'].get('details', ''), 'communication protocol', overall_score),
                'regulations_compliance': extract_subcategory_score(parsed_data['audit_and_compliance'].get('details', ''), 'regulation', overall_score),
                'sensitive_info': extract_subcategory_score(parsed_data['audit_and_compliance'].get('details', ''), 'sensitive information', overall_score),
                'resolution': extract_subcategory_score(parsed_data['audit_and_compliance'].get('details', ''), 'resolution', overall_score)
            }
            
            parsed_data['audit_and_compliance']['subcategories'] = subcategories
            
        return parsed_data
    
    except ET.ParseError as e:
        return {"error": f"Failed to parse the XML response: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

def process_call_recording(audio_file, agent_id, team_id):
    """
    Process the uploaded audio file using Gemini 2.0 Flash model
    """
    try:
        # Check if API key is configured
        if not os.getenv("GOOGLE_API_KEY"):
            return {"error": "Google API key not configured. Please set GOOGLE_API_KEY in your .env file."}
            
        # Read the uploaded file
        audio_bytes = audio_file.read()
        encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')

        # Prepare the prompt
        prompt_text = """You are an AI assistant specialized in analyzing customer service calls. You will receive an audio recording of a call, and your task is to process it according to the following steps:

1. **Transcribe the audio to text.** If the audio quality is poor and prevents accurate transcription, indicate this in the `<call_summary>` section and skip to step 4.

2. **Provide a concise summary of the call.** Include key details such as the customer's issue, the agent's response, and the resolution (if any). Place the summary within the `<call_summary>` tags.

3. **Perform an audit and compliance check.**
   - Evaluate the call against relevant regulations and company policies.
   - Assign an audit score out of 10 based on:
     a) Adherence to company communication protocols
     b) Compliance with industry regulations
     c) Handling of sensitive customer information
     d) Resolution effectiveness
   - Provide specific details about any compliance issues or strengths.
   - Place the score within `<score>` tags and details within `<details>` tags under the `<audit_and_compliance>` section.

4. **Rate the agent's performance on a scale of 1-10.**  Consider factors such as clarity of communication, problem-solving skills, empathy, and adherence to company protocols. Include a brief justification for the score. Place the score within the `<score>` tags and the justification within the `<justification>` tags under the `<agent_evaluation>` section.

5. **Provide recommended actions for improvement.** Offer specific and actionable suggestions for the agent to enhance their performance in future calls. Place the recommendations within the `<recommendations>` tags.

6. **Structure the output in XML format.** Ensure all information is placed within the appropriate tags.

Audio Recording:"""

        # Prepare the model input
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate content
        response = model.generate_content(
            [
                prompt_text,
                {
                    'mime_type': 'audio/mpeg',
                    'data': encoded_audio
                }
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=8192
            )
        )

        # Parse response
        parsed_data = parse_xml_to_json(response.text)
        
        # Add metadata
        parsed_data['timestamp'] = datetime.now().isoformat()
        parsed_data['agent_id'] = agent_id
        parsed_data['team_id'] = team_id
        parsed_data['filename'] = audio_file.filename
        
        # Save to database
        save_analysis_result(parsed_data)
        
        return parsed_data

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def save_analysis_result(analysis_result):
    """Save analysis result to database file"""
    try:
        results = get_all_analysis_results()
        results.append(analysis_result)
        
        with open('data/analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        print(f"Error saving to database: {str(e)}")

def get_all_analysis_results():
    """Get all analysis results from database"""
    try:
        with open('data/analysis_results.json', 'r') as f:
            content = f.read().strip()
            # Check if file is empty or just whitespace
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file doesn't exist or has invalid JSON, return empty list
        # Also create a new empty database file
        with open('data/analysis_results.json', 'w') as f:
            json.dump([], f)
        return []

def get_team_metrics():
    """Calculate team-level metrics from all call analyses with subcategories"""
    results = get_all_analysis_results()
    if not results:
        empty_subcats = {
            'communication': 0, 'problem_solving': 0, 'empathy': 0, 'adherence': 0,
            'communication_protocols': 0, 'regulations_compliance': 0, 'sensitive_info': 0, 'resolution': 0
        }
        return {
            'team_metrics': {'overall': {'avg_agent_score': 0, 'avg_audit_score': 0, 'call_count': 0, 'recent_calls': 0, 'subcategories': empty_subcats}}, 
            'agent_metrics': {}
        }
    
    # Convert to DataFrame for easier analysis
    try:
        df = pd.json_normalize(results)
        
        # Handle potential missing columns
        if 'agent_evaluation.score' not in df.columns:
            df['agent_evaluation.score'] = 0
        if 'audit_and_compliance.score' not in df.columns:
            df['audit_and_compliance.score'] = 0
            
        # Create columns for subcategories if they don't exist
        agent_subcats = ['communication', 'problem_solving', 'empathy', 'adherence']
        for subcat in agent_subcats:
            col_name = f'agent_evaluation.subcategories.{subcat}'
            if col_name not in df.columns:
                df[col_name] = 0
                
        audit_subcats = ['communication_protocols', 'regulations_compliance', 'sensitive_info', 'resolution']
        for subcat in audit_subcats:
            col_name = f'audit_and_compliance.subcategories.{subcat}'
            if col_name not in df.columns:
                df[col_name] = 0
        
        # Convert score columns to numeric
        df['agent_evaluation.score'] = pd.to_numeric(df['agent_evaluation.score'], errors='coerce')
        df['audit_and_compliance.score'] = pd.to_numeric(df['audit_and_compliance.score'], errors='coerce')
        
        # Convert subcategory columns to numeric
        for subcat in agent_subcats:
            col_name = f'agent_evaluation.subcategories.{subcat}'
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            
        for subcat in audit_subcats:
            col_name = f'audit_and_compliance.subcategories.{subcat}'
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        # Calculate team metrics
        team_metrics = {}
        
        # Overall team metrics
        team_metrics['overall'] = {
            'avg_agent_score': round(df['agent_evaluation.score'].mean(), 2) if not df['agent_evaluation.score'].isna().all() else 0,
            'avg_audit_score': round(df['audit_and_compliance.score'].mean(), 2) if not df['audit_and_compliance.score'].isna().all() else 0,
            'call_count': len(df),
            'recent_calls': len(df[pd.to_datetime(df['timestamp'], errors='coerce') > pd.Timestamp.now() - pd.Timedelta(days=7)]),
            'subcategories': {
                # Agent subcategories
                'communication': round(df['agent_evaluation.subcategories.communication'].mean(), 2) if 'agent_evaluation.subcategories.communication' in df.columns and not df['agent_evaluation.subcategories.communication'].isna().all() else 0,
                'problem_solving': round(df['agent_evaluation.subcategories.problem_solving'].mean(), 2) if 'agent_evaluation.subcategories.problem_solving' in df.columns and not df['agent_evaluation.subcategories.problem_solving'].isna().all() else 0,
                'empathy': round(df['agent_evaluation.subcategories.empathy'].mean(), 2) if 'agent_evaluation.subcategories.empathy' in df.columns and not df['agent_evaluation.subcategories.empathy'].isna().all() else 0,
                'adherence': round(df['agent_evaluation.subcategories.adherence'].mean(), 2) if 'agent_evaluation.subcategories.adherence' in df.columns and not df['agent_evaluation.subcategories.adherence'].isna().all() else 0,
                # Audit subcategories
                'communication_protocols': round(df['audit_and_compliance.subcategories.communication_protocols'].mean(), 2) if 'audit_and_compliance.subcategories.communication_protocols' in df.columns and not df['audit_and_compliance.subcategories.communication_protocols'].isna().all() else 0,
                'regulations_compliance': round(df['audit_and_compliance.subcategories.regulations_compliance'].mean(), 2) if 'audit_and_compliance.subcategories.regulations_compliance' in df.columns and not df['audit_and_compliance.subcategories.regulations_compliance'].isna().all() else 0,
                'sensitive_info': round(df['audit_and_compliance.subcategories.sensitive_info'].mean(), 2) if 'audit_and_compliance.subcategories.sensitive_info' in df.columns and not df['audit_and_compliance.subcategories.sensitive_info'].isna().all() else 0,
                'resolution': round(df['audit_and_compliance.subcategories.resolution'].mean(), 2) if 'audit_and_compliance.subcategories.resolution' in df.columns and not df['audit_and_compliance.subcategories.resolution'].isna().all() else 0
            }
        }
        
        # Team-specific metrics
        if 'team_id' in df.columns:
            for team in df['team_id'].unique():
                team_df = df[df['team_id'] == team]
                team_metrics[team] = {
                    'avg_agent_score': round(team_df['agent_evaluation.score'].mean(), 2) if not team_df['agent_evaluation.score'].isna().all() else 0,
                    'avg_audit_score': round(team_df['audit_and_compliance.score'].mean(), 2) if not team_df['audit_and_compliance.score'].isna().all() else 0,
                    'call_count': len(team_df),
                    'recent_calls': len(team_df[pd.to_datetime(team_df['timestamp'], errors='coerce') > pd.Timestamp.now() - pd.Timedelta(days=7)]),
                    'subcategories': {
                        # Agent subcategories
                        'communication': round(team_df['agent_evaluation.subcategories.communication'].mean(), 2) if 'agent_evaluation.subcategories.communication' in team_df.columns and not team_df['agent_evaluation.subcategories.communication'].isna().all() else 0,
                        'problem_solving': round(team_df['agent_evaluation.subcategories.problem_solving'].mean(), 2) if 'agent_evaluation.subcategories.problem_solving' in team_df.columns and not team_df['agent_evaluation.subcategories.problem_solving'].isna().all() else 0,
                        'empathy': round(team_df['agent_evaluation.subcategories.empathy'].mean(), 2) if 'agent_evaluation.subcategories.empathy' in team_df.columns and not team_df['agent_evaluation.subcategories.empathy'].isna().all() else 0,
                        'adherence': round(team_df['agent_evaluation.subcategories.adherence'].mean(), 2) if 'agent_evaluation.subcategories.adherence' in team_df.columns and not team_df['agent_evaluation.subcategories.adherence'].isna().all() else 0,
                        # Audit subcategories
                        'communication_protocols': round(team_df['audit_and_compliance.subcategories.communication_protocols'].mean(), 2) if 'audit_and_compliance.subcategories.communication_protocols' in team_df.columns and not team_df['audit_and_compliance.subcategories.communication_protocols'].isna().all() else 0,
                        'regulations_compliance': round(team_df['audit_and_compliance.subcategories.regulations_compliance'].mean(), 2) if 'audit_and_compliance.subcategories.regulations_compliance' in team_df.columns and not team_df['audit_and_compliance.subcategories.regulations_compliance'].isna().all() else 0,
                        'sensitive_info': round(team_df['audit_and_compliance.subcategories.sensitive_info'].mean(), 2) if 'audit_and_compliance.subcategories.sensitive_info' in team_df.columns and not team_df['audit_and_compliance.subcategories.sensitive_info'].isna().all() else 0,
                        'resolution': round(team_df['audit_and_compliance.subcategories.resolution'].mean(), 2) if 'audit_and_compliance.subcategories.resolution' in team_df.columns and not team_df['audit_and_compliance.subcategories.resolution'].isna().all() else 0
                    }
                }
        
        # Agent-specific metrics
        agent_metrics = {}
        if 'agent_id' in df.columns:
            for agent in df['agent_id'].unique():
                agent_df = df[df['agent_id'] == agent]
                agent_metrics[agent] = {
                    'avg_agent_score': round(agent_df['agent_evaluation.score'].mean(), 2) if not agent_df['agent_evaluation.score'].isna().all() else 0,
                    'avg_audit_score': round(agent_df['audit_and_compliance.score'].mean(), 2) if not agent_df['audit_and_compliance.score'].isna().all() else 0,
                    'call_count': len(agent_df),
                    'recent_calls': len(agent_df[pd.to_datetime(agent_df['timestamp'], errors='coerce') > pd.Timestamp.now() - pd.Timedelta(days=7)]),
                    'subcategories': {
                        # Agent subcategories
                        'communication': round(agent_df['agent_evaluation.subcategories.communication'].mean(), 2) if 'agent_evaluation.subcategories.communication' in agent_df.columns and not agent_df['agent_evaluation.subcategories.communication'].isna().all() else 0,
                        'problem_solving': round(agent_df['agent_evaluation.subcategories.problem_solving'].mean(), 2) if 'agent_evaluation.subcategories.problem_solving' in agent_df.columns and not agent_df['agent_evaluation.subcategories.problem_solving'].isna().all() else 0,
                        'empathy': round(agent_df['agent_evaluation.subcategories.empathy'].mean(), 2) if 'agent_evaluation.subcategories.empathy' in agent_df.columns and not agent_df['agent_evaluation.subcategories.empathy'].isna().all() else 0,
                        'adherence': round(agent_df['agent_evaluation.subcategories.adherence'].mean(), 2) if 'agent_evaluation.subcategories.adherence' in agent_df.columns and not agent_df['agent_evaluation.subcategories.adherence'].isna().all() else 0,
                        # Audit subcategories
                        'communication_protocols': round(agent_df['audit_and_compliance.subcategories.communication_protocols'].mean(), 2) if 'audit_and_compliance.subcategories.communication_protocols' in agent_df.columns and not agent_df['audit_and_compliance.subcategories.communication_protocols'].isna().all() else 0,
                        'regulations_compliance': round(agent_df['audit_and_compliance.subcategories.regulations_compliance'].mean(), 2) if 'audit_and_compliance.subcategories.regulations_compliance' in agent_df.columns and not agent_df['audit_and_compliance.subcategories.regulations_compliance'].isna().all() else 0,
                        'sensitive_info': round(agent_df['audit_and_compliance.subcategories.sensitive_info'].mean(), 2) if 'audit_and_compliance.subcategories.sensitive_info' in agent_df.columns and not agent_df['audit_and_compliance.subcategories.sensitive_info'].isna().all() else 0,
                        'resolution': round(agent_df['audit_and_compliance.subcategories.resolution'].mean(), 2) if 'audit_and_compliance.subcategories.resolution' in agent_df.columns and not agent_df['audit_and_compliance.subcategories.resolution'].isna().all() else 0
                    }
                }
        
        return {
            'team_metrics': team_metrics,
            'agent_metrics': agent_metrics
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        empty_subcats = {
            'communication': 0, 'problem_solving': 0, 'empathy': 0, 'adherence': 0,
            'communication_protocols': 0, 'regulations_compliance': 0, 'sensitive_info': 0, 'resolution': 0
        }
        return {
            'team_metrics': {'overall': {'avg_agent_score': 0, 'avg_audit_score': 0, 'call_count': 0, 'recent_calls': 0, 'subcategories': empty_subcats}}, 
            'agent_metrics': {}
        }

@app.route('/')
def index():
    """Main dashboard page"""
    metrics = get_team_metrics()
    results = get_all_analysis_results()
    return render_template('index.html', metrics=metrics, results=results)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload and analyze call recording"""
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        
        agent_id = request.form.get('agent_id', 'unknown')
        team_id = request.form.get('team_id', 'unknown')
        
        if file:
            result = process_call_recording(file, agent_id, team_id)
            return render_template('analysis.html', result=result)
    
    return render_template('upload.html')

@app.route('/api/analysis_results')
def api_analysis_results():
    """API endpoint for analysis results"""
    results = get_all_analysis_results()
    return jsonify(results)

@app.route('/api/team_metrics')
def api_team_metrics():
    """API endpoint for team metrics"""
    metrics = get_team_metrics()
    return jsonify(metrics)

@app.route('/analysis/<timestamp>')
def view_analysis(timestamp):
    """View a specific analysis"""
    results = get_all_analysis_results()
    result = next((item for item in results if item.get('timestamp') == timestamp), None)
    if result:
        return render_template('analysis.html', result=result)
    return "Analysis not found", 404

@app.route('/agent/<agent_id>')
def agent_dashboard(agent_id):
    """Agent-specific dashboard"""
    results = get_all_analysis_results()
    agent_results = [r for r in results if r.get('agent_id') == agent_id]
    metrics = get_team_metrics()
    agent_metrics = metrics.get('agent_metrics', {}).get(agent_id, {})
    return render_template('agent.html', agent_id=agent_id, results=agent_results, metrics=agent_metrics)

@app.route('/team/<team_id>')
def team_dashboard(team_id):
    """Team-specific dashboard"""
    results = get_all_analysis_results()
    team_results = [r for r in results if r.get('team_id') == team_id]
    metrics = get_team_metrics()
    team_metrics = metrics.get('team_metrics', {}).get(team_id, {})
    return render_template('team.html', team_id=team_id, results=team_results, metrics=team_metrics)

@app.route('/download/json')
def download_json():
    """Download all analysis data as JSON"""
    return send_file('data/analysis_results.json', as_attachment=True)

@app.route('/download/csv')
def download_csv():
    """Download all analysis data as CSV"""
    results = get_all_analysis_results()
    if not results:
        return "No data available", 404
    
    df = pd.json_normalize(results)
    csv_path = 'data/analysis_results.csv'
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)