# Call Center AI AInsights

A Flask-based dashboard for analyzing customer service call recordings, evaluating agent performance, and tracking team compliance metrics.

## Features

- **Call Analysis**: Upload and analyze customer service call recordings
- **Agent Performance**: Track individual agent performance metrics over time
- **Team Compliance**: Monitor team compliance with company policies and regulations
- **Visual Analytics**: Interactive charts and visualizations for performance trends
- **Leaderboards**: Compare agent and team performance with ranking systems
- **Downloadable Reports**: Export analysis data in JSON or CSV format

## How It Works

1. **Upload Call Recordings**: Users upload audio recordings of customer service calls along with agent and team information
2. **AI Analysis**: The system uses Google's Gemini 2.0 Flash model to analyze call content, agent performance, and compliance
3. **Dashboard View**: Results are displayed in an intuitive dashboard with detailed breakdowns and visualizations
4. **Agent & Team Tracking**: Performance is tracked over time for individual agents and across teams

## Requirements

- Python 3.8+
- Flask
- Google GenerativeAI API Key
- Modern web browser

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/call-analysis-dashboard.git
   cd call-analysis-dashboard
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. Create the data directory structure:
   ```
   mkdir -p data
   echo "[]" > data/analysis_results.json
   ```

6. Run the application:
   ```
   flask run
   ```

7. Access the dashboard at http://127.0.0.1:5000/

## System Architecture

The system consists of the following components:

- **Flask Backend**: Handles routing, file processing, and data management
- **Google Gemini AI**: Processes call recordings and generates analysis
- **Bootstrap UI**: Responsive front-end interface
- **Chart.js**: Interactive data visualizations
- **JSON Storage**: Simple file-based storage for analysis results

## Data Privacy Considerations

- Audio files are processed by Google's Gemini AI and are subject to Google's privacy policies
- No recordings are stored permanently on the server
- Analysis results do not include full transcripts but only summaries and metrics
- Consider adding authentication for production deployments

## Extending the System

- **Database Integration**: Replace JSON storage with a proper database system
- **User Authentication**: Add login system for secure access
- **Batch Processing**: Enable processing multiple calls at once
- **Advanced Analytics**: Implement more sophisticated reporting and trend analysis
- **Real-time Monitoring**: Add websocket support for live updates

## License

[MIT License](LICENSE)
