# Langgraph demo

## Pre-requisites

- Create python environment : python -m venv venv
- install all required packages: pip install -r requirements.txt
- Add all environment variables
- Load environment variables in app.py

## High level steps

### Resume Analysis Workflow

This application uses LangGraph to create an automated resume screening pipeline that processes multiple PDF resumes and makes hiring decisions.

#### 1. **Input Processing**

- User provides a folder path containing PDF resumes. **You must provide non-protected PDF**.
- System discovers all `.pdf` files in the specified directory
- Each PDF is processed individually through the workflow

#### 2. **Resume Parsing (Node: `parse_resume`)**

- Extracts text content from PDF files using `pdfplumber`
- Sends extracted text to Azure OpenAI for content cleaning
- Stores processed resume text in the workflow state

#### 3. **Experience Level Categorization (Node: `categorize_experience_level`)**

- Analyzes resume content to determine experience level
- Classifies candidates as: `Junior`, `Mid-level`, or `Senior`
- Uses LLM to provide reasoning for the categorization

#### 4. **Skills Analysis (Node: `analyze_skills`)**

- Compares resume skills against job description requirements
- Reads job requirements from `job_description.txt`
- Generates detailed skill match analysis and numerical score (0-100)

#### 5. **Decision Routing (Function: `route_decision`)**

- Makes hiring decisions based on experience level and skill match score
- Routes to one of three outcomes:
  - **Schedule Interview**: Score >70 + Mid/Senior level
  - **Manual Review**: Score 40-70 + Senior level
  - **Reject**: Score <40 or Junior level

#### 6. **Final Decision Nodes**

- **Schedule Interview**: Marks candidate for immediate interview
- **Manual Review**: Flags for human review
- **Reject**: Automatically rejects the application

#### 7. **Results & Reporting**

- Displays individual results for each processed resume
- Generates summary report of all candidates
- Creates workflow visualization graph (`workflow_graph.png`)
- Exports workflow structure in Mermaid format

### Workflow Architecture

```mermaid
START → Parse Resume → Categorize Experience → Analyze Skills → Route Decision
                                                                      ↓
                                                    ┌─────────────────┼─────────────────┐
                                                    ↓                 ↓                 ↓
                                              Schedule          Manual Review        Reject
                                              Interview              ↓                 ↓
                                                    ↓                 ↓                 ↓
                                                  END               END               END
```

### Key Features

- **Batch Processing**: Handles multiple PDF resumes automatically
- **State Management**: Maintains candidate data throughout the workflow
- **Conditional Logic**: Smart routing based on multiple criteria
- **Error Handling**: Graceful handling of processing failures
- **Comprehensive Reporting**: Individual and summary results
- **Visual Workflow**: Generates workflow diagrams for transparency
