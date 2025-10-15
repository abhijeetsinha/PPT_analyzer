import os
import glob
import langgraph
import langchain
import langchain_community
import langchain_core
import langchain_openai
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import pdfplumber

load_dotenv()

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_chat_model = os.getenv("AZURE_OPENAI_CHAT_MODEL")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

#print("Azure OpenAI API Key:", azure_openai_api_key)
#print("Azure OpenAI Endpoint:", azure_openai_endpoint)
#print("Azure OpenAI Chat Model:", azure_openai_chat_model)
#print("Azure OpenAI API Version:", azure_openai_api_version)

llm = AzureChatOpenAI(
    model=azure_openai_chat_model,
    api_key=azure_openai_api_key,
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint,
    temperature=0
)
print("LLM initialized successfully.")

class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    skill_match_score: int
    final_decision: str
    pdf_file_path: str  # Add this to track which PDF we're processing


def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def parse_resume(state: State) -> State:
    # Get the PDF file path from state, or use default
    pdf_file = state["pdf_file_path"]
    resume_text = extract_pdf_text(pdf_file)

    # Create a prompt to extract information from the resume. It takes resume text as input. The value will be supplied later.
    prompt = ChatPromptTemplate.from_template(
        "Extract text from the resume: {resume_text}. Parse the resume and return the text content only. Do not add any extra information."
    )
    # Create a chain to invoke the LLM with the prompt. The paramater of the prompt is supplied at the time of invoking the chain.
    chain = prompt | llm | StrOutputParser()

    # Invoke the chain with parameters (resume_text in this case)
    state["application"] = chain.invoke({"resume_text": resume_text})
    print(f"Parsed resume: {state['application']}\n")
    return state


def categorize_experience_level(state: State) -> State:
    # Get resume text from state instead of reading file
    resume_text = state["application"]
    
    prompt = ChatPromptTemplate.from_template(
        "Given the resume text: {resume_text}, categorize the experience level as one of the following: 'Junior', 'Mid-level', or 'Senior'. Return only the experience level."
    )

    chain = prompt | llm | StrOutputParser()
    experience_level = chain.invoke({"resume_text": resume_text})
    print(f"Experience level: {experience_level}\n")
    
    # Return updated state in a differet way by using unpacking of dictionary.
    return {
        **state, # dictionary unpacking
        "experience_level": experience_level
    }


def analyze_skills(state: State) -> State:
    with open("job_description.txt","r") as f:
        job_description = f.read()

    # Create a more specific prompt that returns structured output
    prompt = ChatPromptTemplate.from_template(
        """Given the resume text: {resume_text} and the job description: {job_description}, 
        analyze the skills mentioned in the resume and compare them to the skills required in the job description.
        
        Return your response in this exact format:
        SKILL_MATCH: [Your detailed analysis of skill match]
        SKILL_MATCH_SCORE: [A score between 0-100]
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "resume_text": state["application"],
        "job_description": job_description
    })
    
    # Parse the response to extract both values
    lines = response.strip().split('\n')
    skill_match = ""
    skill_match_score = 0
    
    for line in lines:
        if line.startswith("SKILL_MATCH:"):
            skill_match = line.replace("SKILL_MATCH:", "").strip()
            print(f"Skill match analysis: {skill_match}\n")
        elif line.startswith("SKILL_MATCH_SCORE:"):
            try:
                skill_match_score = int(line.replace("SKILL_MATCH_SCORE:", "").strip())
                print(f"Skill match score: {skill_match_score}\n")
            except ValueError:
                skill_match_score = 0

    # Update state with skill match information
    state["skill_match"] = skill_match
    state["skill_match_score"] = skill_match_score
    return state


def route_decision(state: State) -> str:
    prompt = ChatPromptTemplate.from_template(
        """Based on the experience level: {experience_level} and skill match score: {skill_match_score}, 
        make a final hiring decision. Return only one of the following values: 'Schedule interview', 'Need manual review', or 'Reject'.
        'Schedule interview' should be suggested when skill_match score is above 70 and experience level is mid-level or senior.
        'Need manual review' should be suggested when skill_match score is between 40 and 70 but experience level is senior.
        'Reject' should be suggested when skill_match score is below 40 or experience level is junior.
        
        IMPORTANT: Start your response with exactly one of these three phrases: 'Schedule interview', 'Need manual review', or 'Reject'.
        Then you can add explanation after a dash.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "experience_level": state["experience_level"],
        "skill_match_score": state["skill_match_score"]
    })

    print(f"Routing decision response: {response}")
    
    # Check if response starts with any of the expected phrases
    if response.startswith("Schedule interview"):
        return "node_schedule_interview"
    elif response.startswith("Need manual review"):
        return "node_manual_review"
    elif response.startswith("Reject"):
        return "node_reject"
    else:
        print(f"Unexpected response from LLM: {response}")
        print("Defaulting to reject...")
        return "node_reject"


def schedule_interview(state: State) -> State:
    # Implement scheduling logic here
    print("Scheduling interview...")
    state["final_decision"]="Schedule interview"
    return state


def manual_review(state: State) -> State:
    # Implement manual review logic here
    print("Flagging for manual review...")
    state["final_decision"]="Need manual review"
    return state


def reject(state: State) -> State:
    # Implement rejection logic here
    print("Rejecting application...")
    state["final_decision"]="Reject"
    return state


graph = StateGraph(State)
graph.add_node("node_parse_resume", parse_resume)
graph.add_node("node_categorize_experience_level", categorize_experience_level)
graph.add_node("node_analyze_skills", analyze_skills)
graph.add_node("node_schedule_interview", schedule_interview)
graph.add_node("node_manual_review", manual_review)
graph.add_node("node_reject", reject)

graph.add_edge(START, "node_parse_resume")
graph.add_edge("node_parse_resume", "node_categorize_experience_level")
graph.add_edge("node_categorize_experience_level", "node_analyze_skills")
graph.add_conditional_edges("node_analyze_skills", route_decision)
graph.add_edge("node_schedule_interview", END)
graph.add_edge("node_manual_review", END)
graph.add_edge("node_reject", END)

app = graph.compile()

def process_folder(folder_path):
    """Process all PDF files in the given folder"""
    import os
    import glob
    
    # Get all PDF files in the folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    print(f"PDF files found: {pdf_files}")
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    all_results = []
    
    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(pdf_file)}")
        print(f"{'='*60}")
        
        # Create initial state for this PDF
        initial_state = {
            "application": "",
            "experience_level": "",
            "skill_match": "",
            "skill_match_score": 0,
            "final_decision": "",
            "pdf_file_path": pdf_file
        }
        
        try:
            # Process this PDF
            result = app.invoke(initial_state)
            
            # Store result with filename
            result_with_file = {
                "filename": os.path.basename(pdf_file),
                "final_decision": result['final_decision'],
                "experience_level": result['experience_level'],
                "skill_match_score": result['skill_match_score'],
                "skill_match": result['skill_match']
            }
            all_results.append(result_with_file)
            
            # Print individual result
            print(f"\n📋 RESULT for {os.path.basename(pdf_file)}:")
            print(f"Final Decision: {result['final_decision']}")
            print(f"Experience Level: {result['experience_level']}")
            print(f"Skill Match Score: {result['skill_match_score']}")
            
        except Exception as e:
            print(f"❌ ERROR processing {os.path.basename(pdf_file)}: {e}")
            all_results.append({
                "filename": os.path.basename(pdf_file),
                "error": str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY OF ALL RESUMES")
    print(f"{'='*60}")
    
    for result in all_results:
        if "error" in result:
            print(f"❌ {result['filename']}: ERROR - {result['error']}")
        else:
            print(f"✅ {result['filename']}: {result['final_decision']} (Score: {result['skill_match_score']})")
    
    return all_results

# Get folder path from user or use current directory
folder_path = input("Enter folder path (or press Enter for current directory). Make sure that the resumes are NOT protected files: ").strip()
if not folder_path:
    folder_path = "."  # Current directory

# Process all PDFs in the folder
results = process_folder(folder_path)

# Save the graph visualization as a PNG file
try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("workflow_graph.png", "wb") as f:
        f.write(png_data)
    print("Graph saved as 'workflow_graph.png' - you can open it to view the workflow!")
except Exception as e:
    print(f"Could not generate graph image: {e}")

# Alternative: Print the mermaid text representation
try:
    mermaid_text = app.get_graph().draw_mermaid()
    print("\nWorkflow Graph (Mermaid syntax):")
    print(mermaid_text)
except Exception as e:
    print(f"Could not generate mermaid text: {e}")

# display(Image(app.get_graph().draw_mermaid_png()))