from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

from dotenv import load_dotenv

topic ="Medical assistant using Generative AI"

#Topic 1
llm = LLM(model="gpt-4")

#Topic 2

search_tool = SerperDevTool(n=10)

# Agent 1

senior_research_analyst = Agent(
    role = "Senior Research Analyst",
    goal = f"Reasearch, analyze, and synthesize comprehensive information on {topic} from reliable web sources.",
    backstory = """ You're an expert research analyst with advanced web research skills,
        You excel at finding, analyzing, and synthesizing information from
        across the internet using search tools. You're skilled at
        distinguishing reliable sources from unreliable ones,
        fact-checking, cross-referencing information, and
        identifying key patterns and insights. You provide
        well-organized research briefs with proper citations
        and source verification. Your analysis includes both
        raw data and interpreted insights, making complex
        information accessible and actionable.""",
    allow_delegation=False,
    verbose=True,
    tools=[search_tool],
    llm=llm,
)

# Agent 2 - Content Writer

content_writer = Agent(
    role = " Content Writer",
    goal = "Transform research findings into a well-structured, engaging article on the topic of Medical assistant using Generative AI.",
    backstory = """You’re a skilled content writer specialized in creating 
        engaging, accessible content from technical research. 
        You work closely with the Senior Research Analyst and excel at maintaining the perfect 
        balance between informative and entertaining writing, 
        while ensuring all facts and citations from the research 
        are properly incorporated. You have a talent for making 
        complex topics approachable without oversimplifying them..""",
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Reasearch Analyst Task

research_task = Task(
    description =("""Conduct comprehensive research on {topic} including:
        1. Recent developments and news
        2. Key industry trends and innovations
        3. Expert opinions and analyses
        4. Statistical data and market insights
        5. Evaluate source credibility and fact-check all information
        6. Organize findings into a structured research brief
        7. Include all relevant citations and sources."""),
    expected_output =("""A detailed research report containing:
        - Executive summary of key findings
        - Comprehensive analysis of current trends and developments
        - List of verified facts and statistics
        - All citations and links to original sources
        - Clear categorization of main themes and patterns
        Please format with clear sections and bullet points for easy reference."""),
    agent=senior_research_analyst,
)

# Content Writing Task

writing_task = Task(
    description = ("""Using the research brief provided, create an engaging blog post that:
        1. Transforms technical information into accessible content
        2. Maintains all factual accuracy and citations from the research
        3. Includes:
        - Attention-grabbing introduction
        - Well-structured body sections with clear headings
        - Compelling conclusion
        4. Preserves all source citations in [Source: URL] format
        5. Includes a References section at the end"""),
    expected_output =("""A polished blog post in markdown format that:
        - Engages readers while maintaining accuracy
        - Contains properly structured sections
        - Includes inline citations hyperlinked to the original source URL
        - Presents information in an accessible yet informative way
        - Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections"""),
    agent=content_writer,
)

crew = Crew(
    agents = [senior_research_analyst, content_writer],
    tasks = [research_task, writing_task],
    verbose=True,
)

results = crew.kickoff(inputs={"topic": topic})
print(results)

