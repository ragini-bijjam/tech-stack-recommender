# 🚀 Tech Stack Recommender

## Artificial Intelligence - Project 3

A content-based recommendation system that recommends suitable
career paths based on a user's technical skills.

The system uses TF-IDF and Cosine Similarity to compare the user's
skills with the skill requirements of different career roles.
## Project Objective
The objective of this project is to build a simple AI recommendation
system that:

1. Takes user skills as input.
2. Matches user skills with career-role attributes.
3. Calculates similarity between the user profile and career roles.
4. Ranks career paths according to similarity.
5. Displays the Top 3 recommendations.

## Technologies Used
- Python
- CSV
- TF-IDF
- Cosine Similarity
- Streamlit
- Mathematical vector operations

## System Workflow
User enters skills
↓
Input normalization
↓
Skill alias recognition
↓
Dataset validation
↓
TF-IDF vectorization
↓
Cosine similarity calculation
↓
Career ranking
↓
Top 3 recommendations
↓
Matched skills + skills to learn

## Recommendation Algorithm

### 1.Term Frequency (TF)
TF measures how frequently a skill occurs within a role.
TF = Number of times a term appears / Total number of terms

### 2.Inverse Document Frequency (IDF)
IDF measures how unique a skill is across all career roles.
IDF = log(Total documents / Documents containing the term)

### 3.TF-IDF
TF-IDF combines TF and IDF.
TF-IDF = TF × IDF
The resulting value represents the importance of a skill.

### 4.Cosine Similarity
Cosine similarity measures the similarity between the user skill
vector and each career-role vector.
Similarity = (A · B) / (||A|| × ||B||)
A higher score indicates a stronger match.

##  Dataset
The `raw_skills.csv` file contains career roles and their associated
technical skills.
Example:
Cloud Architect - AWS, Cloud Computing, Docker, Kubernetes 
Data Scientist - Python, SQL, Machine Learning, Statistics 
DevOps Engineer - AWS, Docker, Kubernetes, Linux 
ML Engineer - Python, Machine Learning, TensorFlow 

## How to Run
### Step 1 - Check Python
```bash
python --version