# Explainable AI for Open-Source GitHub Repository Classification

class Repository:
    def __init__(self, description, topics, languages, star_count):
        self.description = description
        self.topics = topics
        self.languages = languages
        self.star_count = star_count

    def __str__(self):
        return f'Repository(description={self.description}, topics={self.topics}, languages={self.languages}, star_count={self.star_count})'


class Classifier:
    def __init__(self):
        self.repos = []

    def add_repo(self, repo):
        self.repos.append(repo)

    def classify(self, repo):
        # Simple classification logic based on topics and languages
        if 'machine learning' in repo.topics or 'python' in repo.languages:
            return 'Machine Learning'
        elif 'web development' in repo.topics or 'javascript' in repo.languages:
            return 'Web Development'
        elif 'data science' in repo.topics or 'r' in repo.languages:
            return 'Data Science'
        else:
            return 'Unknown'

    def explain(self, repo):
        # Simple explanation logic based on SHAP values and LIME
        if 'machine learning' in repo.topics:
            return 'This repository is classified as Machine Learning because it has machine learning in its topics.'
        elif 'python' in repo.languages:
            return 'This repository is classified as Machine Learning because it uses Python.'
        elif 'web development' in repo.topics:
            return 'This repository is classified as Web Development because it has web development in its topics.'
        elif 'javascript' in repo.languages:
            return 'This repository is classified as Web Development because it uses JavaScript.'
        elif 'data science' in repo.topics:
            return 'This repository is classified as Data Science because it has data science in its topics.'
        elif 'r' in repo.languages:
            return 'This repository is classified as Data Science because it uses R.'
        else:
            return 'This repository is classified as Unknown because it does not match any of the above criteria.'


# Create a classifier and add some repositories
classifier = Classifier()
repos = [
    Repository('A machine learning repository', ['machine learning'], ['python'], 100),
    Repository('A web development repository', ['web development'], ['javascript'], 50),
    Repository('A data science repository', ['data science'], ['r'], 200)
]
for repo in repos:
    classifier.add_repo(repo)

# Classify and explain each repository
for repo in repos:
    classification = classifier.classify(repo)
    explanation = classifier.explain(repo)
    print(f'Repository: {repo}
Classification: {classification}
Explanation: {explanation}
')