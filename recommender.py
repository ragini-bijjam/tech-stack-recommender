import csv
import math


skill_aliases = {
    "cloud": "cloud computing",
    "ml": "machine learning",
    "k8s": "kubernetes",
    "docker container": "docker",
    "amazon web services": "aws",
    "artificial intelligence": "machine learning"
}


def normalize(text):
    text = text.strip().lower()

    if text in skill_aliases:
        return skill_aliases[text]

    return text


def load_data(filename):
    roles = []

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            skills = [normalize(x) for x in row["skills"].split(",")]

            roles.append({
                "role": row["role"],
                "skills": skills
            })

    return roles


def build_vocabulary(roles, user_skills):
    vocabulary = set(user_skills)

    for role in roles:
        vocabulary.update(role["skills"])

    return sorted(vocabulary)


def tf(skill_list, skill):
    if len(skill_list) == 0:
        return 0

    return skill_list.count(skill) / len(skill_list)


def idf(skill, documents):
    total_documents = len(documents)

    documents_with_skill = 0

    for document in documents:
        if skill in document:
            documents_with_skill += 1

    if documents_with_skill == 0:
        return 0

    return math.log(total_documents / documents_with_skill)


def tf_idf_vector(skill_list, vocabulary, documents):
    vector = []

    for skill in vocabulary:
        value = tf(skill_list, skill) * idf(skill, documents)
        vector.append(value)

    return vector


def cosine_similarity(a, b):
    dot_product = 0
    magnitude_a = 0
    magnitude_b = 0

    for i in range(len(a)):
        dot_product += a[i] * b[i]
        magnitude_a += a[i] * a[i]
        magnitude_b += b[i] * b[i]

    magnitude_a = math.sqrt(magnitude_a)
    magnitude_b = math.sqrt(magnitude_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


def recommend(user_skills, roles, top_n=3):
    user_skills = [normalize(x) for x in user_skills]

    documents = [role["skills"] for role in roles]

    vocabulary = build_vocabulary(roles, user_skills)

    user_vector = tf_idf_vector(
        user_skills,
        vocabulary,
        documents
    )

    results = []

    for role in roles:

        role_vector = tf_idf_vector(
            role["skills"],
            vocabulary,
            documents
        )

        score = cosine_similarity(
            user_vector,
            role_vector
        )

        matched_skills = []

        for skill in user_skills:
            if skill in role["skills"]:
                matched_skills.append(skill)

        missing_skills = []

        for skill in role["skills"]:
            if skill not in user_skills:
                missing_skills.append(skill)

        results.append({
            "role": role["role"],
            "score": score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_n]


def main():

    print("TECH STACK RECOMMENDER")
    print("----------------------")

    user_input = input(
        "Enter at least 3 skills separated by commas: "
    )

    user_skills = [
        normalize(x)
        for x in user_input.split(",")
        if x.strip()
    ]

    if len(user_skills) < 3:

        print(
            "Please enter at least 3 skills."
        )

        return

    roles = load_data(
        "raw_skills.csv"
    )

    all_skills = set()

    for role in roles:
        all_skills.update(
            role["skills"]
        )

    valid_skills = []

    unknown_skills = []

    for skill in user_skills:

        if skill in all_skills:
            valid_skills.append(skill)

        else:
            unknown_skills.append(skill)

    if unknown_skills:

        print(
            "Skills not found in dataset:",
            ", ".join(unknown_skills)
        )

    if len(valid_skills) == 0:

        print(
            "None of the entered skills are "
            "available in our dataset."
        )

        return

    results = recommend(
        valid_skills,
        roles,
        3
    )

    print(
        "\nTop 3 Recommended Career Paths"
    )

    print(
        "--------------------------------"
    )

    for i, result in enumerate(
        results,
        1
    ):

        print(
            f"{i}. {result['role']}"
        )

        print(
            f"   Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            "   Matched skills: "
            + ", ".join(
                result["matched_skills"]
            )
        )

        print(
            "   Skills to learn: "
            + ", ".join(
                result["missing_skills"]
            )
        )

        print()


if __name__ == "__main__":
    main()