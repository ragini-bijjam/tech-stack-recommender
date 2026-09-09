import streamlit as st
from recommender import load_data, recommend, normalize


st.set_page_config(
    page_title="Tech Stack Recommender",
    layout="centered"
)


st.title("Tech Stack Recommender")

st.write(
    "Discover career paths that best match your "
    "technical skills using TF-IDF and Cosine Similarity."
)


roles = load_data("raw_skills.csv")


all_skills = set()

for role in roles:
    all_skills.update(role["skills"])


st.subheader("Available Skills")

st.write(
    ", ".join(sorted(all_skills))
)


st.subheader("Enter Your Skills")

user_input = st.text_input(
    "Enter at least 3 skills separated by commas",
    placeholder="Python, Cloud, Automation"
)


if st.button("Recommend Career Paths"):

    if not user_input.strip():

        st.warning(
            "Please enter your skills."
        )

    else:

        user_skills = [
            normalize(x)
            for x in user_input.split(",")
            if x.strip()
        ]

        if len(user_skills) < 3:

            st.warning(
                "Please enter at least 3 skills."
            )

        else:

            valid_skills = []
            unknown_skills = []

            for skill in user_skills:

                if skill in all_skills:
                    valid_skills.append(skill)

                else:
                    unknown_skills.append(skill)


            if unknown_skills:

                st.warning(
                    "Skills not found in dataset: "
                    + ", ".join(unknown_skills)
                )


            if len(valid_skills) == 0:

                st.error(
                    "None of the entered skills are "
                    "available in our dataset."
                )

            else:

                st.success(
                    f"{len(valid_skills)} skill(s) recognized."
                )

                results = recommend(
                    valid_skills,
                    roles,
                    3
                )


                st.subheader(
                    "Top 3 Recommended Career Paths"
                )


                for i, result in enumerate(
                    results,
                    1
                ):

                    st.markdown(
                        f"### {i}. {result['role']}"
                    )


                    percentage = (
                        result["score"] * 100
                    )


                    st.write(
                        f"Match Score: "
                        f"{percentage:.2f}%"
                    )


                    st.progress(
                        min(
                            max(
                                result["score"],
                                0.0
                            ),
                            1.0
                        )
                    )


                    if result["matched_skills"]:

                        st.write(
                            "Matched Skills: "
                            + ", ".join(
                                result["matched_skills"]
                            )
                        )


                    if result["missing_skills"]:

                        st.write(
                            "Skills to Learn: "
                            + ", ".join(
                                result["missing_skills"]
                            )
                        )


                    st.divider()


st.caption(
    "AI Recommendation Logic | "
    "TF-IDF + Cosine Similarity"
)