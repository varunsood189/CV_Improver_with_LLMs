import streamlit as st
import json, warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import datetime, json
import time
def delayed():
    time.sleep(1) 
from langchain.prompts import PromptTemplate

from app_constants import (
    templates,
    PROMPT_IMPROVE_WORK_EXPERIENCE,
    PROMPT_IMPROVE_PROJECT,
    PROMPT_EVALUATE_RESUME,
    PROMPT_IMPROVE_SUMMARY,
)
import retrieval


def create_prompt_template(resume_sections, language="english"):
    """create the promptTemplate.
    Parameters:
       resume_sections (list): List of CV sections from which information will be extracted.
       language (str): the language of the assistant, default="english".
    """

    # Create the Template
    template = f"""For the following resume, output in {language} the following information:\n\n"""

    for key in resume_sections:
        template += key + ": " + templates[key] + "\n\n"

    template += "For any requested information, if it is not found, output 'unknown' or ['unknown'] accordingly.\n\n"
    template += (
        """Format the final output as a json dictionary with the following keys: ("""
    )

    for key in resume_sections:
        template += "" + key + ", "
    template = template[:-2] + ")"  # remove the last ", "

    template += """\n\nResume: {text}"""

    # Create the PromptTemplate
    prompt_template = PromptTemplate.from_template(template)

    return prompt_template


def extract_from_text(text, start_tag, end_tag=None):
    """Use start and end tags to extract a substring from text.
    This helper function is used to parse the response content on the LLM in case 'json.loads' fails.
    """
    start_index = text.find(start_tag)
    if end_tag is None:
        extacted_txt = text[start_index + len(start_tag) :]
    else:
        end_index = text.find(end_tag)
        extacted_txt = text[start_index + len(start_tag) : end_index]

    return extacted_txt


def convert_text_to_list_of_dicts(text, dict_keys):
    """Convert text to a python list of dicts.
    Parameters:
     - text: string containing a list of dicts
     - dict_keys (list): the keys of the dictionary which will be returned.
    Output:
     - list_of_dicts (list): the list of dicts to return.
    """
    list_of_dicts = []

    if text != "":
        text_splitted = text.split("},\n")
        dict_keys.append(None)

        for i in range(len(text_splitted)):
            dict_i = {}

            for j in range(len(dict_keys) - 1):
                key_value = extract_from_text(
                    text_splitted[i], f'"{dict_keys[j]}": ', f'"{dict_keys[j+1]}": '
                )
                key_value = key_value[: key_value.rfind(",\n")].strip()[1:-1]
                dict_i[dict_keys[j]] = key_value

            list_of_dicts.append(dict_i)  # add the dict to the list.

    return list_of_dicts


def get_current_time():
    current_time = (datetime.datetime.now()).strftime("%H:%M:%S")
    return current_time


def invoke_LLM(
    llm,
    documents,
    resume_sections: list,
    info_message="",
    language="english",
):
    """Invoke LLM and get a response.
    Parameters:
     - llm: the LLM to call
     - documents: our Langchain Documents. Will be use to format the prompt_template.
     - resume_sections (list): List of resume sections to be parsed.
     - info_message (str): display an informational message.
     - language (str): Assistant language. Will be use to format the prompt_template.

     Output:
     - response_content (str): the content of the LLM response.
     - response_tokens_count (int): count of response tokens.
    """
    delayed()
    # 1. display the info message
    st.info(f"**{get_current_time()}** \t{info_message}")
    print(f"**{get_current_time()}** \t{info_message}")

    # 2. Create the promptTemplate.
    prompt_template = create_prompt_template(
        resume_sections,
        language=st.session_state.assistant_language,
    )

    # 3. Format promptTemplate with the full documents
    if language is not None:
        prompt = prompt_template.format_prompt(text=documents, language=language).text
    else:
        prompt = prompt_template.format_prompt(text=documents).text
    # 4. Invoke LLM
    delayed()
    response = llm.invoke(prompt)

    response_content = response.content[
        response.content.find("{") : response.content.rfind("}") + 1
    ]
    response_tokens_count = sum(retrieval.tiktoken_tokens([response_content]))

    return response_content, response_tokens_count


def ResponseContent_Parser(
    response_content, list_fields, list_rfind, list_exclude_first_car
):
    """This is a function for parsing any response_content.
    Parameters:
    - response_content (str): the content of the LLM response we are going to parse.
    - list_fields (list): List of dictionary fields returned by this function.
        A field can be a dictionary. The key of the dict will not be parsed.
        Example: [{'Contact_information':['candidate_location','candidate_email','candidate_phone','candidate_social_media']},
                   'CV_summary']
                   We will not parse the content for 'Contact_information'.
    - list_rfind (list): To parse the content of a field, first we will extract the text between this field and the next field.
        Then, extract text using `rfind` Python command, which returns the highest index in the text where the substring is found.
    - list_exclude_first_car (list): Exclusion or not of the first and last characters.

    Output:
      - INFORMATION_dict: dictionary, where fields are the keys and parsed texts are the values.

    """

    list_fields_detailed = (
        []
    )  # list of tupples. tupple = (field,extract info (boolean), parent field)

    for field in list_fields:
        if type(field) is dict:
            list_fields_detailed.append(
                (list(field.keys())[0], False, None)
            )  # We will not extract any value for the text between this tag and the next.
            for val in list(field.values())[0]:
                list_fields_detailed.append((val, True, list(field.keys())[0]))
        else:
            list_fields_detailed.append((field, True, None))

    list_fields_detailed.append((None, False, None))

    # Parse the response_content
    INFORMATION_dict = {}

    for i in range(len(list_fields_detailed) - 1):
        if list_fields_detailed[i][1] is False:  # Extract info = False
            INFORMATION_dict[list_fields_detailed[i][0]] = {}  # Initialize the dict
        if list_fields_detailed[i][1]:
            extracted_value = extract_from_text(
                response_content,
                f'"{list_fields_detailed[i][0]}": ',
                f'"{list_fields_detailed[i+1][0]}":',
            )
            extracted_value = extracted_value[
                : extracted_value.rfind(list_rfind[i])
            ].strip()
            if list_exclude_first_car[i]:
                extracted_value = extracted_value[1:-1].strip()
            if list_fields_detailed[i][2] is None:
                INFORMATION_dict[list_fields_detailed[i][0]] = extracted_value
            else:
                INFORMATION_dict[list_fields_detailed[i][2]][
                    list_fields_detailed[i][0]
                ] = extracted_value

    return INFORMATION_dict


def Extract_contact_information(llm, documents):
    """Extract Contact Information: Name, Title, Location, Email, Phone number and Social media profiles."""
    
    try:
        response_content, response_tokens_count = invoke_LLM(
            llm,
            documents,
            resume_sections=["Contact_information"],
            info_message="Extract and evaluate contact information...",
            language=st.session_state.assistant_language,
        )

        try:
            # Load response_content to json dictionary
            CONTACT_INFORMATION = json.loads(response_content, strict=False)
        except Exception as e:
            print("[ERROR] json.loads returns error:", e)
            print("\n[INFO] Parse response content...\n")

            list_fields = [
                {
                    "Contact_information": [
                        "candidate_name",
                        "candidate_title",
                        "candidate_location",
                        "candidate_email",
                        "candidate_phone",
                        "candidate_social_media",
                        "evaluation_ContactInfo",
                        "score_ContactInfo",
                    ]
                }
            ]
            list_rfind = [",\n", ",\n", ",\n", ",\n", ",\n", ",\n", ",\n", ",\n", "}\n"]
            list_exclude_first_car = [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
            ]
            CONTACT_INFORMATION = ResponseContent_Parser(
                response_content, list_fields, list_rfind, list_exclude_first_car
            )
            # convert score to int
            try:
                CONTACT_INFORMATION["Contact_information"]["score_ContactInfo"] = int(
                    CONTACT_INFORMATION["Contact_information"]["score_ContactInfo"]
                )
            except:
                CONTACT_INFORMATION["Contact_information"]["score_ContactInfo"] = -1

    except Exception as exception:
        print(f"[Error] {exception}")
        CONTACT_INFORMATION = {
            "Contact_information": {
                "candidate_name": "unknown",
                "candidate_title": "unknown",
                "candidate_location": "unknown",
                "candidate_email": "unknown",
                "candidate_phone": "unknown",
                "candidate_social_media": "unknown",
                "evaluation_ContactInfo": "unknown",
                "score_ContactInfo": -1,
            }
        }

    return CONTACT_INFORMATION


def Extract_Evaluate_Summary(llm, documents):
    """Extract, evaluate and strengthen the summary."""

    ######################################
    # 1. Extract the summary
    ######################################
    try:
        response_content, response_tokens_count = invoke_LLM(
            llm,
            documents,
            resume_sections=["CV_summary"],
            info_message="Extract and evaluate the Summary....",
            language=st.session_state.assistant_language,
        )
        try:
            # Load response_content to json dictionary
            SUMMARY_SECTION = json.loads(response_content, strict=False)
        except Exception as e:
            print("[ERROR] json.loads returns error:", e)
            print("\n[INFO] Parse response content...\n")

            list_fields = ["CV_summary"]
            list_rfind = ["}\n"]
            list_exclude_first_car = [True]

            SUMMARY_SECTION = ResponseContent_Parser(
                response_content, list_fields, list_rfind, list_exclude_first_car
            )

    except Exception as exception:
        print(f"[Error] {exception}")
        SUMMARY_SECTION = {"CV_summary": "unknown"}

    ######################################
    # 2. Evaluate the summary
    ######################################

    try:
        prompt_template = PromptTemplate.from_template(PROMPT_IMPROVE_SUMMARY)

        prompt = prompt_template.format_prompt(
            resume=documents,
            language=st.session_state.assistant_language,
            summary=SUMMARY_SECTION["CV_summary"],
        ).text

        # Invoke LLM
        delayed()
        response = llm.invoke(prompt)
        response_content = response.content[
            response.content.find("{") : response.content.rfind("}") + 1
        ]

        try:
            SUMMARY_EVAL = {}
            SUMMARY_EVAL["Summary_evaluation"] = json.loads(
                response_content, strict=False
            )
        except Exception as e:
            print("[ERROR] json.loads returns error:", e)
            print("\n[INFO] Parse response content...\n")

            list_fields = [
                "evaluation_summary",
                "score_summary",
                "CV_summary_enhanced",
            ]
            list_rfind = [",\n", ",\n", "}\n"]
            list_exclude_first_car = [True, False, True]
            SUMMARY_EVAL["Summary_evaluation"] = ResponseContent_Parser(
                response_content, list_fields, list_rfind, list_exclude_first_car
            )
            # convert score to int
            try:
                SUMMARY_EVAL["Summary_evaluation"]["score_summary"] = int(
                    SUMMARY_EVAL["Summary_evaluation"]["score_summary"]
                )
            except:
                SUMMARY_EVAL["Summary_evaluation"]["score_summary"] = -1

    except Exception as e:
        print(e)
        SUMMARY_EVAL = {
            "Summary_evaluation": {
                "evaluation_summary": "unknown",
                "score_summary": -1,
                "CV_summary_enhanced": "unknown",
            }
        }

    SUMMARY_EVAL["CV_summary"] = SUMMARY_SECTION["CV_summary"]

    return SUMMARY_EVAL


def Extract_Education_Language(llm, documents):
    """Extract and evaluate education and language sections."""

    try:
        response_content, response_tokens_count = invoke_LLM(
            llm,
            documents,
            resume_sections=[
                "CV_Education",
                "Education_evaluation",
                "CV_Languages",
                "Languages_evaluation",
            ],
            info_message="Extract and evaluate education and language sections...",
            language=st.session_state.assistant_language,
        )

        try:
            # Load response_content to json dictionary
            Education_Language_sections = json.loads(response_content, strict=False)
        except Exception as e:
            print("[ERROR] json.loads returns error:", e)
            print("\n[INFO] Parse response content...\n")

            list_fields = [
                "CV_Education",
                {"Education_evaluation": ["score_edu", "evaluation_edu"]},
                "CV_Languages",
                {"Languages_evaluation": ["score_language", "evaluation_language"]},
            ]

            list_rfind = [",\n", ",\n", ",\n", ",\n", ",\n", ",\n", ",\n", "\n"]
            list_exclude_first_car = [True, True, False, True, True, True, False, True]

            Education_Language_sections = ResponseContent_Parser(
                response_content, list_fields, list_rfind, list_exclude_first_car
            )

            # Convert scores to int
            try:
                Education_Language_sections["Education_evaluation"]["score_edu"] = (
                    int(
                        Education_Language_sections["Education_evaluation"][
                            "score_edu"
                        ]
                    )
                )
            except:
                Education_Language_sections["Education_evaluation"]["score_edu"] = -1

            try:
                Education_Language_sections["Languages_evaluation"][
                    "score_language"
                ] = int(
                    Education_Language_sections["Languages_evaluation"][
                        "score_language"
                    ]
                )
            except:
                Education_Language_sections["Languages_evaluation"][
                    "score_language"
                ] = -1

            # Split languages and educational texts into a Python list of dict
            languages = Education_Language_sections["CV_Languages"]
            Education_Language_sections["CV_Languages"] = (
                convert_text_to_list_of_dicts(
                    text=languages[
                        languages.find("[") + 1 : languages.rfind("]")
                    ].strip(),
                    dict_keys=["spoken_language", "language_fluency"],
                )
            )
            education = Education_Language_sections["CV_Education"]
            Education_Language_sections["CV_Education"] = (
                convert_text_to_list_of_dicts(
                    text=education[
                        education.find("[") + 1 : education.rfind("]")
                    ].strip(),
                    dict_keys=[
                        "edu_college",
                        "edu_degree",
                        "edu_start_date",
                        "edu_end_date",
                    ],
                )
            )
    except Exception as exception:
        print(exception)
        Education_Language_sections = {
            "CV_Education": [],
            "Education_evaluation": {"score_edu": -1, "evaluation_edu": "unknown"},
            "CV_Languages": [],
            "Languages_evaluation": {
                "score_language": -1,
                "evaluation_language": "unknown",
            },
        }

    return Education_Language_sections


def Extract_Skills_and_Certifications(llm, documents):
    """Extract skills and certifications and evaluate these sections."""
    try:
        response_content, response_tokens_count = invoke_LLM(
            llm,
            documents,
            resume_sections=[
                "candidate_skills",
                "Skills_evaluation",
                "CV_Certifications",
                "Certif_evaluation",
            ],
            info_message="Extract and evaluate the skills and certifications...",
            language=st.session_state.assistant_language,
        )
        try:
            # Load response_content to json dictionary
            SKILLS_and_CERTIF = json.loads(response_content, strict=False)
        except Exception as e:
            print("[ERROR] json.loads returns error:", e)
            print("\n[INFO] Parse response content...\n")

            skills = extract_from_text(
                response_content, '"candidate_skills": ', '"Skills_evaluation":'
            )
            skills = skills.replace("\n  ", "\n").replace("],\n", "").replace("[\n", "")
            score_skills = extract_from_text(
                response_content, '"score_skills": ', '"evaluation_skills":'
            )
            evaluation_skills = extract_from_text(
                response_content, '"evaluation_skills": ', '"CV_Certifications":'
            )

            certif_text = extract_from_text(
                response_content, '"CV_Certifications": ', '"Certif_evaluation":'
            )
            certif_score = extract_from_text(
                response_content, '"score_certif": ', '"evaluation_certif":'
            )
            certif_eval = extract_from_text(
                response_content, '"evaluation_certif": ', None
            )

            # Create the dictionary
            SKILLS_and_CERTIF = {}
            SKILLS_and_CERTIF["candidate_skills"] = [
                skill.strip()[1:-1] for skill in skills.split(",\n")
            ]
            try:
                score_skills_int = int(score_skills[0 : score_skills.rfind(",\n")])
            except:
                score_skills_int = -1
            SKILLS_and_CERTIF["Skills_evaluation"] = {
                "score_skills": score_skills_int,
                "evaluation_skills": evaluation_skills[
                    : evaluation_skills.rfind("}\n")
                ].strip()[1:-1],
            }

            # Convert certificate text to list of dictionaries
            list_certifs = convert_text_to_list_of_dicts(
                text=certif_text[
                    certif_text.find("[") + 1 : certif_text.rfind("]")
                ].strip(),  # .strip()[1:-1]
                dict_keys=[
                    "certif_title",
                    "certif_organization",
                    "certif_date",
                    "certif_expiry_date",
                    "certif_details",
                ],
            )
            SKILLS_and_CERTIF["CV_Certifications"] = list_certifs
            try:
                certif_score_int = int(certif_score[0 : certif_score.rfind(",\n")])
            except:
                certif_score_int = -1
            SKILLS_and_CERTIF["Certif_evaluation"] = {
                "score_certif": certif_score_int,
                "evaluation_certif": certif_eval[: certif_eval.rfind("}\n")].strip()[
                    1:-1
                ],
            }

    except Exception as exception:
        SKILLS_and_CERTIF = {
            "candidate_skills": [],
            "Skills_evaluation": {
                "score_skills": -1,
                "evaluation_skills": "unknown",
            },
            "CV_Certifications": [],
            "Certif_evaluation": {
                "score_certif": -1,
                "evaluation_certif": "unknown",
            },
        }
        print(exception)
    return SKILLS_and_CERTIF


def Extract_PROFESSIONAL_EXPERIENCE(llm, documents):
    """Extract list of work experience and projects."""

    try:
        response_content, response_tokens_count = invoke_LLM(
            llm,
            documents,
            resume_sections=["Work_experience", "CV_Projects"],
            info_message="Extract list of work experience and projects...",
            language=st.session_state.assistant_language,
        )

        try:
            # Load response_content to json dictionary
            PROFESSIONAL_EXPERIENCE = json.loads(response_content, strict=False)
        except Exception as e:
            print("[ERROR] json.loads returns error:", e)
            print("\n[INFO] Parse response content...\n")

            work_experiences = extract_from_text(
                response_content, '"Work_experience": ', '"CV_Projects":'
            )
            projects = extract_from_text(response_content, '"CV_Projects": ', None)

            # Create the dictionary
            PROFESSIONAL_EXPERIENCE = {}
            PROFESSIONAL_EXPERIENCE["Work_experience"] = convert_text_to_list_of_dicts(
                text=work_experiences[
                    work_experiences.find("[") + 1 : work_experiences.rfind("]")
                ].strip()[1:-1],
                dict_keys=[
                    "job_title",
                    "job_company",
                    "job_start_date",
                    "job_end_date",
                ],
            )
            PROFESSIONAL_EXPERIENCE["CV_Projects"] = convert_text_to_list_of_dicts(
                text=projects[projects.find("[") + 1 : projects.rfind("]")].strip()[
                    1:-1
                ],
                dict_keys=[
                    "project_title",
                    "project_start_date",
                    "project_end_date",
                ],
            )
        # Exclude 'unknown' projects and work experiences
        try:
            for work_experience in PROFESSIONAL_EXPERIENCE["Work_experience"]:
                if work_experience["job_title"] == "unknown":
                    PROFESSIONAL_EXPERIENCE["Work_experience"].remove(work_experience)
        except Exception as e:
            print(e)
        try:
            for project in PROFESSIONAL_EXPERIENCE["CV_Projects"]:
                if project["project_title"] == "unknown":
                    PROFESSIONAL_EXPERIENCE["CV_Projects"].remove(project)
        except Exception as e:
            print(e)

    except Exception as exception:
        PROFESSIONAL_EXPERIENCE = {"Work_experience": [], "CV_Projects": []}
        print(exception)

    return PROFESSIONAL_EXPERIENCE


def get_relevant_documents(query, documents):
    """Retreieve most relevant documents from Langchain documents using the CoherRerank retriever."""

    # 1.1. Retrieve documents using the CohereRerank retriever

    retrieved_docs = st.session_state.retriever.get_relevant_documents(query)

    # 1.2. Keep only relevant documents where relevance_score >= (max(relevance_scores) - 0.1)

    relevance_scores = [
        retrieved_docs[j].metadata["relevance_score"]
        for j in range(len(retrieved_docs))
    ]
    max_relevance_score = max(relevance_scores)
    threshold = max_relevance_score - 0.1

    relevant_doc_ids = []

    for j in range(len(retrieved_docs)):

        # keep relevant documents with (relevance_score >= threshold)

        if retrieved_docs[j].metadata["relevance_score"] >= threshold:
            # Append the retrieved document
            relevant_doc_ids.append(retrieved_docs[j].metadata["doc_number"])

    # Append the next document to the most relevant document, as relevant information may be split between two documents.
    relevant_doc_ids.append(min(relevant_doc_ids[0] + 1, len(documents) - 1))

    # Sort document ids
    relevant_doc_ids = sorted(set(relevant_doc_ids))

    # Get the most relevant documents
    relevant_documents = [documents[k] for k in relevant_doc_ids]

    return relevant_documents


def Extract_Job_Responsibilities(llm, documents, PROFESSIONAL_EXPERIENCE):
    """Extract job responsibilities for each job in PROFESSIONAL_EXPERIENCE."""

    st.info(f"**{get_current_time()}** \tExtract work experience responsibilities...")
    print(f"**{get_current_time()}** \tExtract work experience responsibilities...")

    for i in range(len(PROFESSIONAL_EXPERIENCE["Work_experience"])):
        try:
            Work_experience_i = PROFESSIONAL_EXPERIENCE["Work_experience"][i]

            # 1. Extract relevant documents
            query = f"""Extract from the resume delimited by triple backticks \
all the duties and responsibilities of the following work experience: \
(title = '{Work_experience_i['job_title']}'"""
            if str(Work_experience_i["job_company"]) != "unknown":
                query += f" and company = '{Work_experience_i['job_company']}'"
            if str(Work_experience_i["job_start_date"]) != "unknown":
                query += f" and start date = '{Work_experience_i['job_start_date']}'"
            if str(Work_experience_i["job_end_date"]) != "unknown":
                query += f" and end date = '{Work_experience_i['job_end_date']}'"
            query += ")\n"

            try:
                relevant_documents = get_relevant_documents(query, documents)
            except Exception as err:
                st.error(f"get_relevant_documents error: {err}")
                relevant_documents = documents

            # 2. Invoke LLM

            prompt = (
                query
                + f"""Output the duties in a json dictionary with the following keys (_duty_id_,_duty_). \
Use this format: "1":"duty","2":"another duty".
Resume:\n\n ```{relevant_documents}```"""
            )
            delayed()
            response = llm.invoke(prompt)

            # 3. Convert the response content to json dict and update work_experience
            response_content = response.content[
                response.content.find("{") : response.content.rfind("}") + 1
            ]

            try:
                Work_experience_i["work_duties"] = json.loads(
                    response_content, strict=False
                )  # Convert the response content to a json dict
            except Exception as e:
                print("\njson.loads returns error:", e, "\n\n")
                print("\n[INFO] Parse response content...\n")

                Work_experience_i["work_duties"] = {}
                list_duties = (
                    response_content[
                        response_content.find("{") + 1 : response_content.rfind("}")
                    ]
                    .strip()
                    .split(",\n")
                )

                for j in range(len(list_duties)):
                    try:
                        Work_experience_i["work_duties"][f"{j+1}"] = (
                            list_duties[j].split('":')[1].strip()[1:-1]
                        )
                    except:
                        Work_experience_i["work_duties"][f"{j+1}"] = "unknown"

        except Exception as exception:
            Work_experience_i["work_duties"] = {}
            print(exception)

    return PROFESSIONAL_EXPERIENCE


def Extract_Project_Details(llm, documents, PROFESSIONAL_EXPERIENCE):
    """Extract project details for each project in PROFESSIONAL_EXPERIENCE."""

    st.info(f"**{get_current_time()}** \tExtract project details...")
    print(f"**{get_current_time()}** \tExtract project details...")

    for i in range(len(PROFESSIONAL_EXPERIENCE["CV_Projects"])):
        try:
            project_i = PROFESSIONAL_EXPERIENCE["CV_Projects"][i]

            # 1. Extract relevant documents
            query = f"""Extract from the resume (delimited by triple backticks) what is listed about the following project: \
(project title = '{project_i['project_title']}'"""
            if str(project_i["project_start_date"]) != "unknown":
                query += f" and start date = '{project_i['project_start_date']}'"
            if str(project_i["project_end_date"]) != "unknown":
                query += f" and end date = '{project_i['project_end_date']}'"
            query += ")"

            try:
                relevant_documents = get_relevant_documents(query, documents)
            except Exception as err:
                st.error(f"get_relevant_documents error: {err}")
                relevant_documents = documents

            # 2. Invoke LLM

            prompt = (
                query
                + f"""Format the extracted text into a string (with bullet points).
Resume:\n\n ```{relevant_documents}```"""
            )
            delayed()
            response = llm.invoke(prompt)

            response_content = response.content
            project_i["project_description"] = response_content

        except Exception as exception:
            project_i["project_description"] = "unknown"
            print(exception)

    return PROFESSIONAL_EXPERIENCE


###############################################################################
#           Improve Work Experience and Project texts
###############################################################################


def improve_text_quality(PROMPT, text_to_imporve, llm, language):
    """Invoke LLM to improve the text quality."""
    query = PROMPT.format(text=text_to_imporve, language=language)
    delayed()
    response = llm.invoke(query)
    return response


def improve_work_experience(WORK_EXPERIENCE: list, llm):
    """Improve each bullet point in the work experience responsibilities."""

    message = f"**{get_current_time()}** \tImprove the quality of the work experience section..."
    st.info(message)
    print(message)

    # Call LLM for any work experience to get a better and stronger text.
    for i in range(len(WORK_EXPERIENCE)):
        try:
            WORK_EXPERIENCE_i = WORK_EXPERIENCE[i]

            # 1. Convert the responsibilities from dict to string

            text_duties = ""
            for duty in list(WORK_EXPERIENCE_i["work_duties"].values()):
                text_duties += "- " + duty
            # 2. Call LLM

            response = improve_text_quality(
                PROMPT_IMPROVE_WORK_EXPERIENCE,
                text_duties,
                llm,
                st.session_state.assistant_language,
            )
            response_content = response.content

            # 3. Convert response content to json dict with keys:
            # ('Score_WorkExperience','Comments_WorkExperience','Improvement_WorkExperience')

            response_content = response_content[
                response_content.find("{") : response_content.rfind("}") + 1
            ]

            try:
                list_fields = [
                    "Score_WorkExperience",
                    "Comments_WorkExperience",
                    "Improvement_WorkExperience",
                ]
                list_rfind = [",\n", ",\n", "\n"]
                list_exclude_first_car = [False, True, True]
                response_content_dict = ResponseContent_Parser(
                    response_content, list_fields, list_rfind, list_exclude_first_car
                )
                try:
                    response_content_dict["Score_WorkExperience"] = int(
                        response_content_dict["Score_WorkExperience"]
                    )
                except:
                    response_content_dict["Score_WorkExperience"] = -1

            except Exception as e:
                response_content_dict = {
                    "Score_WorkExperience": -1,
                    "Comments_WorkExperience": "",
                    "Improvement_WorkExperience": "",
                }
                print(e)
                st.error(e)

            # 4. update PROFESSIONAL_EXPERIENCE: Add the new keys (overall_quality, comments, Improvement.)

            WORK_EXPERIENCE_i["Score_WorkExperience"] = response_content_dict[
                "Score_WorkExperience"
            ]
            WORK_EXPERIENCE_i["Comments_WorkExperience"] = response_content_dict[
                "Comments_WorkExperience"
            ]
            WORK_EXPERIENCE_i["Improvement_WorkExperience"] = response_content_dict[
                "Improvement_WorkExperience"
            ]

        except Exception as exception:
            st.error(exception)
            print(exception)
            WORK_EXPERIENCE_i["Score_WorkExperience"] = -1
            WORK_EXPERIENCE_i["Comments_WorkExperience"] = ""
            WORK_EXPERIENCE_i["Improvement_WorkExperience"] = ""

    return WORK_EXPERIENCE


def improve_projects(PROJECTS: list, llm):
    """Improve project text with LLM."""

    st.info(f"**{get_current_time()}** \tImprove the quality of the project section...")
    print(f"**{get_current_time()}** \tImprove the quality of the project section...")

    for i in range(len(PROJECTS)):
        try:
            PROJECT_i = PROJECTS[i]  # the ith project.

            # 1. LLM call to improve the text quality of each duty
            response = improve_text_quality(
                PROMPT_IMPROVE_PROJECT,
                PROJECT_i["project_title"] + "\n" + PROJECT_i["project_description"],
                llm,
                st.session_state.assistant_language,
            )
            response_content = response.content

            # 2. Convert response content to json dict with keys:
            # ('Score_project','Comments_project','Improvement_project')

            response_content = response_content[
                response_content.find("{") : response_content.rfind("}") + 1
            ]

            try:
                list_fields = [
                    "Score_project",
                    "Comments_project",
                    "Improvement_project",
                ]
                list_rfind = [",\n", ",\n", "\n"]
                list_exclude_first_car = [False, True, True]

                response_content_dict = ResponseContent_Parser(
                    response_content, list_fields, list_rfind, list_exclude_first_car
                )
                try:
                    response_content_dict["Score_project"] = int(
                        response_content_dict["Score_project"]
                    )
                except:
                    response_content_dict["Score_project"] = -1

            except Exception as e:
                response_content_dict = {
                    "Score_project": -1,
                    "Comments_project": "",
                    "Improvement_project": "",
                }
                print(e)

            # 3. Update PROJECTS
            PROJECT_i["Score_project"] = response_content_dict["Score_project"]
            PROJECT_i["Comments_project"] = response_content_dict["Comments_project"]
            PROJECT_i["Improvement_project"] = response_content_dict[
                "Improvement_project"
            ]

        except Exception as exception:
            print(exception)

            PROJECT_i["Score_project"] = -1
            PROJECT_i["Comments_project"] = ""
            PROJECT_i["Improvement_project"] = ""

    return PROJECTS


###############################################################################
#                           Evaluate the Resume
###############################################################################


def Evaluate_the_Resume(llm, documents):
    try:
        st.info(
            f"**{get_current_time()}** \tEvaluate, outline and analyse \
the resume's top 3 strengths and top 3 weaknesses..."
        )
        print(
            f"**{get_current_time()}** \tEvaluate, outline and analyse \
the resume's top 3 strengths and top 3 weaknesses..."
        )

        prompt_template = PromptTemplate.from_template(PROMPT_EVALUATE_RESUME)
        prompt = prompt_template.format_prompt(
            text=documents, language=st.session_state.assistant_language
        ).text

        # Invoke LLM
        delayed()
        response = llm.invoke(prompt)
        response_content = response.content[
            response.content.find("{") : response.content.rfind("}") + 1
        ]
        try:
            RESUME_EVALUATION = json.loads(response_content)
        except Exception as e:
            print("[ERROR] json.loads returns error:", e)
            print("\n[INFO] Parse response content...\n")

            list_fields = ["resume_cv_overview", "top_3_strengths", "top_3_weaknesses"]
            list_rfind = [",\n", ",\n", "\n"]
            list_exclude_first_car = [True, True, True]
            RESUME_EVALUATION = ResponseContent_Parser(
                response_content, list_fields, list_rfind, list_exclude_first_car
            )

    except Exception as error:
        RESUME_EVALUATION = {
            "resume_cv_overview": "unknown",
            "top_3_strengths": "unknown",
            "top_3_weaknesses": "unknown",
        }
        print(f"An error occured: {error}")

    return RESUME_EVALUATION


def get_section_scores(SCANNED_RESUME):
    """Output in a dictionary the scores of all the sections of the resume (summary, skills...)"""
    dict_scores = {}
    # Summary, Skills, EDUCATION
    dict_scores["ContactInfo"] = max(
        -1, SCANNED_RESUME["Contact_information"]["score_ContactInfo"]
    )
    dict_scores["summary"] = max(
        -1, SCANNED_RESUME["Summary_evaluation"]["score_summary"]
    )
    dict_scores["skills"] = max(
        -1, SCANNED_RESUME["Skills_evaluation"]["score_skills"]
    )
    dict_scores["education"] = max(
        -1, SCANNED_RESUME["Education_evaluation"]["score_edu"]
    )
    dict_scores["language"] = max(
        -1, SCANNED_RESUME["Languages_evaluation"]["score_language"]
    )
    dict_scores["certfication"] = max(
        -1, SCANNED_RESUME["Certif_evaluation"]["score_certif"]
    )
    # Work_experience: The score is the average of the scores of all the work experiences.
    scores = []
    for work_experience in SCANNED_RESUME["Work_experience"]:
        score = work_experience["Score_WorkExperience"]
        if score > -1:
            scores.append(score)
    try:
        dict_scores["work_experience"] = int(sum(scores) / len(scores))
    except:
        dict_scores["work_experience"] = 0

    # Projects: The score is the average of the scores of all projects.
    scores = []
    for project in SCANNED_RESUME["CV_Projects"]:
        score = project["Score_project"]
        if score > -1:
            scores.append(score)
    try:
        dict_scores["projects"] = int(sum(scores) / len(scores))
    except:
        dict_scores["projects"] = 0

    return dict_scores


###############################################################################
#                           Put it all together
###############################################################################


def resume_analyzer_main(llm, llm_creative, documents):
    """Put it all together: Extract, evaluate and improve all resume sections.
    Save the final results in a dictionary.
    """
    # 1. Extract Contact information: Name, Title, Location, Email,...
    CONTACT_INFORMATION = Extract_contact_information(llm, documents)

    # 2. Extract, evaluate and improve the Summary
    Summary_SECTION = Extract_Evaluate_Summary(llm, documents)

    # 3. Extract and evaluate education and language sections.
    Education_Language_sections = Extract_Education_Language(llm, documents)

    # 4. Extract and evaluate the SKILLS.
    SKILLS_and_CERTIF = Extract_Skills_and_Certifications(llm, documents)

    # 5. Extract Work Experience and Projects.
    PROFESSIONAL_EXPERIENCE = Extract_PROFESSIONAL_EXPERIENCE(llm, documents)

    # 6. EXTRACT WORK EXPERIENCE RESPONSIBILITIES.
    PROFESSIONAL_EXPERIENCE = Extract_Job_Responsibilities(
        llm, documents, PROFESSIONAL_EXPERIENCE
    )

    # 7. EXTRACT PROJECT DETAILS.
    PROFESSIONAL_EXPERIENCE = Extract_Project_Details(
        llm, documents, PROFESSIONAL_EXPERIENCE
    )

    # 8. Improve the quality of the work experience section.
    PROFESSIONAL_EXPERIENCE["Work_experience"] = improve_work_experience(
        WORK_EXPERIENCE=PROFESSIONAL_EXPERIENCE["Work_experience"], llm=llm_creative
    )

    # 9. Improve the quality of the project section.
    PROFESSIONAL_EXPERIENCE["CV_Projects"] = improve_projects(
        PROJECTS=PROFESSIONAL_EXPERIENCE["CV_Projects"], llm=llm_creative
    )

    # 10. Evaluate the Resume
    RESUME_EVALUATION = Evaluate_the_Resume(llm_creative, documents)

    # 11. Put it all together: create the SCANNED_RESUME dictionary
    SCANNED_RESUME = {}
    for dictionary in [
        CONTACT_INFORMATION,
        Summary_SECTION,
        Education_Language_sections,
        SKILLS_and_CERTIF,
        PROFESSIONAL_EXPERIENCE,
        RESUME_EVALUATION,
    ]:
        SCANNED_RESUME.update(dictionary)
    
    st.info(f"**{get_current_time()}** \tcreate the SCANNED_RESUME dictionary")
    print(f"**{get_current_time()}** \tcreate the SCANNED_RESUME dictionary")
    
    # 12. Save the Scanned resume
    try:
        now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
        file_name = "results_" + now
        with open(f"./data/{file_name}.json", "w") as fp:
            json.dump(SCANNED_RESUME, fp)
    except:
        pass
    st.info(f"**{get_current_time()}** \tSave the Scanned resume")
    print(f"**{get_current_time()}** \tSave the Scanned resume")
    
    return SCANNED_RESUME
