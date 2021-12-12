# importing `search` should work after pip installing from repo
  # python pip install git+https://github.com/sanjibg01/CS410-project-team-trailblazers.git

from typing import List
import pandas as pd
from search import SearchEngine, TfidfCosineSearch

def make_scored_docs(query, documents, titles):
    scored_docs = TfidfCosineSearch(query, documents).make_document_scores_df()
    scored_docs["title"] = scored_docs["document_id"].map(dict(enumerate(titles)))
    scored_docs["document_preview"] = scored_docs["document"].str.slice(0, 50)
    scored_docs = scored_docs[["document_id", "title", "score", "document", "document_preview"]]
    return scored_docs

se = SearchEngine(course_transcript_filename='transcripts_text-retrieval_txt.json')

# list of document strings; we will map the index/order to a document title/label later
arxiv_documents: List[str] = list(se.arxiv_data.values())
lecture_documents: List[str] = list(se.lecture_data.values())

arxiv_titles = se.arxiv_data.keys()
lecture_titles = se.lecture_data.keys()

query = "natural language processing"

arxiv_df = make_scored_docs(query=query, 
                            documents=arxiv_documents, 
                            titles=arxiv_titles)

lecture_df = make_scored_docs(query=query, 
                              documents=lecture_documents, 
                              titles=lecture_titles)

# leaving out `document` column for formatting (over-wraps table)
print(
    (arxiv_df.loc[:, ['document_id', 'title', 'document_preview']]
             .head(5)
             .to_markdown()
    )
    )

print('\n')

print(
    (lecture_df.loc[:, ['document_id', 'title', 'document_preview']]
                .head(5)
                .to_markdown()
    )
    )

# |    |   document_id | title                                                                                | document_preview                                 |
# |---:|--------------:|:-------------------------------------------------------------------------------------|:-------------------------------------------------|
# |  0 |          3369 | 0704.3370 - Natural boundary of Dirichlet series                                     | We prove some conditions on the existence of nat |
# |  1 |          5300 | 0705.1298 - Mykyta the Fox and networks of language                                  | The results of quantitative analysis of word dis |
# |  2 |          1318 | 0704.1319 - Using conceptual metaphor and functional grammar to explore how language | This paper introduces a theory about the role of |
# |    |               |   used in physics affects student learning                                           |                                                  |
# |  3 |          3664 | 0704.3665 - On the Development of Text Input Method - Lessons Learned                | Intelligent Input Methods (IM) are essential for |
# |  4 |           690 | 0704.0691 - Birth, survival and death of languages by Monte Carlo simulation         | Simulations of physicists for the competition be |


# |    |   document_id | title                                                          | document_preview                                   |
# |---:|--------------:|:---------------------------------------------------------------|:---------------------------------------------------|
# |  0 |             2 | 01_lesson-1-1-natural-language-content-analysis.en.txt         | [SOUND] >> This lecture is about Natural Language  |
# |  1 |            21 | 02_lesson-4-2-statistical-language-model.en.txt                | [SOUND] This lecture is about the statistical lang |
# |  2 |            44 | 10_lesson-6-10-course-summary.en.txt                           | [NOISE] This lecture is a summary of this course.  |
# |  3 |            29 | 03_lesson-5-3-feedback-in-text-retrieval-feedback-in-lm.en.txt | [SOUND] This lecture is about the feedback in the  |
# |  4 |            25 | 06_lesson-4-6-smoothing-methods-part-1.en.txt                  | [SOUND] This lecture is about the specific smoothi |