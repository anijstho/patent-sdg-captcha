<p align="center">
  <a href="https://captcha.thomasanijs.com/">
    <img src="/captsdga.png" alt="CAPTSDGA" width="288" height="288">
  </a>
</p>
<h1 align="center">CAPTSDGA</h1>
<p align="center">
  Completely Automated Public Turing Test to tell Sustainable Development Goals Apart
</p>

## Table of Contents

1. [Project Motivation](#1-project-motivation)
2. [How to interact with the code](#2-how-to-interact-with-the-code)
	1. [SDG Classification Model](#21-sdg-classification-model)
	2. [CAPTCHA Tests](#22-captcha-tests)
3. [Ensemble Approach](#3-ensemble-approach)
	1. [Query Models](#31-query-models)
	2. [sdgBERT](#32-sdgbert)
	3. [Mistral LLM](#33-mistral-llm)
4. [Model Training](#4-model-training)
	1. [Training Data](#41-training-data)
	2. [CAPTCHA Responses](#42-captcha-responses)
	3. [Model Performance](#43-model-performance)
5. [Conclusions](#5-conclusions)
6. [Acknowledgements](#6-acknowledgements)

## 1. Project Motivation

The goal of this project is to develop a system that automates the classification of patent data based on its potential contribution to the [UN Sustainable Development Goals (SDGs)](https://sdgs.un.org/goals).\
This project is performed as a submission for the [EPO CodeFest Spring 2025 on classifying patent data for sustainable development](https://www.epo.org/en/news-events/in-focus/codefest/codefest-spring-2025-classifying-patent-data-sustainable-development). 

To enhance the accessibility and strategic use of patent information and to benefit users, special attention is paid for the solution to be:
- ***Creative and innovative*** by building an original solution incorporating challenge-response Turing tests (CAPTCHAs) in the context of sustainable development goals for the first time

- ***Functional and usable*** through interacting via simple [classification](https://codefest.thomasanijs.com) and [CAPTCHA](https://captcha.thomasanijs.com) web apps, providing an ever expanding labeled dataset for analyses and supervised learning
- ***Well-implemented*** using iteratively learning ensemble models demonstrated to exceed the performance of all tested existing SDG classification models in patent labeling
- ***Impactful on sustainable innovation*** by not only serving patent enthusiasts, but appealing to a broader audience through CAPTCHAs, inspiring more people to contribute to sustainable development

## 2. How to interact with the code

The solution to the SDG classification problem for patent data consists of two distinct parts. Users interested in identifying relevant SDGs from (potential) patents can do so by interacting with the current classification model via a web interface as explained in [Section 2.1](#21-sdg-classification-model).
People who want to challenge themselves to identify innovations for sustainable development from existing patents can contribute to training of the classification model by playing an online game based on CAPTCHA tests, elaborated in [Section 2.2](#22-captcha-tests).

The source code of both web applications is shared in the [Source Code](/Source%20Code) folder.

### 2.1 SDG Classification Model

You can use the latest SDG classification model by providing a patent description on the classification web app (_currently offline_). The model will then return the final SDGs it has identified from the text input based on the nine underlying SDG classification model outcomes following the [Ensemble Approach](#3-ensemble-approach). The web application will give the most reliable outcome when provided an English patent abstract of around 50 to 200 words, since the model is specifically trained on this type of patent data (see [Training Data](#41-training-data)).

The text is being analysed in ten steps, with the process being indicated live in the terminal output. The steps consist of processing eight SDG classification submodels, applying the [ensemble model](#3-ensemble-approach), and extracting the [most predictive](#43-model-performance) model outcomes, respectively. Due to limited capabilities of the current web server, the classification process can take a few minutes depending on the length of the text input.

### 2.2 CAPTCHA Tests

A key feature of the implemented approach is that the classification model iteratively evolves by using new CAPTCHA answers as additional training input. Everyone can contribute to patent labeling by classifying existing patent abstracts through dedicated online CAPTCHA tests (_currently offline_).

The CAPTCHA tests try to identify whether you are a human or robot based on consensus of your answer with the outcome of the 8 classification submodels. You are requested to select an SDG belonging to the displayed prompt, which describes the practical use of a patent from the [training data](#41-training-data) as extracted by the [Mistral LLM](#33-mistral-llm).

To keep track of the options, only 9 out of the 16 considered SDGs are displayed per prompt. The options are selected based on their relevance to the patent based on the following criteria:

- **Consensus**: All SDGs identified by two or more submodels are shown
- **Probability**: To end up with nine SDGs, goals are added or removed based on their probability according to the ``sdgBERT`` model, leaving out the least probable SDGs

Since these CAPTCHAs do not grant access to a targeted web page like typical CAPTCHAs, a streak counter is introduced to encourage the player to continue. The goal is to reach a high score by getting as much answers in a row correct. The submitted SDG is rated by its consensus with the 8 submodels in the following manner:

- **Correct**: 2 or more submodels agree with the selected SDG; the streak counter will increase by 1
- **Not sure**: The selected SDG is found in only one submodel; the streak is retained
- **Incorrect**: None of the submodels agree with the selected SDG; the streak is being reset to zero

In case you think a patent is not related to any of the displayed SDGs, 'None' can be submitted when no SDG is selected. In this case, the maximum number of models positive on any SDG is considered in the following way (effect of the verdict on the streak remains the same):

- **Correct**: Any SDG is detected by a maximum of only 2 submodels
- **Not sure**: 3 submodels have identified the same SDG(s)
- **Incorrect**: 4 or more submodels are positive on at least one SDG

How the responses are being taken into account in labeling the training data is explained in [Section 4.2](#42-captcha-responses).

## 3. Ensemble Approach

The classification system classifies a patent text based on 16 out of the 17 SDGs. SDG 17 ('Partnerships for the goals') has been left out as this goal does not reflect any particular research area or application for patentable technology. Each patent can include multiple SDGs.

To create a classification model specifically for labeling SDGs in patent data, an ensemble model has been set up largely following the approach of [Wulff et al.](https://doi.org/10.1007/s11625-024-01516-3) combining multiple SDG classification models. The authors demonstrated that the ensemble approach was able to achieve higher specificity and sensitivity compared to all individual SDG labeling systems. The ability to tailor the ensemble model to a given use case, in this context the labeling of patent data, makes it a very suitable approach for the current classification problem.

A few changes have been made to the ensemble approach in comparison to the method of [Wulff et al.](https://doi.org/10.1007/s11625-024-01516-3). The new ensemble method is based on 8 instead of 7 existing classification models, to improve performance; next to the same 6 query models, the ``sdgBERT`` classifier and ``Mistral`` LLM have been added, while the ``OSDG`` system has been removed due to inaccessibility. A short description of all eight classification submodels is given below.\
The patent ensemble models have been trained using the extreme gradient boosting algorithm of ``XGBoost`` instead of random forests; performance of both algorithms was reported to be similar, but an important advantage of the ``XGBoost`` library in this case is that it supports incremental learning, meaning the ensemble model can be trained on only new data becoming available over time from CAPTCHAs while retaining its learnings from older data.

### 3.1 Query Models

Six query models have been used in classification:
- ``auckland``: [University of Auckland](https://doi.org/10.21203/rs.3.rs-2544385/v2)
- ``aurora v5.0``: [Aurora Universities Network](https://doi.org/10.5281/zenodo.3817445)
- ``elsevier v1``: [Elsevier SDGs Mapping](https://doi.org/10.17632/87txkw7khs.1)
- ``sdgo``: Open Source SDG
- ``sdsn``: [Sustainable Development Solutions Network](https://ap-unsdsn.org/regional-initiatives/universities-sdgs/)
- ``siris v1.2``: [SIRIS Academic](https://zenodo.org/record/3567769)

These models determine the presence of an SDG by relying on Lucene-style queries to detect keywords and apply syntax-based Boolean operations, wildcards and proximity searches. All six models have been introduced as standalone systems to identify the SDGs from academic research papers, but suffer from severe shortcomings when applying them seperately to patent abstracts ([Model Performance](#43-model-performance)). Since [Wulff et al.](https://doi.org/10.1007/s11625-024-01516-3) demonstrated that ensemble modeling was able to overcome limitations related to different text sources, and these query models made up 6 of the 7 implemented submodels, the query strings were adopted directly from the R package of the ensemble approach. To interpret the queries in the Python backend of the patent classification app ([``sdg_matrix.py``](/Source%20Code/Classification%20App)), the syntax is parsed by ``Lark`` and the searches are performed using ``Elasticsearch``.

### 3.2 sdgBERT

To introduce a more advanced model compared to the relatively straightforward query models, a BERT language model is introduced. In contrast to query models, BERT can structurally take the context of a text into account by transforming text into a sequency of vectors, and is therefore an important tool for natural language processing and LLMs. The transformer-based SDG classification model ``sdgBERT`` (109M parameters) from [Deakin University](https://huggingface.co/sadickam/sdg-classification-bert) is implemented as the BERT model. Although multiple pre-trained BERT-based SDG classification models are available, this model was chosen because it has a reported accuracy of 0.90, it was made in an academic context, and it is the most-used SDG classification model from the popular ML model library Hugging Face.

An important limitation of this model, however, is that it returns the relative probabilities of the first 16 SDGs in relation to each other instead of the probability of their presence in the text, meaning that the sum of all 16 probabilities is always one despite the presence and number of SDGs in the text. Therefore it is not possible to reconstruct the actual (number of) SDGs it detects in the text. To take the outcome along in the ensemble approach, the probabilities are treated very conservatively: an SDG is only considered to be present with a probability of 0.95 or higher, meaning only one SDG can be detected per patent, implying a low sensitivity and high specificity. This limitation is the main reason for the low ``sdgBERT`` F1 scores reported under [Model Performance](#43-model-performance).\
An advantage of the relative probabilities of SDGs to each other is that the ``sdgBERT`` outcome is useful for determining the most relevant SDGs to display in the [CAPTCHA tests](#22-captcha-tests).

### 3.3 Mistral LLM

With the popularity and rapid advancements of large language models (LLMs) such as ChatGPT, Gemini and Mistral, an obvious approach is to ask an LLM to perform the classification task. To test the performance of a modern LLM, patents are classified using [``Mistral Small 3.1``](https://mistral.ai/news/mistral-small-3-1). This LLM is chosen because it is a European (French) lightweight open-source model which performs excellent for its weight class. Since the energy footprint of large LLMs is increasingly becoming a concern for sustainable development, the energy savings over time by using such a smaller LLM could be considerable in case of relatively simple repetitive tasks like text classification. Even though this model is considered lightweight for an LLM, it still consists of 24B parameters compared to 109M parameters for the BERT language model. Since the web server of the classification app does not have the performance to run the LLM directly, the requests are submitted to Mistral's La Plateforme using a free API key. For local deployment of the web environment, a personal API key can be requested [here](https://console.mistral.ai/).

To label the SDGs using the LLM, the patent is processed by two subsequent system prompts. The first prompt is used to summarize the patent use in one sentence. The patent abstract is used as user prompt, while the system prompt reads:

``Describe the application of the invention described in the patent abstract, including any environmental, social and economic aspects. The summary should be in a single sentence, and should not exceed 50 words. Do not mention that the summary concerns an invention or patent, and avoid technical details and demonstrative pronouns. The summary should be clear and concise, and should be suitable for a general audience.``

The invention is first summarized in one sentence to avoid differences in subsequently identied SDG labels caused by variations in length and structure of the original patent abstracts. For the training data, the resulting sentences are also used as the patent descriptions for which a corresponding SDG should be selected in the CAPTCHAs.\
The single sentence is used as user prompt within the LLM classification task. This task is described in the system prompt as follows:

``Report a maximum of six SDGs that are most relevant to the text. See below for SDG descriptions. Use Python list format and report only the SDG number(s). If you detect no SDGs, report [0].``

``SDG 1: End poverty in all its forms everywhere``, followed by the other SDG descriptions as used on the [UN SDGs frontpage](https://sdgs.un.org)

## 4. Model Training
### 4.1 Training Data
The patent data used to train the SDG classification model is derived from the EP Full Text Library (EPAB). The abstract fields of all patents in the test environment with an indexed English abstract (3113 publications) have been evaluated by the nine SDG classification models and are taken along for labeling within the (random) rotation of the CAPTCHA app. Patent abstracts have been considered since the abstract field is where typically the practical use of the patent is described.

The current model implemented in the web interface is based on 2094 manual ratings; to ensure an even distribution over the 16 SDGs in the initial dataset, every SDG has manual labels for at least 100 patents in which the SDG has been detected by at least 1 submodel. All training data is saved into a ``pandas`` DataFrame in [``training_data.pkl``](/Supporting%20Materials%20and%20Tools/Training%20Data). The frequency at which each SDG has been detected by the 8 submodels is displayed in the bar graph below (generated by [``plot_sdg_model_frequencies.py``](/Supporting%20Materials%20and%20Tools/Training%20Data)):

![SDG-Model Frequence Plot](/Supporting%20Materials%20and%20Tools/Training%20Data/sdg_model_frequencies.png)

### 4.2 CAPTCHA Responses
All CAPTCHA responses in the web application are tracked in the SQLite database ``captcha_responses.db`` on the server, and the statistics of the manual labels in the training data are updated by processing the answers using [``update_testenv_from_responses.py``](/Supporting%20Materials%20and%20Tools/CAPTCHA%20Responses). For every response in which an SDG is selected, a positive vote for that SDG is added to the patent in case the answer is not considered 'incorrect' [according to the test](#22-captcha-tests) (otherwise the answer is ignored to maintain data quality). In case no SDG is selected, all nine displayed SDGs are counted as negative for the related patent. The relative weights assigned to the binary labels in the gradient boosting algorithm are defined as the product of the consensus and number of responses, favouring the most reliable answers in training the ensemble model.

### 4.3 Model Performance
Performance of the eight classification submodels and the tuned ensemble models with and without the Mistral model has been evaluated by the F1 score following stratified 5-fold cross-validation ([``performance_analysis.py``](/Supporting%20Materials%20and%20Tools/Model%20Performance)) to get the general predictive performance, balancing precision and sensitivity, when applying it to new data. The heatmap of F1 scores by model and SDG is shown below. The rightmost column shows the F1 scores of the best-performing model per SDG, and reflects the performance of the SDG classification results as returned by the [web app](https://codefest.thomasanijs/com).

![F1 Heatmap](/Supporting%20Materials%20and%20Tools/Model%20Performance/f1_heatmap.png)

## 5. Conclusions
Important conclusions which can be drawn from the current SDG classification solution:
- The implemented ('best') model has **good overall performance** with a mean F1 of 0.715
	- SDGs 1 ('No Poverty') and 16 ('Strong Institutions') perform poorly (F1<0.5), likely because of the small number of positives in the labeled training data. Following more CAPTCHA responses, **model performance can increase over time**
	
- Out of the individual classification models, the **Mistral LLM clearly scores the highest** (F1: 0.61)
	- **All other single classification models score very poorly** (F1<0.4) on balancing precision and recall in patent data
- The **full ensemble model outperforms all submodels**, but the difference with only using Mistral is quite small (F1 0.69 vs. 0.61)
	- Because of the nature of the patent data and classification task, performance increase levels off for well-performing models
- The **ensemble model without Mistral outperforms the Mistral LLM** on average and for 9 out of the 16 SDGs, demonstrating the true benefit of the ensemble approach
	- The ensemble model is able to combine simple models which perform poorly by themselves into a reliable model which outperforms a computationally expensive LLM
	- The ensemble model without Mistral can act as a **much more energy-efficient alternative to using an LLM**, with similar performance for this classification task


## 6. Acknowledgements
Wulff, D. U., Meier, D. S., & Mata, R. (2024). Using novel data and ensemble models to improve automated labeling of Sustainable Development Goals. Sustainability Science. https://doi.org/10.1007/s11625-024-01516-3

Sadick, A.M. (2023). SDG classification with BERT. https://huggingface.co/sadickam/sdg-classification-bert
