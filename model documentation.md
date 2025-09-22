# Model Documentation Template
AI Academy Report Generator

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>, paragraph 1
    <!-- The AI Academy Report Generator is designed for automated report generation using the CrewAI framework. The system requires Python 3.10 or higher, at least 4GB RAM, and a stable internet connection for API calls and model access. It also depends on external services like Azure OpenAI for API integration. Minimum hardware requirements include 4GB RAM (8GB+ recommended), 2GB free disk space. Ensure the package is installed in editable mode: "pip install -e" , Check Python version compatibility (3.10+ required), verify all dependencies are properly installed. 
    The AI system is put into service through a modular, three-stage pipeline architecture, where specialized AI crews work sequentially to perform distinct tasks. These crews are configured independently and are responsible for specific functions:
      Sanitize Crew: This crew is responsible for input sanitization. It includes agents like the "input_checker" and "input_sanitizer," which validate the security and safety of the input, detect potential threats, and optimize the user query. These agents are configured using YAML files and utilize the Azure GPT-4o model for their tasks.
      Analysis Crew: Although the detailed configuration of this crew is not provided in the context, it is part of the sequential pipeline and likely focuses on analyzing the sanitized input to extract insights or prepare data for the next stage.
      Writer Crew: This crew is tasked with professional report generation and formatting. It includes components such as the RAG Searcher for retrieving information from the knowledge base and the Writer for creating comprehensive content tailored to the target audience. The output of this crew includes the final report and a generation summary.
    The system is designed to optimize performance through token management, parallel processing, caching, and scalability practices such as horizontal scaling and resource management. Security considerations are also integral, with the Sanitize Crew implementing measures to ensure input security. -->
    <p></p>
</div>

**Model Owner**: Alessio Buda, Tiziano Bardini, em-rg, Danilo Santo
<br>**Document Version**: 1.2
<br>**Reviewers**: Alessio Buda, Tiziano Bardini, em-rg, Danilo Santo

## Overview 

<div style="color:gray">
EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>, paragraph 1

<!-- info: This section enables all stakeholders to have a glimpse into the model selection, design, and development processes.  
You can use this section to provide transparency to users and high-level information to all relevant stakeholders.-->
<p></p>
</div>

### Model Type

**Model Type:** Multiagent platform named CrewAI

### Model Description 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 1(a)
    <p></p>
</div>

* The AI system is put into service through a modular, three-stage pipeline architecture, where specialized AI crews work sequentially to perform distinct tasks. These crews are configured independently and are responsible for specific functions:
      Sanitize Crew: This crew is responsible for input sanitization. It includes agents like the "input_checker" and "input_sanitizer," which validate the security and safety of the input, detect potential threats, and optimize the user query. These agents are configured using YAML files and utilize the Azure GPT-4o model for their tasks.
      Analysis Crew: Although the detailed configuration of this crew is not provided in the context, it is part of the sequential pipeline and likely focuses on analyzing the sanitized input to extract insights or prepare data for the next stage.
      Writer Crew: This crew is tasked with professional report generation and formatting. It includes components such as the RAG Searcher for retrieving information from the knowledge base and the Writer for creating comprehensive content tailored to the target audience. The output of this crew includes the final report and a generation summary.
<!-- info: Brief (max 200 words) description of the model architecture and the task(s) it was trained to solve, architecture (e.g., FPN, PointRend, MnasNet), size/latency optimization, and intended purpose. -->

### Status 
<!-- scope: telescope -->
<!-- info: Select **one:** -->
**Status Date:** 2025-09-19

**Status:** Regularly Updated

<!-- * **Under Preparation** -- The model is still under active development and is not yet ready for use due to active "dev" updates.
* **Regularly Updated** -- New versions of the model have been or will continue to be made available.
* **Actively Maintained** -- No new versions will be made available, but this model will be actively maintained.
* **Limited Maintenance** -- The model will not be updated, but any technical issues will be addressed.
* **Deprecated** -- This model is obsolete or is no longer being maintained. -->

### Relevant Links
<!-- info: User studies show document users find quick access to relevant artefacts like papers, model demos, etc..
very useful. -->

[AI-Academy-Final-Project](https://github.com/alessio-buda/AI-Academy-Final-Project)

### Developers

* Alessio Buda
* Tiziano Bardini 
* em-rg
* Danilo Santo

### Owner
<!-- info: Remember to reference developers and owners emails. -->
* Alessio Buda
* Tiziano Bardini 
* em-rg
* Danilo Santo

## Version Details and Artifacts 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 1(c)
    <p></p>
</div>

<!-- scope: periscope -->
<!-- info: Provide details about the current model version
and which model version the current model card corresponds to.

For models without version number, use "Not currently tracked"
but be sure to track the release date of the model.
-->


**Current Model Version:** 1.2

**Model Version Release Date:** 2025/09/05

**Model Version at last Model Documentation Update:** 1.2

**Artifacts:**

* Azure OpenAI models
* YAML files for CrewAI <!-- (s3://your-bucket/path/to/config.yaml)-->

## Intended and Known Usage

### Intended Use
<!-- info: This section focuses on the initial purpose and/or reasoning
for creating the model. It is important to define this section as the intended use directly affects the AI Act classification. For example:
A face recognition model for personal photo apps → Limited risk
The same model used for law enforcement → High or unacceptable risk


Example Use Case: A university research team develops a machine learning model to predict the likelihood of hospital readmission among diabetic patients over the age of 65, using data from a regional healthcare network. The model is trained and validated specifically on this elderly population and is intended to support hospital planning and academic research. However, the team does not document the model’s intended use or demographic limitations. A health-tech company later integrates the model into a mobile app aimed at helping diabetes patients of all ages manage their care. The model performs poorly for younger users, frequently overestimating their risk of readmission. This leads to unnecessary anxiety, inappropriate self-care decisions, and false alerts to care providers. The misapplication draws criticism for lacking transparency, and regulators question the ethics of deploying a model outside its original context.   -->

The initial purpose and reasoning for creating the model is to develop an enterprise-ready solution for automated report generation. The model is designed with a modular architecture, comprehensive security features, and professional output quality, making it suitable for client deliverables and production environments. This intended use suggests a focus on professional and secure applications in business contexts.

### Domain(s) of use

The domains of use for the AI Academy Report Generator include:
* Client Deliverables: The solution is designed to produce professional-quality reports suitable for delivery to clients. 
* Production Environments: Its modular architecture and comprehensive security features make it appropriate for deployment in production settings. 
* Security Validation and Monitoring: The system incorporates security validation metrics and monitoring through tools like MLflow, making it suitable for environments requiring robust security assessments. 
* Experiment Management and Performance Monitoring: The integration with MLflow for tracking experiments and performance metrics makes it applicable in research and development contexts. 
* Automated Report Generation: The primary domain of use is automated report generation, leveraging advanced AI models for efficiency and accuracy. 


**Specific tasks performed:**
Automated Report Generation, Security Validation and Performance Monitoring


 **Instructions for use for deployers**:

Configure the environment by copying .env.example to .env and editing it with your settings, including the model and Azure OpenAI API details.
For local Qdrant server setup, use Docker or download the binary from the provided link.
Optimize system performance by monitoring memory usage, token consumption, and processing speed. Use caching, pagination, and parallel processing where applicable.
Ensure operational security by securing API keys, updating dependencies, and implementing network security best practices. 
<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 13</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> 
    <p></p>
</div>

### Out Of Scope Uses

Provide potential applications and/or use cases for which use of the model is not suitable.

### Known Applications 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 1(f)
    <p></p>
</div>

It is integrated with MLflow for built-in experiment tracking and evaluation capabilities, supports flexible output formats with detailed artifacts and process summaries, and provides professional-quality enterprise-grade documentation and reporting suitable for client deliverables. Additionally, it is designed for scalability and performance optimization, including caching RAG results and intermediate outputs, non-blocking operations, and efficient handling of large language model operations.

| **Application**   | **Purpose of Model Usage**                                                 | **[AI Act Risk](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)** |
|-------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [Sanitize Crew]() | To ensure input security by detecting and preventing prompt injection attacks, inappropriate content, and security threats. It also improves and clarifies user queries based on security validation. | Limited                                                                                         |
| [Analysis Crew]() | Comprehensive project analysis and research    | Limited  
| [Writer Crew]() | Professional report generation and formatting |  Limited                                                                                         |
                                                                                         |


## Model Architecture 

<div style="color:gray">
EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 2(b), 2(c)

Info – AI Act requirements:  
This section should contain a description of the elements of the model and the processes of its training and development.  

Article 11(2)(b) requires the design specifications of a system, model selection, and what the system is designed to optimize for, as well as potential trade-offs.  

Article 11(2)(c) requires a description of the system’s architecture, how software components are built on or feed into each other, and the computational resources needed to develop, train, test, and validate the system.
</div>


<!-- Info: Describing the architecture is fundamental for reproducibility, transparency, and effective maintenance. Without clear records of the model’s layers, activation functions, input/output shapes, and training configurations, it becomes difficult to reproduce results, debug issues, or update the model reliably.  -->

The AI Academy Report Generator employs a modular, three-stage pipeline architecture. This architecture is designed to function with specialized AI crews working sequentially to achieve the following tasks:

* Input Sanitization: The first stage focuses on cleaning and preparing the input data.
* Analysis: The second stage involves analyzing the prepared data.
* Report Writing: The final stage generates comprehensive, professional reports based on the analysis.

Key components are the three crews used for the project, as Sanitize Crew, Analysis Crew and Writer Crew.
    
The compute resources used for this project include:

* Storage: 2GB of free disk space for dependencies and output files.
* Network: A stable internet connection for API calls and model access [source:temp_uploads\DOCUMENTATION.md, PARAGRAPH: Network].
* Dependencies: UV package manager (recommended) or pip for managing dependencies
    

### Data Collection and Preprocessing

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 2(d)
    <p></p>
</div>

<!--check data documentation to avoid duplicates of information and link it in this sectiion

In Article 11, 2 (d) a datasheet is required which describes all training methodologies and techniques as well as the characteristics of the training dataset, general description of the dataset, information about their provenance, scope and main characteristics, how the data was obtained and selected labelling procedures conducted and data cleaning methodologies deployed -->

      
    
## Model Training Process 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 2(c, g), paragraph 3

<!-- AI Act requirements info:  
In Article 11 paragraph 2(c), details about the computational resources needed to develop, train, test, and validate AI systems are required.  

Moreover, in accordance with Article 11 paragraph 2(g), this section must include the validation and testing procedures used, the data involved, and the main metrics adopted to measure accuracy, robustness, and compliance with the requirements laid out in Chapter III, Section 2.  

Paragraph 3 further requires detailed information about the monitoring, functioning, and control of the system, as well as logging of testing, with reports dated and signed by responsible stakeholders.-->
<p></p>
</div>


**Details of Processes**:

The testing procedure for the Report Generator is implemented through a comprehensive testing and evaluation framework that validates performance, security, and quality across three main evaluation modules:

* Sanitize Crew Evaluation: Focuses on security validation and threat detection testing.
* Analysis Crew Evaluation: Assesses the quality and relevance of project analysis.
* RAG Evaluation: Tests the performance of Retrieval-Augmented Generation.

The framework uses MLflow for experiment tracking and supports both automated and manual testing capabilities. Additionally, the testing files structure and operational security measures are outlined to ensure secure and effective testing processes.
        
## Model Training and Validation 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 2(g)
    <br>EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 15</a>
    <p></p>
</div>

The model's purpose is to evaluate and generate reports based on specific project descriptions, outlines, and target audiences. It involves tasks such as retrieving relevant information, ensuring the faithfulness of responses to the context, and maintaining high-quality outputs.

The business goals for the model include:
* Accuracy: Ensuring the retrieved and generated content is precise and relevant to the input context.
* Fairness: Providing unbiased and contextually appropriate responses.
* Speed: Delivering results efficiently to meet enterprise-level performance standards.

The testing framework consists of three main evaluation modules:
  - RAG System: The evaluation metrics are based on the RAGAS framework and include:
      * Context Precision: Measures the accuracy of retrieved chunks of information.
      * Context Recall: Assesses the coverage of relevant information in the retrieval process.
      * Faithfulness: Evaluates whether the responses are anchored to the provided context.
      * Answer Relevancy: Determines the pertinence of the response to the questions asked.
      * Answer Correctness: Compares the response to ground truth data (when available).
  - Sanitize Crew: 
      * Test Dataset: 34 test cases including:
      - Safe inputs: Technical projects, business presentations, educational content (13 cases)
      - Malicious inputs: Prompt injection, social engineering, inappropriate content (21 cases)
  - Analysis Crew:
      * LLM-as-a-Judge: Semantic relevance evaluation using Azure OpenAI
      * Keyword Coverage: Automated analysis of topic completeness
      * Performance Metrics: Execution time and success rate tracking

<!--
- Performance metrics (e.g., accuracy, F1 score, RMSE) are monitored.
- Documeting this step is important as it enables to detect errors and performance issues early on: Overfitting can be detected using validation loss trends.
-->

<!--### Performance Metrics

- Evaluation Metrics Used (e.g., Accuracy, Precision, Recall, AUC-ROC)
- Benchmarking Results
- Validation Process
- Real-World Performance
- Stress testing
- Performance across different environments and populations -->
    

## Model Testing and Evaluation

<!--
- Performance metrics (e.g., accuracy, F1 score, RMSE) are monitored.
- Documeting this step is important as it enables to detect errors and performance issues early on: Overfitting can be detected using validation loss trends.
-->

<!-- Example: In medical diagnosis, using accuracy alone can be misleading in imbalanced datasets, potentially missing critical cases like cancer. Metrics like recall (which measures the percentage of actual cancer cases the model correctly identifies. Critical for minimizing missed diagnoses), precision ( to ensure that when the model predicts cancer, it’s actually correct—important to reduce false alarms), F1 score, and AUC-ROC provide a more meaningful assessment by accounting for the real-world impact of false positives and false negatives. Choosing the right metrics ensures models are effective, trustworthy, and aligned with practical goals and consequences.

## Model Validation and Testing
- **Assess the metrics of model performance** 
   - accuracy:
   - precision: 
   - recall:
   - F1 score:

- **Advanced performance metrics**
  - ROC-AUC:
    - trade-off between true positive rate and false positive rate
  - PR- AUC
     - Evaluating precision and recall trade-off
  - Specificity
    - (True Negatives/(True Negatives+False Positives))
  - Log Loss (Cross-Entropy Loss):
    - Penalises incorrect probabilities assigned to classes.


- **Context dependant metrics**: 
  - Regression Metrics: For tasks predicting continuous values
  - Clustering Metrics: for tasks grouping similar data points
  - Ranking Metrics: for tasks predicting rankings (e.g., search engines recommendation systems)
  - NLP processing metrics (e.g., text classification, sequence-to-sequence tasks)


- **Fairness Metrics**:
    
    - Ensure the model treats different groups (e.g., based on gender, race) equitably.
    - Examples: Demographic parity, equal opportunity, and disparate impact.
- **Explainability Metrics**:
    
    - Measure how understandable and interpretable are the model’s decisions.
    - Examples: Feature importance, fidelity (how well explanations match the model), and sparsity (using fewer features for explanations).
    - 
- **Robustness Metrics**:
    
    - Assess how well the model performs under challenging or unexpected conditions.
    - Examples: Adversarial robustness, performance under data drift, and sensitivity to input changes.
 
- Limitations of the performance after the tests
- Simulate deployment scenarios to understand real-world implications.
- Define thresholds for acceptable performance levels.
- Justify the choice of metrics based on the application’s purpose.
   
--> 

 **Performance Metrics**:
    
Component	Success Rate	Key Metric	Score
Sanitize Crew	67.6%	Security Accuracy	91.3%
Analysis Crew	100%	LLM Relevance	8.5/10
Writer Crew (RAG)	TBD	Context Precision	TBD


 **Robustness Testing**:

* Tested with diverse inputs to ensure robustness across use cases



### EU Declaration of conformity 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/47/" style="color:blue; text-decoration:underline">Article 47</a>(d)
    <p></p>
</div>

 <!-- when applicable and certifications are available: it requires a systems name as well as the name and address of the provider; a statement that the EU declaration of conformity referred to in Article 47 is issued under the sole responsibility of the provider; a statement that the AI system is in conformity with this Regulation and, if applicable, with any other relevant Union law that provides for the issuing of the EU declaration of conformity referred to in Article 47, Where an AI system involves the processing of personal data;  a statement that that AI system complies with Regulations (EU) 2016/679 and (EU) 2018/1725 and Directive (EU) 2016/680, reference to the harmonised standards used or any other common specification in relation to which
conformity is declared; the name and identification number of the notified body, a description of the conformity
assessment procedure performed, and identification of the certificate issued; the place and date of issue of the declaration, the name and function of the person who signed it, as well as an
indication for, or on behalf of whom, that person signed, a signature.-->
DICHIARAZIONE UE DI CONFORMITÀ
(ai sensi dell’Articolo 47 del Regolamento (UE) 2024/1689)

1. Sistema di IA: AI Academy Report Generator 
2. Fornitore: XYZ s.r.l.
3. La presente dichiarazione di conformità è rilasciata sotto l’esclusiva responsabilità del fornitore.
4. Si dichiara che il sistema di IA sopra identificato è conforme al Regolamento (UE) 2024/1689 e, se applicabile, ad ogni altra legislazione dell’Unione pertinente che prevede la dichiarazione UE di conformità.
5. Il sistema di IA rispetta i requisiti dei Regolamenti (UE) 2016/679, (UE) 2018/1725 e della Direttiva (UE) 2016/680, per quanto applicabile.
6. Standard utilizzati: EN ISO/IEC 42001:2023, EN ISO 9001:2015
8. Luogo e data: Roma, 19 settembre 2025
9. Firmato da: Dott. Mario Rossi – Amministratore Delegato
   Firma: ___________________


### Standards applied

MLflow, API Security, Infrastructure Security, Development Security Best Practices, CrewAI

## Documentation Metadata

### Version
0.1

### Template Version
[Template-Documentation](https://github.com/aloosley/techops/blob/main/template/model%20documentation.md)

### Documentation Authors
<!-- info: Give documentation authors credit

Select one or more roles per author and reference author's
emails to ease communication and add transparency. -->

* Agostino D'Ambrosio
* Claudia Marano 
* Marco Cotugno
* Marco Diomedi

