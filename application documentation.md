# Application Documentation Template

**Application Owner**: Alessio Buda, Tiziano Bardini, em-rg, Danilo Santo
<br>**Document Version**: 1.2
<br>**Reviewers**: Alessio Buda, Tiziano Bardini, em-rg, Danilo Santo

## Key Links

* [Code Repository](https://github.com/alessio-buda/AI-Academy-Final-Project/tree/main)
* [Deployment Pipeline](https://github.com/alessio-buda/AI-Academy-Final-Project/blob/main/deliverables/DOCUMENTATION.md)
* [API](https://github.com/alessio-buda/AI-Academy-Final-Project/blob/main/deliverables/DOCUMENTATION.md) ([Swagger Docs]())
* [Cloud Account](https://github.com/alessio-buda/AI-Academy-Final-Project/blob/main/deliverables/DOCUMENTATION.md)
* [Project Management Board](https://github.com/alessio-buda/AI-Academy-Final-Project/blob/main/deliverables/DOCUMENTATION.md)
* [Application Architecture](https://github.com/alessio-buda/AI-Academy-Final-Project/blob/main/deliverables/DOCUMENTATION.md)

## General Information 

<div style="color: gray">
EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 1, 2, 3
<!-- info: this section covers the AI Act requirement of a description of the intended purpose, version and provider, relevant versions and updates. In Article 11, 2(d) a datasheet is required which describes all training methodologies and techniques as well as the characteristics of the training dataset, general description of the dataset, information about their provenance, scope and main characteristics, how the data was obtained and selected, labelling procedures conducted, and data cleaning methodologies deployed. -->
<p></p>
</div>


**Purpose and Intended Use**:

The initial purpose and reasoning for creating the model is to develop an enterprise-ready solution for automated report generation. The model is designed with a modular architecture, comprehensive security features, and professional output quality, making it suitable for client deliverables and production environments. This intended use suggests a focus on professional and secure applications in business contexts. The AI application aims to solve the problem of automated report generation by providing a sophisticated, enterprise-ready solution that emphasizes reliability, scalability, and maintainability. It ensures high standards of security, performance, and professional output quality, making it suitable for client deliverables and production environments. Users are Technical Audience, Business Audience and General Audience. Security measures and operational safeguards to prevent misuse, such as input security, API security, and development security practices. These include detection of prompt injection attacks, content filtering, secure storage of API keys, rate limiting, and regular security assessments.
cloud-based operations for API calls and model access. The system architecture includes modularity and reliability, and there is a reference to running a local Qdrant server using Docker



## Risk classification

<div style="color: gray">
Prohibited Risk: EU AI Act Chapter II <a href="https://artificialintelligenceact.eu/article/5/" style="color:blue; text-decoration:underline">Article 5</a>
<br>High-Risk: EU AI Act Chapter III, Section 1 <a href="https://artificialintelligenceact.eu/article/6/" style="color:blue; text-decoration:underline">Article 6</a>, <a href="https://artificialintelligenceact.eu/article/7/" style="color:blue; text-decoration:underline">Article 7</a>  
<br>Limited Risk: Chapter IV <a href="https://artificialintelligenceact.eu/article/50/" style="color:blue; text-decoration:underline">Article 50</a>
<p></p>
</div>

<!--info: The AI Act classifies AI systems into four different risk categories. The EU AI Act categorizes AI systems into four risk levels: unacceptable, high, limited, and minimal risk, each with corresponding regulatory requirements.  
Unacceptable risk (Chapter II, Article 5) includes systems that pose a clear threat to safety or fundamental rights (e.g. social scoring, recidivism scoring) and are banned.  
High-risk systems are delineated in Chapter III, Section 1, Articles 6 and 7, including AI used in sensitive domains like healthcare, law enforcement, education, employment, and critical infrastructure. These must meet strict requirements and conduct conformity assessment practices, including risk management, transparency, and human oversight.  
Limited-risk systems, delineated in Chapter IV Article 50, such as chatbots, must meet transparency obligations (e.g. disclosing AI use).  
Minimal-risk systems, like spam filters or AI in video games, face no specific requirements. -->

* Limited
* System that requires authorization and transparency for the user
   
## Application Functionality 

<div style="color: gray">
EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>, paragraph 1, 2, 3
<!-- Info: this section covers the delineation of the general purpose of the system required in Article 1, with a focus on defining what the system should do and how it should work.-->
<p></p>
</div>


* **Instructions for use for deployers**: <div style="color: gray">(EU AI Act <a href="https://artificialintelligenceact.eu/article/13/" style="color:blue; text-decoration:underline">Article 13</a>)</div>
* **Model Capabilities**:
  * Modular architecture with three specialized crews for input sanitization, analysis, and report writing .
  * AI-powered content generation using advanced large language models for intelligent report creation .
  * Retrieval-Augmented Generation (RAG) integration with Qdrant vector database for enhanced knowledge capabilities .
  * Flexible output formats with detailed artifacts and process summaries .
  * Enterprise-grade documentation and reporting suitable for client deliverables
  * Dependency on external services like Azure OpenAI and Qdrant, which may encounter connectivity or configuration issues
* **Input Data Requirements**:
  * Inputs must adhere to the required parameters:
      - project_description: A detailed string describing the project.
      - outline: A comma-separated string of topics to cover.
      - audience: A string specifying the target audience ("technical", "business", or "general").
      Inputs must be sanitized and validated for security, including schema validation and risk assessment
  * ' user_input = {
    "project_description": """
    A microservices-based e-commerce platform built with:
    - Node.js and Express for API services
    - React for frontend
    - MongoDB for product catalog
    - Redis for session management
    - Docker for containerization
    """,
    "outline": "Architecture, Microservices Design, Data Flow, Security, Scalability, Deployment",
    "audience": "technical"
    }
    '
* **Output Explanation**:
  * Outputs are generated as professional-grade documentation and reports tailored to the specified audience and topics.
    Outputs include detailed artifacts and process summaries, ensuring clarity and usability
* **System Architecture Overview**:
  * The system operates with a modular architecture comprising three specialized crews using CrewAI:
        Input sanitization.
        Analysis.
        Report writing.
    AI-powered content generation is integrated with Retrieval-Augmented Generation (RAG) using the Qdrant vector database

## Models and Datasets

<div style="color: gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 2 (d)
<p></p>
</div>

<!--All information about models and datasets that are used in the application should be found in their respective dataset or model documentation.  The purpose here is mainly to provide links to those documentation. --> 
<!--In Article 11, 2 (d) a datasheet is required which describes all training methodologies and techniques as well as the charatcteristics of the training dataset, general description of the dataset, information about their provenance, scope and main characteristics, how the data was obtained and selected labelling procedures conducted and data cleaning methodologies deployed -->

### Models

Link to all model integrated in the AI/ML System

| Model   | Link to Single Source of Truth | Description of Application Usage |
|---------|--------------------------------|----------------------------------|
| Model 1 | [Sanitize Crew](https://github.com/alessio-buda/AI-Academy-Final-Project/tree/main/report_generator/src/report_generator/crews/sanitize_crew)  | To ensure input security by detecting and preventing prompt injection attacks, inappropriate content, and security threats. It also improves and clarifies user queries based on security validation.                              |
| Model 2 | [Analysis Crew](https://github.com/alessio-buda/AI-Academy-Final-Project/tree/main/report_generator/src/report_generator/crews/analysis_crew)  | Comprehensive project analysis and research                              |
| Model 3 | [Writer Crew](https://github.com/alessio-buda/AI-Academy-Final-Project/tree/main/report_generator/src/report_generator/crews/writer_crew)  | Professional report generation and formatting                              |
| Model 4 | [GitHub Repo](https://github.com/alessio-buda/AI-Academy-Final-Project/blob/main/deliverables/DOCUMENTATION.md)             | ...                              |

### Datasets

Link to all dataset documentation and information used to evaluate the AI/ML System.  
(Note, Model Documentation should also contain dataset information and links for all datasets used to train and test each respective model) 

| Dataset   | Link to Single Source of Truth | Description of Application Usage |
|-----------|--------------------------------|----------------------------------|
| Dataset 1 | [MLFlow Data Document](https://github.com/alessio-buda/AI-Academy-Final-Project/tree/main/report_generator/src/report_generator/evaluation)   | ...                              |
| Dataset 2 | [GitHub Repo](https://github.com/alessio-buda/AI-Academy-Final-Project/tree/main)             | ...                              |

## Deployment
    
* The project utilizes Qdrant and Azure OpenAI as follows:
    Qdrant: It is used as a vector database for Retrieval-Augmented Generation (RAG) implementation.
    Azure OpenAI: Azure OpenAI GPT models are the primary language models used for text generation in the project.


### Infrastructure and Environment Details

* **Cloud Setup**:
  * Qdrant and Azure Open AI
* **APIs**:
  * Azure OpenAI service setup:
    - by env global configuration, as:
      # Azure OpenAI Configuration (Required)
      AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
      AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
      AZURE_OPENAI_API_VERSION=2024-02-15-preview
      AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

      # Qdrant Vector Database Configuration (Required)
      QDRANT_URL=http://localhost:6333  # For local Qdrant
      # QDRANT_URL=https://your-cluster.qdrant.io  # For Qdrant Cloud
      # QDRANT_API_KEY=your_qdrant_api_key_here     # For Qdrant Cloud only
  * Qdrant setup:
    - by cloud, using [Qdrant Cloud](https://cloud.qdrant.io/)
    - Local using Docker as "docker run -p 6333:6333 qdrant/qdrant"

## Integration with External Systems

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 1 (b, c, d, g, h), 2 (a)
  <p></p>
</div>

* **Systems**:
  * UV package manager (recommended) or pip
  * python 3.10>
  * Git for repository cloning
  * Minimum 4GB RAM available
  * Troubleshooting Tests:
      Common issues include:
      MLflow connection errors: Ensure MLflow server is running.
      Azure OpenAI timeouts: Check API quotas and connectivity.
      Qdrant connection issues: Verify vector database is accessible.
      Memory issues during evaluation: Use smaller test datasets.

## Deployment Plan



## Lifecycle Management

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 6
  <p></p>
</div>
    
* **Metrics**:
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

  Common issues such as MLflow connection errors, API timeouts, and memory issues are identified, along with troubleshooting steps. The system's modular architecture is designed for adaptability to evolving requirements. 
* **Documentation Needs**:
  * **Monitoring Logs**: Real-time data on accuracy, latency, and uptime as part of monitoring performance.
  * **Incident Reports**: Incident response procedures and monitoring are part of operational security


### Risk Management System

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/9/" style="color:blue; text-decoration:underline">Article 9</a>
  <br>EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>
  ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>
  <p></p>
</div>
<!--**Instructions:**  A thorough risk management system is mandated by the AI Act, especially for high-risk AI systems. This section documents the  proactive efforts to ensure the AI system operates safely and ethically. In general in this section you should document all the measures undertaken to make sure that a system operates safely on the market. Example: Consider a facial recognition system used for real-time law enforcement in public spaces. This is categorized as high-risk under the EU AI Act. If developers document the risk that the system might misidentify individuals—particularly among minority groups due to biased training data—they can plan for rigorous dataset audits, independent bias testing, and establish human oversight in decision-making. Without documenting this risk, the system might be deployed without safeguards, leading to wrongful detentions and legal liabilities. Systematic documentation ensures these issues are not only identified but addressed before harm occurs.-->


**Risk Assessment Methodology:** Detection of prompt injection attacks, content filtering, schema validation, and risk assessment and categorization, regular monitoring of API usage, secure storage of credentials, and rate limiting to prevent abuse.

**Identified Risks:** 

**Potential Harmful Outcomes:** The possible negative effects of the identified risks include:

- Biased Decisions: Resulting from manipulated or malicious inputs.
- Privacy Breaches: Exposure of personal information due to inadequate anonymization or secure transmission.
- System Downtime: Caused by resource mismanagement or API quota exhaustion.
- Unauthorized Access: Leading to misuse of API credentials or tokens


#### Risk Mitigation Measures

**Preventive Measures:** Detail actions taken to prevent risks, like implementing data validation checks or bias reduction techniques.

**Protective Measures:** Describe contingency plans and safeguards in place to minimize the impact if a risk materializes.

## Testing and Validation (Accuracy, Robustness, Cybersecurity)

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/15/" style="color:blue; text-decoration:underline">Article 15</a>
  <p></p>
</div>

**Testing and Validation Procedures (Accuracy):**

**Performance Metrics:** List the metrics used to evaluate the AI system, such as accuracy, precision, recall, F1 score, or mean squared error.

**Validation Results:** Summarize the outcomes of testing, including any benchmarks or thresholds met or exceeded.

**Measures for Accuracy:** High-quality data, algorithm optimisation, evaluation metrics, and real-time performance tracking.

  
### Accuracy throughout the lifecycle

**Data Quality and Management:** High-Quality Training Data: Data Preprocessing, techniques like normalisation, outlier removal, and feature scaling to improve data consistency, Data Augmentation, Data Validation

**Model Selection and Optimisation:** Algorithm selection suited for the problem, Hyperparameter Tuning (grid search, random search, Bayesian optimization), Performance Validation( cross-validation by splitting data into training and testing sets, using k-fold or stratified cross-validation), Evaluation Metrics (precision,recall, F1 score, accuracy, mean squared error (MSE), or area under the curve (AUC).

**Feedback Mechanisms:** Real-Time Error Tracking, Incorporate mechanisms to iteratively label and include challenging or misclassified examples for retraining.

### Robustness 

<-- Add outlier detection and all possible post analysis, what are the criticalities -->

**Robustness Measures:**

* Adversarial training, stress testing, redundancy, error handling, and domain adaptation.

**Scenario-Based Testing:**

* Plan for adversarial conditions, edge cases, and unusual input scenarios.
    
* Design the system to degrade gracefully when encountering unexpected inputs.
    

**Redundancy and Fail-Safes:**
    
* Introduce fallback systems (e.g., rule-based or simpler models) to handle situations where the main AI system fails.
    
**Uncertainty Estimation:**
    
* Include mechanisms to quantify uncertainty in the model’s predictions (e.g., Bayesian networks or confidence scores).
    

### Cybersecurity 

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 2 (h)
  <p></p>
</div>

**Data Security:**

**Access Control:**

**Incident Response :**


These measures include threat modelling, data security, adversarial robustness, secure development practices, access control, and incident response mechanisms.

Post-deployment monitoring, patch management, and forensic logging are crucial to maintaining ongoing cybersecurity compliance.

Documentation of all cybersecurity processes and incidents is mandatory to ensure accountability and regulatory conformity.

  

## Human Oversight 

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>;; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 2(e)
  <br>EU AI Act <a href="https://artificialintelligenceact.eu/article/14/" style="color:blue; text-decoration:underline">Article 14</a>
  <p></p>
</div>

<!-- info: AI Act Article 11, paragraph 2(e) requirements: assessment of the human oversight measures needed in accordance with Article 14, including the assessment of the technical measures needed to facilitate the integration of the outputs of the AI systems by deployers. -->


**Human-in-the-Loop Mechanisms:**  Explain how human judgment is incorporated into the AI system’s decision-making process, such as requiring human approval before action.

**Override and Intervention Procedures:** Describe how users or operators can intervene or disable the AI system in case of errors or emergencies.

**User Instructions and Training:** Provide guidelines and training materials to help users understand how to operate the AI system safely and effectively.

**Limitations and Constraints of the System:** Clearly state what the AI system cannot do, including any known weaknesses or scenarios where performance may degrade.


## Incident Management
<!-- what happens when things go wrong. This part is particularly important to provide information on how incidents were dealth with and the processes put in place to minimize damage when things go wrong. -->
* **Common Issues**:
  * List common errors and their solutions.
  * Logs or debugging tips for advanced troubleshooting.
* **Support Contact**:
  * How to reach technical support or community forums.


### Troubleshooting AI Application Deployment

This section outlines potential issues that can arise during the deployment of an AI application, along with their causes, resolutions, and best practices for mitigation.


#### Infrastructure-Level Issues

##### Insufficient Resources

* **Problem**: Inaccurate resource estimation for production workloads.
  * Unexpected spikes in user traffic can lead to insufficient resources such as compute, memory or storage that can lead to crashes and bad performance

* **Mitigation Strategy**:
<!-- describe here the resolution strategy such as:
-  Enable autoscaling (e.g., Kubernetes Horizontal Pod Autoscaler).
  - Monitor usage metrics and adjust resource allocation dynamically.
  - Implement rate-limiting for traffic spikes. -->


##### Network Failures

* **Problem**:  network bottlenecks  can lead to inaccessible or experiences latency of the application.

* **Mitigation Strategy**:
<!-- 
  - Test network connectivity 
  - Use content delivery networks (CDNs) or regional load balancers.
  - Ensure proper failover mechanisms.-->


##### Deployment Pipeline Failures

* **Problem**: pipeline fails to build, test, or deploy because of issues of compatibility between application code and infrastructure, environment variables or credentials misconfiguration.

* **Mitigation Strategy**: 
<!--:
  - Roll back to the last stable build.
  - Fix pipeline scripts and use containerisation for environment consistency.
  - Enable verbose logging for error diagnostics.-->


#### Integration Problems

##### API Failures

* **Problem**: External APIs or internal services are unreachable due to network errors or authentication failures.

* **Mitigation Strategy**:
<!--:
  - Implement retries with exponential backoff.
  - Validate API keys or tokens and refresh as needed.
  - Log and monitor API responses for debugging. -->

##### Data Format Mismatches

* **Problem**: Crashes or errors due to unexpected data formats such as changes in the schema of external data sources or missing data validation steps.

* **Mitigation Strategy**: 

<!--
  - Use schema validation tools (e.g., JSON schema validators).
  - Add versioning to APIs and validate inputs before processing.-->

#### Data Quality Problems

* **Problem**: Inaccurate or corrupt data leads to poor predictions.
* **Causes**:
  * No data validation or cleaning processes.
  * Inconsistent labelling in training datasets.

* **Mitigation Strategy**: 
<!--
- **Resolution**:
  - Automate data quality checks (e.g., Great Expectations framework).
  - Regularly audit and clean production data.-->


#### Model-Level Issues

##### Performance or Deployment Issues

* **Problem**: Incorrect or inconsistent results due to data drift or inadequate training data for the real world deployment domain. 

* **Mitigation Strategy**:

<!--
- **Resolution**:
  - Monitoring for data drift and retraining of the model as needed.
  - Regularly update the model -->


#### Safety and Security Issues

##### Unauthorised Access

* **Problem**: Sensitive data or APIs are exposed due to misconfigured authentication and authorization.

##### Data Breaches

* **Problem**: User or model data is compromised due to insecure storage or lack of monitoring and logging of data access. 

* **Mitigation Strategy**: 
<!--
- **Resolution**:
  - Use secure storage services (e.g., AWS KMS).
  - Implement auditing for data access and alerts for unusual activity.
  6.1. Delayed or Missing Data-->


#### Monitoring and Logging Failures

##### Missing or Incomplete Logs

* **Problem**: Lack of information to debug issues due to inefficient logging. Critical issues go unnoticed, or too many false positives occur by lack of implementation ofactionable information in alerts. 

* **Mitigation Strategy**: 


<!--
- **Resolution**:
  - Fine-tune alerting thresholds and prioritise critical alerts.
  - Use tools like Prometheus Alertmanager to manage and group alerts. -->


#### Recovery and Rollback

##### Rollback Mechanisms

* **Problem**: New deployment introduces critical errors.

* **Mitigation Strategy**: 

<!--
- **Resolution**:
  - Use blue-green or canary deployments to minimise impact.
  - Maintain backups of previous versions and configurations. -->

##### Disaster Recovery

* **Problem**: Complete system outage or data loss.

* **Mitigation Strategy**:

<!--
- **Resolution**:
  - Test and document disaster recovery plans.
  - Use automated backups and verify restore procedures.-->

### EU Declaration of conformity 

<div style="color: gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/47/" style="color:blue; text-decoration:underline">Article 47</a>
  <p></p>
</div>

<!-- when applicable and certifications are available: it requires a systems name as well as the name and address of the provider; a statement that the EU declaration of conformity referred to in Article 47 is issued under the sole responsibility of the provider; a statement that the AI system is in conformity with this Regulation and, if applicable, with any other relevant Union law that provides for the issuing of the EU declaration of conformity referred to in Article 47, Where an AI system involves the processing of personal data;  a statement that that AI system complies with Regulations (EU) 2016/679 and (EU) 2018/1725 and Directive (EU) 2016/680, reference to the harmonised standards used or any other common specification in relation to which
conformity is declared; the name and identification number of the notified body, a description of the conformity
assessment procedure performed, and identification of the certificate issued; the place and date of issue of the declaration, the name and function of the person who signed it, as well as an
indication for, or on behalf of whom, that person signed, a signature.-->

### Standards applied

<!-- Document here the standards and frameworks used-->

## Documentation Metadata

### Template Version
<!-- info: link to model documentation template (i.e. could be a GitHub link) -->

### Documentation Authors
<!-- info: Give documentation authors credit

Select one or more roles per author and reference author's
emails to ease communication and add transparency. -->

* **Name, Team:** (Owner / Contributor / Manager)
* **Name, Team:** (Owner / Contributor / Manager)
* **Name, Team:** (Owner / Contributor / Manager)
