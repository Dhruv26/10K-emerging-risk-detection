# Emerging Risk Detection
Company annual financial reports are a very useful source of information about the company's **performance**, **opportunities** and **risks** in the industries. Considering the amount of data contained in every report and large number of reports from different companies available, there is a strong motivation to automate processing of this information, therefore gaining more insights into the current state of industries.

This project is aimed to create an automated method that is able to  
identify emerging risks faced by multiple businesses and industries,  
and the trends of those risks.  
The research focuses on **topic modelling**, **keyword extraction**, **sentiment analysis** and **clustering** in *annual financial reports*.  
It compares the performance of the following methods when it comes to identifying emerging risks and clustering them:-
- topic modelling
- keyword extraction, supplemented with sentiment analysis and clustering

Extracting the most useful information from the textual financial  
report data and correctly interpreting it is an ongoing challenge and  
hopefully methods used in this project will prove to be useful in future
research.

This repo contains a framework that:-
- Analyze the dataset and 10-K reports
- Extract the *Risk Factors* section from 10-K reports
- Develop a method to identify risks disclosed in the *Risk Factors* section
- Classify a risk as an emerging one
- Analyze and evaluate the results

The framework contains two main packages:-
- `risk_detection.preprocessing`: This contains code to process 10-K reports and extract the **Risk Factors** section from *10-K Reports*.
- `risk_detection.analysis`: This contains code to analyze the *Risk Factors* sections. It is responsible for training **Topic Models**, **Keyword Extraction**, **Sentiment Analysis** and **Clustering**. It also contains an algorithm which compares clusters across years and detect new clusters. This package also contains code to group 10-K reports by industries and train any models required separately.

Most of the analysis has been done in Jupyter Notebooks and are available in the *notebooks/final_submission* folder.
