# Meeting 27.03
1. Feature extractions and preparation of the training data:

   1.1. Decide for label for the supervised learning - Pneumonia or Pneumotorax (Pneumonia may be hard to detect using only images and not clinical history)

   1.2. Features extraction directly from the images:
      - corners detection, grey scale histogram, letters in the edges (traditional features extracted from the images)
      - medical papers supported features: assymetry, density (consolidation), presense of tubes (more like a negative feature(prone to bias the model)), check for more
      - for tubes detection check the other group project
      - use masks to ensure capturing only the lungs - better for assymetry and will exclude the corners (noise and unnecassary letters)
  
2. Models and experiments

   2.1. Use all of the features together (decided not to use different trining sets, but to have different models to evaluate the detection of the disease)

   2.2. Decide on the labels
      - Having Pneumonia/Not having Pneumonia
      - Having Pneumonia/Having another disease/Being healthy
   
   2.3. Have a simple base model (KNN, Logistic regression) and choose evaluation metrics (AUC, accuracy)

   2.4. Prepare more complex models for further comparisson
      - Random forest
      - Any NN architecture that showed significant thruthfulness in image recognition (e.g. ResNet)
  
3. Literature review

   3.1. Think of the structure of the paper - what parts/segments I want to have in the LR

   3.2. Medical papers part:
      - It is good to have a few papers(2,3) to sow the medical significance some fetaures have (e.g, the ones extracted from the images)
      - Use scientifically proven papers
   
   3.3. Technical part:
      - E.g. papers about consolidation detection in X-rays images
      - ResNet explaination and the benefits of using such network
   
   3.4. Part combining the previous two parts:
      - Significance of having ML tools analyzing medical images
      - Benefits of having more clinically proven features when developing ML models for medical analysis

# Questions for meeting on 27.03:
1. About features extraction and preparation of the training data:

   1.1 As I want to do a supervised learning, should I have only one of the diseases (e.g. Pneumonia) as a label for checking how well the model is performing on detecting the said illness (using meta data and the presense of the other diseases as features), or I should use all of the diseases as labels and check how the different models and the different training datasets detect the different diseases?

   1.2 Regarding features extraction directly from the images:
      - Assymetry: size and shape of the two lungs
      - Density of each side: medical reports state that if one of the lungs is dense and white it indicates for consolidation (Lung consolidation is when the air in the small airways of the lungs is replaced with a fluid, solid, or other material such as pus, blood, water, stomach contents, or cells. It can be caused by conditions like aspiration, pneumonia, and lung cancer.)
      - Presense of tubes
  
3. About literature review:
  - Still at the very beginning (taking more detailed notes on the papers)
  - Q: Amount of papers that should be included in the review and the amount of pages that are expected as a final product (for the review and for the project as a whole)


# Update on the project until 25.03.2025
1. Finished exploring the data
   - performed statistical analysis on all the features
   - prepared plots for every feature that can be extracted from the csvs
2. Started on finding ways of how to extract features from the images
3. Started on the literature review - preparing notes and drafts on the papers used for the problem statement
   - general idea and practices used so far in the topic research

# 20.02.2025
1. Report from Gabi's side:
   - connected to the HPC
   - downloaded the data
   - set up the folders
   - started on the EDA
   - send the GitHub repo link

# 13.02.2025 - Individual meeting with Amelia and Veronika
1. Discussed different approaches for the model training and testing:
  - using only simple features, extracted directly from the images for the training
  - using simple features + using democraphics data available in the meta data part of the dataset
  - using only demographics data for training

2. Features extraction
  - try with histograms
  - then try feature extraction with neural networks

3. For the model I will use random forest (just a suggestion)
   - will try to detect chest drains
  
4. Evaluation of the models
   – evaluate biases on different models
   - for each model I will report not only perfromance metrics, but also training time (important for model evaluation)
   - after modle training -> plot histogram of posterior probability (best case scenario: increase)
   - metrics reloaded paper (dont read the whole paper, just the calibration metrics)
   - read Trine and Cathrine's master thesis for brief overview of 'Metrics reloaded' paper

# 06.02.2025 - Lab meeting
1. Find out about different projects and how the meetings are usually led

2. Should prepare for every meeting a presentation on what I have done since the last meeting

3. Share the repo and the Notion notes with Amelia and Veronika

4. No lab meeting on 13.02.2025

# 03.02.2025 - Individual meeting
1. Decided on weekly meetings: 
  - every Thursday 11:00 - 11:30
2. Decided on a deadline for the final draft
  - 30.04.2025

3. Discussed the use of HPC, needed datsets should be already in the cluster

4. A good practice - update the meeting agenda the day before so Amelia and Veronika coild have time to take a look at the code/question

5. In march Amelia would be on a holiday
