# Questions and updates for meeting on 10.03.2025
1. After trying logistic regression and KNN, I decided to use XGBoost because:

   1.1. There are a lot of NaNs for different diseases; e.g. a disease was not examined for so no data available
   
   1.2. The missing data caused the model to fail training
   
   1.3. After researching different approaches how to handle NaN in that case, I found out that all the possible solutions would alter my results (removing all NaNs (impossible, because a lot of data would be removed); exchanging the NaN values with the mean/SD/most frequent label -> the absence of data is desired)

   1.4. XGBoost, CatBoost and LightGBM handle NaN data, decided on XGBoost as the one I feel most comfortable about. The three of them are gradient boosting algorithms and XGBoost and LightGBM have tree-like structure. CatBoost handles numerical and categorical features as two different categories (not very familiar with it, sounds useful)

   1.5. XGBoost reported accuracy of 0.8435 (only on csv data + grey scale histograms + corners detection)

3. Managed to extract grey-scale histogram and corner detection features from the data and added them to the training data

4. Had some problems when applying the masks on top of the images as they used RLE coordinates and it looks not right on the merged image, but I am working on it.

5. Plan for next week:

   4.1. Finish with the masks

   4.2. Start with DL models and maybe finish them

   
# Questions and updates for meeting on 03.04.2025
1. Oral exam date and external examiner

2. Pneumonia vs Pneumothorax vs Pleural Effusion:
   - Pneumonia is detected using CT (computed tomography) and if the cause is uncertain then a X-ray is ordered
   - Pneumothorax is detected usign X-ray and can be easily seen on the images (bonus: there are formulas used to measure the impact of the pneumothorax)
   - Pleural Effusion - the disease we have the most labels for; means fluids in the lungs (blood, water, ect.), pneumonia may be the cause of it, after threatment it may cause peumothorax
  
3. Plan for features:
   - all the features from the csv where each patient will have as many entries in the final dataset as the number of images in their folder
   - greyscale hystogram
   - prsence of tubes
   - impact formula
   - still researching on assymetry, but most certainly will include it
   - will try also PCA to see if it makes difference for the final results
  
4. Plan until next meeting on 10.04:
   - combine the masks with the images
   - extract all the features needed
   - prepare all the data
   - do the baseline model + one of the DL models

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
