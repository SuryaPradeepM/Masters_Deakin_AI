# Index

[i. Load the Data](#i.-Load-the-Data)<br>
[1. Subgroups according to columns 3 to 205](#1.-Subgroups-according-to-columns-3-to-205)<br>
[2. Curse of dimensionality problem](#2.-Curse-of-Dimensionality-problem)<br>
[3. Computation of Variance Explained by Principal Components](#3.-Computation-of-Variance-Explained-by-the-Principal-Componentss)<br>
[4. ML Modelling (obesity data)](#4.-ML-Modelling)<br>
<br>
[References](#References)

## i. Load the Data



> We have a large number of dimensions w.r.t data points, just 70 rows and 205 different features and one target class. This is the problem of **curse of dimensionality**. We have to address this by performing dimensionality reduction later.


> **Classes are imbalanced; There are 7 different groups with a imbalanced dataset**



## 1. Subgroups according to columns 3 to 205



#### Subgroups using no. of unique values in selected columns


> We can see that there are 343 different subgroups when considering 3 to 205 columns. And we have seen that from the 206the attribute. We only have 7 unique classes in the labels

#### Also trying grouping with kmeans clustering instead of direct subgroups


> **The value of Silhouette score varies from -1 to 1. If the score is 1, the cluster is dense and well-separated than other clusters. A value near 0 represents overlapping clusters with samples very close to the decision boundary of the neighbouring clusters. A negative score [-1, 0] indicate that the samples might have got assigned to the wrong clusters.**

<h3><b>Checking from the silhouette coefficient value is best for 5 clusters indicating relatively well-seperated clusters with least number of clusters. Next we can check in the silhouette plots to determine optimum number of clusters</b></h3>


### Optimal number of clusters: 
* **3** seems to be suboptimal as:
    * Thicknes of clusters or number of values in clusters varies with some thicker and other thinner clusters
    * One cluster has score lesser than average
* **4** also seems to suboptimal as:
    * Thicknes of clusters or number of values in clusters varies with some thicker and other thinner clusters
* **5** has the best avg. score:
    * Although it has the best avg. score, there are negative values in some clusters
    * But the thickness of clusters varies widely. One cluster is very thick
*  **7** has a much lower avg. silhouette score
* Hence **6** is the optimal number of clusters:
    * Good avg. silhouette score
    * Thickness is comparitively better

    For n_clusters 6, 
    The average silhouette score: 0.26872377
    purity_score: 0.82857143
    

> **We can see that for the 6 number of clusters, we have good silhouette score and purity_score**

> **This is not exactly same as the seven number of classes that is present in the gt. This is because the separation of some of the classes in the gt might not be too well because, we have too few data points in few classes. Also, overall there are very less number of data points to be able to infer a meaningful pattern out of them. Along with a <i>curse of dimensionality problem</i>.**

## 2. Curse of Dimensionality problem

#### The term "Curse of Dimensionality" refers to the explosive nature of increasing data dimensions and the subsequent exponential increase in computer work required for processing and/or analysis. Richard E. Bellman coined the term to describe the increase in volume of Euclidean space associated with adding extra dimensions in the field of dynamic programming. This phenomena is now being observed in domains such as machine learning, data analysis, and data mining, to mention a few. In principle, increasing the dimensions adds more information to the data, boosting its quality, but it actually increases noise and redundancy during analysis.

#### A feature of an item in machine learning might be an attribute or a characteristic that defines it. Each characteristic represents a dimension, and a collection of dimensions forms a data point. This is a feature vector that defines the data point that will be used by a machine learning algorithm (or algorithms). When we talk about increasing dimensionality, we mean increasing the amount of characteristics utilised to describe the data. In the realm of breast cancer research, for example, age and the number of malignant nodes can be used as features to determine a patient's prognosis. A feature vector's dimensions are made up of these features. However, other factors such as previous surgeries, patient history, tumour kind, and other such characteristics assist a doctor in making a diagnosis are adding dimensions to data

#### Hughes (1968) in his study concluded that with a fixed number of training samples, the predictive power of any classifier first increases as the number of dimensions increase, but after a certain value of number of dimensions, the performance deteriorates. Thus, the phenomenon of curse of dimensionality is also known as Hughes phenomenon.

![image.png](attachment:19eb1c61-7ec7-484a-b873-dacfb1d1902d.png)

#### A range of approaches known as 'Dimensionality reduction techniques' are employed to alleviate the issues associated with high dimensional data. Dimensionality reduction approaches are classified into two types: "feature selection" and "feature extraction."

##### **Feature Selection:** Low Variane Filter, High Correlation Filter, Multicollinearity, Feature Ranking

##### **Feature Ranking:** PCA, Factor Analysis, Independent component analysis, t-SNE



## i. Normalizing data

Normalizing before PCA is very important


## ii. PCA
    
![png](Credit%20Task%201%20-%20ML_files/Credit%20Task%201%20-%20ML_46_0.png)


> This plot shows the cumulative percentage of variance explaiend by each additional principal component. We can see that arund 15 principal components are enough to account for approx. 82% of the variance. Also, around 23 principal components explain 90% of the variance and 30 principal components are able to explain a total of 95% of variance of the data. This means, after doing dimensionality reduction with **PCA** we're able to effectively use just 30 principal components instead of the original 70 dimensions

> **This shows that we indeed have a curse of dimensionality problem where having additional features doesn't add to the predictive power of the overall model considering those additional features**

### iii. t-SNE

    NO. of components 2 | KL Divergence 0.09370
    

### iv. Visualize in 2D plot to check for dimensionality and loss of information

> To illustrate the problem, let's consider a two-dimensional plot. Since the dataset has 203 dimensions, we cannot directly visualize it in a traditional scatter plot.

 > However, we can apply dimensionality reduction techniques like PCA and t-SNE to project the dataset onto a lower-dimensional space, such as a 2D space. By applying PCA and t-SNE, we can transform the dataset into a reduced number of components while preserving the most important information. Then, We can then plot the data in this reduced 2D space. 

> **We can see that the clusters are indeed clearly separated when visualized using just two components of PCA and t-SNE. Thus confirming the presence of curse of dimensionality problem.**

> The loss of information can be measured by comparing the explained variance ratio of the original data with that of the reduced data (in 2D). The reduction in the explained variance indicates the loss of information due to dimensionality reduction

> It's important to note that such a plot will only capture a subset of the information present in the original high-dimensional dataset, resulting in a loss of information.

## 3. Computation of Variance Explained by the Principal Components

#### The percentage of variance for the first N components in PCA is computed based on the eigenvalues of the covariance matrix. Steps:


> 1. Compute the covariance matrix of the original dataset
> 2. Perform an eigendecomposition (SVD) of the covariance matrix, which yields eigenvalues and eigenvectors.
> 3. Sort the eigenvalues in descendi and valculate the total sum of all eigenvalues.
> 4. Compute the cumulative sum of the eigenvalues up to the Nth component.
> 5. After getting the principal components, to compute the percentage of variance (information) accounted for by each component, we divide the eigenvalue of each component by the sum of 
eigenvalues.
<br>
> Percentage of Variance: The percentage of variance explained by each principal component is calculated by dividing its eigenvalue by the sum o **The percentage of variance explained by the first N components indicates how much information is retained when reducing the dataset dimensionality.**f all eigenvalues.

> **Percentage of Variance Explained by PCi = (Eigenvalue of PCi) / (Sum of all Eigenvalues)**

> Cumulative Variance: Often, we are interested in the cumulative variance explained by a subset of principal components. This is useful to determine how much information is retained when using a certain number of principal components. The cumulative variance for the first k principal components is obtained by summing the percentage of variance explained by those components.

> **Cumulative Variance (N) (X%) = Sum of the Percentage of VariaNce for the first N principal components**

> By analyzing the percentage of variance explained by each principal component, one can **make an informed decision about how many principal components to retain for dimens. Higher percentages indicates that a larger proportion of the original dataset's variance is captured by the reduced components.ioality reduction**. Typically, a cumulative variance of around 95% or higher is considered a good choice, as it retains most of the information from the original data while reducing the dimensions

## 4. ML Modelling

#### Dataset description: This dataset includes data for the estimation of obesity levels in individuals based on their eating habits and physical condition. The data contains 17 attributes and 2111 records.

#### Features and labels: The attribute names are listed below. The description of the attributes can be found in this article:
[web link](!https://www.sciencedirect.com/science/article/pii/S2352340919306985)


### i. Check target distribution




    1999    133.644711
    90       93.000000
    1299     88.126544
    189      62.000000
    743      53.783977
    86       83.000000
    279      52.000000
    129      78.000000
    202     102.000000
    1659    121.658729
    Name: Weight, dtype: float64


![png](Credit%20Task%201%20-%20ML_files/Credit%20Task%201%20-%20ML_67_0.png)
    
    
![png](Credit%20Task%201%20-%20ML_files/Credit%20Task%201%20-%20ML_68_0.png)


> It seems to have a bimodal normal distribution for the weight. There do not seem to be many outliers present as well

### ii. Pairplot


    <seaborn.axisgrid.PairGrid at 0x2584a9df7c0>

    
![png](Credit%20Task%201%20-%20ML_files/Credit%20Task%201%20-%20ML_71_1.png)
    

> From the plto: Age, Height, Weight seem to be normally distributed. We can't directly infer heavy correlation between the scatter plots except between Height and Weight which is expected.

### iii. Data Wrangling (features)



> Mean and median of the numeric columns are near to each other
    Gender
    Male      1068
    Female    1043
    Name: count, dtype: int64
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    family_history_with_overweight
    yes    1726
    no      385
    Name: count, dtype: int64
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FAVC
    yes    1866
    no      245
    Name: count, dtype: int64
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CAEC
    Sometimes     1765
    Frequently     242
    Always          53
    no              51
    Name: count, dtype: int64
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SMOKE
    no     2067
    yes      44
    Name: count, dtype: int64
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SCC
    no     2015
    yes      96
    Name: count, dtype: int64
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CALC
    Sometimes     1401
    no             639
    Frequently      70
    Always           1
    Name: count, dtype: int64
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MTRANS
    Public_Transportation    1580
    Automobile                457
    Walking                    56
    Motorbike                  11
    Bike                        7
    Name: count, dtype: int64
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    

> These all seem to be textual, but they are actually just categorical columns, so we can encode them by one-hot encoding or label encoder. There is also an inherent imabalance in all other columns excluding gender.

### iv. Modelling



#### a. Linear Regression

    Best Hyperparameters: {'copy_X': True, 'fit_intercept': True, 'positive': False}
    Mean Squared Error: 301.78330
    R-squared: 0.55141
    

> R2 value is very low and MSE error is high

#### b. Support Vector Regression

    Best Hyperparameters: {'C': 10, 'epsilon': 0.01, 'kernel': 'linear'}
    Mean Squared Error: 314.33939
    R-squared: 0.53275
    

> Even with a possibility of non-linear kernel, support vector regressor didn't perform as well

#### c. Random Forest Regression

    Best Hyperparameters: {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 30}
    Mean Squared Error: 75.81767
    R-squared: 0.88730
    

> Random forest performs much better than simple linear regression or support vector regression, this implies the data is not very simply linear. We require non-linear methods

    Mean Squared Error: 64.33971
    R-squared: 0.90616
    

> Hence, we can see the best MSE error value of `64` and a good R2 value of `0.9`, We can consider this model the champion model for this toy obesity dataset

## References

‌[1] Silhouette Visualizer — Yellowbrick v1.5 documentation n.d., www.scikit-yb.org, viewed 22 July 2023, <https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html?highlight=silhouette><br>
[2] scikit-learn 2019, sklearn.cluster.KMeans — scikit-learn 0.21.3 documentation, Scikit-learn.org<br>
[3] www.datacamp.com. (n.d.). Python t-SNE with Matplotlib. [online] Available at: https://www.datacamp.com/tutorial/introduction-t-sne<br>
[4] Team, G.L. (2020). What is Curse of Dimensionality in Machine Learning? [online] GreatLearning Blog: Free Resources what Matters to shape your Career! Available at: https://www.mygreatlearning.com/blog/understanding-curse-of-dimensionality/<br>
[5] Karanam, S. (2021). Curse of Dimensionality — A ‘Curse’ to Machine Learning. [online] Medium. Available at: https://towardsdatascience.com/curse-of-dimensionality-a-curse-to-machine-learning-c122ee33bfeb<br>
[6] SciKit-Learn (2009). 3.1. Cross-validation: evaluating estimator performance — scikit-learn 0.21.3 documentation. [online] Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/cross_validation.html.

