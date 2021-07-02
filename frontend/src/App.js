import React from 'react';
import './App.css';
import Navigation from './Navigation';
import Predict from './Predict';
import About from './About';
import dataset_image from './images/dataset_image.jpg';
import normalization_results from './images/normalization_results.jpg';
import feature_selection from './images/feature_selection.jpg';
import {BrowserRouter as Router,Switch,Route} from 'react-router-dom';
import knn_line_1 from './images/knn_line_1.jpg'
import knn_matrix_1 from './images/knn_matrix_1.jpg'
import nb_bar_1 from './images/nb_bar_1.jpg';
import nb_matrix_1 from './images/nb_matrix_1.jpg';
import knn_line_2 from './images/knn_line_2.jpg'
import knn_matrix_2 from './images/knn_matrix_2.jpg'
import nb_bar_2 from './images/nb_bar_2.jpg';
import nb_matrix_2 from './images/nb_matrix_2.jpg';
import knn_line_3 from './images/knn_line_3.jpg'
import knn_matrix_3 from './images/knn_matrix_3.jpg'
import nb_bar_3 from './images/nb_bar_3.jpg';
import nb_matrix_3 from './images/nb_matrix_3.jpg';
import kmeans from './images/kmeans.jpg';

// Browser router is used to add the ability to do routing(switching between pages)
// Everything that falls between the router tags will be allowed to switch
// The route tag will only show the heading such as Shop page/about page only when you visit that page 
// Address bar mei local host k agay /shop ya /about lagao aur dekho navigate hojata

function App() {
  return (
    
    <Router>
    
      <div className="App">
        <Navigation />
          <Switch>
            <Route path="/" exact component={Home} /> 
            <Route path="/predict" component={Predict} />
          </Switch>
      </div>
    
    </Router>
  );
}

const Home=()=>{
  return(
    <div>
      
      <h1>KNN VS NAIVE BAYES</h1>
      
      <h1 className="Phase-Heading"><img src={dataset_image} className="Image"></img>Heart Attack Dataset</h1>
      <p className="Description-Text">The data set is called <b>Heart Attack Dataset</b>.
        It contains information about the patients such as age, gender, whether they smoked or not. It also has information regarding 
        whether the patients had diabetes, anamia, high blood pressure, follow up period and eventually
        if they had died during the follow up period or not.
      </p>

      <h1 className="Phase-Heading"><img src={normalization_results} className="Image1"></img>Data Normalization</h1>
      <p className="Description-Text">Before we began our analysis, we checked the impact of normalization
        of our data on each of the classification algorithms using the <b>Full Feature Set</b> with KNN and Naive Bayes. We found out the intital
        accuracy of Naive bayes before and after normalization remained the same but there was a considerable increase
        in accuracy of KNN.      
      </p>

      <h1 className="Phase-Heading"><img src={feature_selection} className="Image2"></img>Feature Subset Selection</h1>
      <p className="Description-Text">In order for us to increase the accuracy of the classifiers by a considerable amount,
        we had to come up with an <b>Ideal Feature Selection</b> method which will reduce the unnecessary dimensions of the Data
        which are reducing the accuracy, performance and efficiency of both our models. We used the following approach
        from the figure and began to test the following techniques and eventually choose the best one.
        <ul>
          <li>Filter Methods (Statistics)</li>
          <li>Wrapper Methods (RFE)</li>
          <li>Hybrid (Trees)</li>
        </ul>
      </p>

        <h2 className="Phase-Heading">1. Filter Methods</h2>
        <p className="Description-Text">Firstly we decided to use filter methods and check how much of an impact 
          they have on the accuracy of both our classifiers. There were 2 filter methods which suited our dataset.
          <ol>
            <li><b>Analysis Of Variance (ANOVA) Method:</b> For Numerical Input and Categorical Output</li><br/>
            <li><b>Chi Square Test:</b> For Categorical Input and Categorical Output</li>
          </ol>
          Using the above techniques, we found out the following features were the most important in our dataset.
          <ul>
            <li>Time</li>
            <li>Ejection.Fraction</li>
            <li>Creatinine</li>
            <li>Anamia</li>
            <li>BP</li>
          </ul>
          We kept these features and discarded the rest. After that we tested these features on both our
          classifiers.
        </p>

        <h3 className="Phase-Heading"><img src={knn_matrix_1} className="Image2"></img><img src={knn_line_1} className="Image3"></img>KNN Performance</h3>
        <p className="Description-Text">We ran <b>KNN Algorithm</b> on 10 values of K from 1-10 and checked the response
          of each value of K in terms of accuracy. We found out the best performance was at <b>K = 4</b> with an accuracy
          of 0.81 which is about <b>81 %</b> which is an improvement from the previous accuracy which was <b>72 %</b>
          The confusion matrix also shows the correctly predicted labels.
        </p>
      
        <h3 className="Phase-Heading"><img src={nb_matrix_1} className="Image2"></img><img src={nb_bar_1} className="Image3"></img>Naive Bayes Performance</h3>
        <p className="Description-Text">We ran <b>Naive Bayes Algorithm</b> and found out the accuracy decreased from previous 
          <b> 81 %</b> to now <b>79 %</b>. So we decided to check out the wrapper methods.
        </p>
 
        <h2 className="Phase-Heading">2. Wrapper Methods</h2>
        <p className="Description-Text">After testing out the filter method techniques, we used a wrapper method
          called <b>Recursive Feature Elimination</b> which uses trees to find out the best features.          
          Using the above techniques, we found out the following features were the most important in our dataset.
          <ul>
            <li>Time</li>
            <li>CPK</li>
          </ul>
          We kept these features and discarded the rest. After that we tested these features on both our
          classifiers.
        </p>

        <h3 className="Phase-Heading"><img src={knn_matrix_2} className="Image2"></img><img src={knn_line_2} className="Image3"></img>KNN Performance</h3>
        <p className="Description-Text">We ran <b>KNN Algorithm</b> on 10 values of K from 1-10 and checked the response
          of each value of K in terms of accuracy. We found out the best performance was at <b>K = 6</b> with an accuracy
          of 0.83 which is about <b>83 %</b> which is an improvement from the previous accuracy which was <b>81 %</b> using the
          filter method. The confusion matrix also shows the correctly predicted labels.
        </p>
      
        <h3 className="Phase-Heading"><img src={nb_matrix_2} className="Image2"></img><img src={nb_bar_2} className="Image3"></img>Naive Bayes Performance</h3>
        <p className="Description-Text">We ran <b>Naive Bayes Algorithm</b> and found out the accuracy increased from previous 
          <b> 79 %</b> to now <b>81 %</b>. 
        </p>
 
        <h2 className="Phase-Heading">3. Hybrid Methods</h2>
        <p className="Description-Text">After testing out both the filter and wrapper method techniques, we tried a combination
          of both techniques. <b>Random Forrest Classifier</b> used both techniques of Filter and wrapper methods to find the best
          features from the dataset. Here are the most important features extracted by Random Forrest Classifier.
          <ul>
            <li>Time</li>
            <li>Creatinine</li>
          </ul>
          We kept these features and discarded the rest. After that we tested these features on both our
          classifiers.
        </p>

        <h3 className="Phase-Heading"><img src={knn_matrix_3} className="Image2"></img><img src={knn_line_3} className="Image3"></img>KNN Performance</h3>
        <p className="Description-Text">We ran <b>KNN Algorithm</b> on 10 values of K from 1-10 and checked the response
          of each value of K in terms of accuracy. We found out the best performance was at <b>K = 5</b> with an accuracy
          of 0.86 which is about <b>86 %</b> which is an improvement from the previous accuracies from both filter and wrapper 
          methods and the best accuracy out of every methods.
          </p>
      
        <h3 className="Phase-Heading"><img src={nb_matrix_3} className="Image2"></img><img src={nb_bar_3} className="Image3"></img>Naive Bayes Performance</h3>
        <p className="Description-Text">We ran <b>Naive Bayes Algorithm</b> and found out the accuracy increased from previous 
          <b> 81 %</b> to now <b>82 %</b> which is the overall best accuracy for Naive Bayes yet. 
        </p>
 
        <h3 className="Phase-Heading">Observations and Conclusions</h3>
        <p className="Phase-Ending">After trying out different preprocessing and feature selection techniques,
          we obtained the following information.
          <ol>
            <li>
              Data Normalization plays a huge role in improving the accuracy of a model, we discussed above on how it improved
              our accuracy from <b>66 % </b> to <b>72 %</b> for KNN.
            </li><br/>
            <li>
              Feature Selection also plays a very critical role as it improved our accuracy for both the classifiers with KNN
              giving a best accuracy of <b>86 % </b> which is overall a very good accuracy considering the amount of training data 
              we had which was very low.
            </li>
          </ol>
        </p>

      <h1 className="Phase-Heading"><img src={kmeans} className="Image2"></img>K-Means Algorithm</h1>
      <p className="Description-Text">After trying out both the classification methods, we implemented K-Means Clustering and segmented 
        patients on the basis of <b>Similar Age Groups</b> and found out the following similarities.
      </p>

 
    </div>
  );
}


export default App;
