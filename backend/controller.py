import pandas as pd
import numpy as np
from sklearn import feature_selection
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2, RFE, f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.tree import DecisionTreeClassifier
from itertools import compress
from sklearn.svm import SVR
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import PCA
import random


class Controller_Class:

    def __init__(self):
        self.heart_data = pd.read_excel('Heart Attack FD Data.xlsx', usecols=[
                                        'TIME', 'Event', 'Gender', 'Smoking', 'Diabetes', 'BP', 'Anaemia', 'Age', 'Ejection.Fraction', 'Sodium', 'Creatinine', 'Pletelets', 'CPK'])
        self.heart_data = self.heart_data[['TIME', 'Age', 'Ejection.Fraction', 'Sodium', 'Creatinine',
                                           'Pletelets', 'CPK', 'Gender', 'Smoking', 'Diabetes', 'Anaemia', 'BP', 'Event']]
        self.X, self.Y = None, None

    def display_data(self):
        print(self.heart_data)

    def min_max_normalize(self):
        features = self.heart_data.columns.to_list()[0:-1]
        for col in features:
            self.heart_data[col] = ((self.heart_data[col] - min(self.heart_data[col])) / (
                max(self.heart_data[col]) - min(self.heart_data[col])))
        return self.heart_data.copy()

    def get_filter_features(self):

        # ANOVA Method (Numerical Columns)
        anova_max_cols = 3
        numerical_features = [
            'TIME', 'Age', 'Ejection.Fraction', 'Sodium', 'Creatinine', 'Pletelets', 'CPK']
        best_features_obj = SelectKBest(score_func=f_classif, k=anova_max_cols)
        best_features_obj.fit_transform(self.X[numerical_features], self.Y)
        best_numeric_features = list(
            compress(numerical_features, list(best_features_obj.get_support())))

        # Chi Square (Categorical Columns)
        chi_max_cols = 2
        categorical_features = [
            'Gender', 'Smoking', 'Diabetes', 'Anaemia', 'BP']
        best_features_obj = SelectKBest(score_func=chi2, k=chi_max_cols)
        best_features_obj.fit_transform(self.X[categorical_features], self.Y)
        best_categoric_features = list(
            compress(categorical_features, list(best_features_obj.get_support())))

        best_features = list()
        best_features.extend(best_numeric_features)
        best_features.extend(best_categoric_features)

        best_data_set = self.heart_data[best_features]
        return best_data_set

    def get_wrapper_features(self):
        max_features = 2
        selector = RFE(estimator=DecisionTreeClassifier(),
                       n_features_to_select=max_features, step=1)
        selector = selector.fit(self.X, self.Y)
        best_features = self.X.columns[selector.support_]
        best_data_set = self.heart_data[best_features]
        return best_data_set

    def get_hybrid_features(self):
        max_features = 2
        model = RandomForestClassifier()
        model.fit(self.X, self.Y)
        importances = model.feature_importances_
        df = pd.DataFrame(
            data={'Features': self.X.columns, 'Importances': importances})
        df.sort_values('Importances', ascending=False, inplace=True)
        best_features = df.nlargest(max_features, 'Importances')[
            'Features'].values
        best_data_set = self.heart_data[best_features]
        return best_data_set

    def get_pca_features(self):
        pca_obj = PCA(n_components=6, random_state=10)
        x_new = self.heart_data.loc[len(self.heart_data)-1, :]
        x_new = np.array(x_new[0:-1])
        self.heart_data.drop(labels=len(
            self.heart_data) - 1, axis=0, inplace=True)
        pca_obj.fit(self.heart_data.iloc[:, 0:-1])
        reduced_features = pca_obj.transform(self.heart_data.iloc[:, 0:-1])
        column_list = []
        for i in range(reduced_features.shape[1]):
            column_list.append('PC'+str(i+1))

        reduced_features = pd.DataFrame(reduced_features, columns=column_list)

        x_new = pca_obj.transform(x_new.reshape(1, -1))
        x_new = x_new[0]
        return reduced_features, x_new

    def check_feature_selection_technique(self, details):
        best_data_set = None
        if(details['feature_technique_var'] == 'filter'):
            best_data_set = self.get_filter_features()
        elif(details['feature_technique_var'] == 'wrapper'):
            best_data_set = self.get_wrapper_features()
        else:
            best_data_set = self.get_hybrid_features()
        return best_data_set

    def calc_KNN(self, k_ranges, x_train, y_train, x_test, y_test, x_new):
        train_set_accuracies = []
        test_set_accuracies = []
        test_set_predictions = []
        best_k = None
        best_score = None

        if(len(k_ranges) <= 2):
            # training on training data and finding accuracy
            k = int(k_ranges)
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(x_train, y_train)
            yhat = neigh.predict(x_test)
            test_set_accuracies.append(
                metrics.accuracy_score(y_test, yhat))
            best_k = k

            predicted_value = neigh.predict([x_new])

            return best_k, round(predicted_value[0], 2), test_set_accuracies[0]

        else:
            for k in k_ranges:
                neigh = KNeighborsClassifier(n_neighbors=k)
                neigh.fit(x_train, y_train)
                yhat = neigh.predict(x_test)
                train_set_accuracies.append(
                    metrics.accuracy_score(y_train, neigh.predict(x_train)))
                test_set_accuracies.append(
                    metrics.accuracy_score(y_test, yhat))
                test_set_predictions.append(yhat)

            # find the best K
            for i in range(len(k_ranges)):
                if (best_k == None):
                    best_k = i + 1
                    best_score = test_set_accuracies[i]
                elif (best_score < test_set_accuracies[i]):
                    best_k = i + 1
                    best_score = test_set_accuracies[i]

            # evaluating new test case for best K
            neigh = KNeighborsClassifier(n_neighbors=best_k)
            neigh.fit(x_train, y_train)
            predicted_value = neigh.predict([x_new])

            return best_k, round(predicted_value[0], 2), test_set_accuracies[best_k-1]

    def calc_NaiveBayes(self, x_train, y_train, x_test, y_test, x_new):
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        yhat = gnb.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, yhat)
        predicted_value = gnb.predict([x_new])

        return predicted_value[0], accuracy

    def evaluate(self, details):

        x_new = [float(details['time']), float(details['age']), float(details['ejection']), float(details['sodium']), float(details['creatinine']), float(details['pletelets']),
                 float(details['cpk']), float(details['gender_var']), float(details['smoking_var']), float(details['diabetes_var']), float(details['anamia_var']), float(details['bp_var'])]
        self.heart_data = self.heart_data.append({'TIME': x_new[0], 'Age': x_new[1], 'Ejection.Fraction': x_new[2], 'Sodium': x_new[3], 'Creatinine': x_new[4], 'Pletelets': x_new[5],
                                                  'CPK': x_new[6], 'Gender': x_new[7], 'Smoking': x_new[8], 'Diabetes': x_new[9], 'Anaemia': x_new[10], 'BP': x_new[11]}, ignore_index=True)
        best_data_set = None

        # checking normalize condition
        if(details['normalize_var'] == 'true'):
            self.heart_data = self.min_max_normalize()

        if(details['feature_technique_var'] == 'pca'):
            best_data_set, x_new = self.get_pca_features()
            self.Y = self.heart_data['Event']
            x_train, x_test, y_train, y_test = train_test_split(
                best_data_set, self.Y, test_size=0.3, random_state=1)

        else:
            x_new = self.heart_data.loc[len(self.heart_data)-1, :]
            self.heart_data.drop(labels=len(self.heart_data) -
                                 1, axis=0, inplace=True)

            # separating features and labels
            self.X, self.Y = self.heart_data.iloc[:,
                                                  :-1], self.heart_data['Event']

            # checking feature subset selection method
            best_data_set = self.check_feature_selection_technique(details)

            # performing train test split
            x_train, x_test, y_train, y_test = train_test_split(
                best_data_set, self.Y, test_size=0.3, random_state=1)

            x_new = x_new[x_train.columns.to_list()]

        # checking which classifier to use
        if(details['classifier_var'] == 'K-Nearest Neighbours'):
            k_ranges = None
            if(details['kvalue_var'] == 'best'):
                k_ranges = [i for i in range(1, 10, 2)]
            else:
                k_ranges = details['kvalue_var']
            best_k, predicted_value, accuracy = self.calc_KNN(
                k_ranges, x_train, y_train, x_test, y_test, x_new)

            return {'best_k': best_k, 'predicted_value': predicted_value, 'accuracy': accuracy}
        else:
            predicted_value, accuracy = self.calc_NaiveBayes(
                x_train, y_train, x_test, y_test, x_new)
            return {'predicted_value': predicted_value, 'accuracy': accuracy}
