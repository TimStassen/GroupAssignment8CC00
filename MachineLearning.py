""" 
This file is created for the group-assignment for the course
Advanced Programming & Biomedical Data Analysis (8CC00).
This utils code file contains the objects and functions
supporting the software training and applying the 
Machine learning models selected.

"""
# calculation/data structuring/data conversions 
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
IPythonConsole.ipython_useSVG=True

# visualization
import matplotlib.pyplot as plt

# ML
import numpy as np
from sklearn import neighbors, metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef
import joblib


# import data

def importFiles(nrFiles=int, writetxt=True):
    """ Import .csv files by selection
    Parameter(s):   nrFiles:                (int) number of files to be selected
    Returns:        allAHDL1InhibitorsDf    (pd.DataFrame) dataframe containing all SMILES and inhibit info
        """
    allAHDL1Inhibitors = []
    for nr, _ in enumerate(range(nrFiles)):
        if nr == 0:
            filename = fd.askopenfilename()
            AHDL1Inhibitors = pd.read_csv(filename ,header = None)
            allAHDL1Inhibitors.append(AHDL1Inhibitors)
        else: 
            filename = fd.askopenfilename()
            AHDL1Inhibitors = pd.read_csv(filename ,header = None)
            # drop header of the consecutive files
            AHDL1Inhibitors = AHDL1Inhibitors.iloc[1:, :]
            # create list of separate pd.dataframes
            allAHDL1Inhibitors.append(AHDL1Inhibitors)
    allAHDL1InhibitorsDf = pd.concat(allAHDL1Inhibitors)
    allAHDL1InhibitorsDf = allAHDL1InhibitorsDf.reset_index(drop=True)
    return allAHDL1InhibitorsDf

def writeNewMolfile(AHDL1InhibitorDf, filename='AllTestedMols.txt'):
    """ write SMILES in 'AHDL1InhibitorDf' dataframe to a .txt file
    Parameter(s):       AHDL1InhibitorDf:   (pd.Dataframe) dataframe containing SMILES and inhibit property
                        fileName:           (str) name for the file to be written
    Returns:            -
    """
    allTestedMolecules = AHDL1InhibitorDf[0] # first 3 for testing, needs to change for all molecules (remove[0:4])
    MolList = allTestedMolecules.values.tolist()
    with open(f'{filename}', 'w') as fp:
        for mol in MolList:
            # write each item on a new line
            fp.write("%s\n" % mol)
        print(f'All molecules stored in: {filename}')

    
def getMolDescriptors(mol, missingVal=None):
    """ Calculate the full list of descriptors for a molecule
    Parameter(s):   mol:  current molecule
                    missingValue: (=None) is set to given value if descriptor 
                                    cannot be calculated
    Returns:        res: (dict): a dictionary containing all descriptor data 
    """
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

def createDescriptorDf(filename='AllTestedMols.txt'):
    """ create pd.Dataframe from text file
    Parameter(s):   filename: (str) file containing SMILES
    Returns:        allDescr: (pd.DataFrame) dataframe containing 
                                molecular descriptors
    """
    RDLogger.DisableLog('rdApp.*')
    suppl = Chem.SmilesMolSupplier('AllTestedMols.txt')
    mols = [m for m in suppl]
    allDescrs = [getMolDescriptors(m) for m in mols]
    allDescrs = pd.DataFrame(allDescrs)
    smiles = pd.read_csv(filename)
    smiles = smiles['SMILES']
    allDescrs.insert(loc=0,column='SMILES',value=smiles)
    return allDescrs


def generateMorganFingerprint(filename='AllTestedMols.txt'):
    """
    Parameter(s):   filename:   (str) file containing SMILES of different molecules
    Returns:        x:          (numpy.array) array containing fingerprint test/train dataset
    """
    RDLogger.DisableLog('rdApp.*')
    suppl = Chem.SmilesMolSupplier(filename)
    mols = [m for m in suppl]
    fingerp = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols]
    fingerprint = []
    for f in fingerp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        fingerprint.append(arr) 
        x = np.asarray(fingerprint)
    
    x = pd.DataFrame(x)

    return x 

def convertToMol(filename='AllTestedMols.txt'):
    """ 
    Parameter(s):   filename: (str) file containing SMILES of different molecules
    Returns:        allDescr: (pd.DataFrame) dataFrame containing molecule-specific
                                objects
    """
    suppl = Chem.SmilesMolSupplier(filename)
    mols = [m for m in suppl]
    RDLogger.DisableLog('rdApp.*')
    allDescrs = [getMolDescriptors(m) for m in mols]
    return pd.DataFrame(allDescrs)

def CombineDescriptorsAndFigerprints(Morganfingerprints, descriptors):
    """ Merge molecular properties inot one dataframe
    Parameter(s):   Morganfingerprints: (pd.DataFrame) containing Morgan fingerprints
                                            (ECFP4) properties
                    descriptors:        (pd.DataFrame) containing molecular descriptors 
                                            extracted from Rdkit package
    Returns:        x:                  (pd.DataFrame) concatenated dataframe of the above
                    x.shape:            (dim) the shape of concatenated dataframe
    """
    smiles = descriptors["SMILES"]
    descriptors = descriptors.drop(columns=["SMILES"])
    x = np.concatenate((Morganfingerprints,descriptors), axis=1)
    x = pd.DataFrame(x)
    x.insert(loc=0,column='SMILES', value=smiles)
    return x, x.shape

# preprocessing

def scaleData(data, scaletype='standardize'):
    """ scale dataset into preferred manner
    Parameter(s):   data:       (pd.Dataframe) dataframe containig all desired molecule
                                    properties
                    scaletype:  (str) 'normalize' or 'standardize' indicating the to be
                                    performed scaling operation
    Returns:        scaledData: (pd.Dataframe) dataframe containig all desired molecule
                                    properties scaled
    """
    smiles = data['SMILES']
    data = data.drop(columns=["SMILES"])
    # data = data.to_numpy()
    data.columns = data.columns.astype(str)
    if scaletype == 'standardize':
        scale = StandardScaler().fit(data)
        scaledData = scale.transform(data)    
    elif scaletype == 'normalize':
        scale = MinMaxScaler().fit(data)
        scaledData = scale.transform(data)  
    else:
        raise ValueError("input should be 'standardize' or 'Normalize'")
    scaledData = pd.DataFrame(scaledData)
    scaledData.insert(loc=0,column='SMILES', value=smiles)

    return scaledData

def PCAfeatureReduction(data, varianceThreshold):
    """ reduce the number of features in the data set explaining (almost)
        no variation
    Parameter(s):   data:               (pd.DataFrame) data set with original nr of features
                    varianceThreshold:  (int) ratio of variance explained by features after reduction
    Return(s):      explainingVariables (pd.DataFrame) data set with nr of features,
                                             explaining the amount of thresholded variance
                    principalDF:        (pd.DataFrame) dataframe containing the principal components
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else: 
        df = data

    smiles = df['SMILES']
    df = df.drop(columns=["SMILES"])
    df.columns = df.columns.astype(str)

    pca = PCA()
    principalComponents = pca.fit_transform(df)
    principalDF = pd.DataFrame(data=principalComponents)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    i = 0
    sumVar = 0
    while sumVar<varianceThreshold:
        sumVar = cumulative_variance_ratio[i]
        i +=1

    explainingVariables = df.iloc[:,:i]
    explainingVariables.insert(loc=0,column='SMILES', value=smiles)

    return explainingVariables, principalDF


def getTargetData(AHDL1Inhibitors):
    """ extract labels from the dataset
    Parameter(s):   AHDL1Inhibitors:    (pd.DataFrame) dataframe consisting of SMILES 
                                            and inhibitor label
    Returns:        y:                  (pd.DataFrame) dataframe consisting of inhibitor label
                    imbalanceIndex:     (int) ratio of the amount of labels==1 compared to all
    """
    y = AHDL1Inhibitors[1][1:].astype(int)
    imbalanceIndex = sum(y)/len(y)
    return y, imbalanceIndex

def splitTrainTestData(x,y, test_size=0.20, nSplits=5, seed=13, printSplit=False):
    """ split train and test set in cross-validation
    Parameter(s):   x:          (pd.DataFrame) feature data set (fingerprints & descriptors)
                    y:          (pd.DataFrame) inhibitor lables (0 or 1)
                    test_size:  (int) ratio of total data set assigned to test set (default = 0.20 )
                    nSplits:    (int) number of cross- validation folds to be made (default = 5)
                    seed:       (int) set seed for consistent splitting of data (default = 13)
                    printSplit: (Boolean) print the folds for cross- validation (default - False)

    Returns:        xTrain:     (pd.DataFrame) train part of x
                    xTest:      (pd.DataFrame) test part of x
                    yTrain:     (pd.DataFrame) corresponding labels for xTrain
                    yTest:      (pd.DataFrame) corresponding labels for xTest
                    cv:         (ndArray) The testing set indices for the number of splits.
    
    """
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=test_size, random_state=seed)
    # create folds for cross- validation
    cv = StratifiedKFold(n_splits=nSplits, shuffle=True,random_state=seed)
    if printSplit == True:
        for i, (train_index, test_index) in enumerate(cv.split(xTrain, yTrain)):
            print("\nFold_" + str(i+1))
            print("TRAIN:", train_index)
            print("TEST:", test_index)
    return xTrain, xTest, yTrain, yTest, cv
 
 # RF

def trainRF(xTrain, yTrain, cv):
    """ train random Forest Classifier
    Parameter(s):   xTrain:                 (pd.DataFrame) feature data set for training
                    yTrain:                 (pd.DataFrame) labels data set for training
                    cv:                     (ndArray) The testing set indices for the number of splits.
    Returns:        trainedRFC:             trained model
                    RFgrid.best_params_:    best parameters found by grid search
                    RFgrid:                 best parameter input for model training
                    param_grid:             complet grid wherein to find optimum
                    RFgrid.best_score_:     best score of all folds 
    """
    
    RFc = RandomForestClassifier()


    param_grid = {"max_features": [xTrain.shape[1] // 10, xTrain.shape[1] // 7, 
                               xTrain.shape[1] // 5, xTrain.shape[1] // 3], 
                  "n_estimators": [100, 250, 500]}
    
    scorer = metrics.make_scorer(metrics.balanced_accuracy_score)
    RFgrid = GridSearchCV(RFc, param_grid, scoring=scorer, error_score="raise",
                           return_train_score=True, n_jobs=2, cv=cv, verbose=1)
    
    # train model
    trainedRFC = RFgrid.fit(xTrain, yTrain)
    print('Best Random Forest Classifier: \n', 'Parameters:', RFgrid.best_params_,
           '\n balanced accuracy: ', RFgrid.best_score_)
    return trainedRFC, RFgrid.best_params_, RFgrid, param_grid, RFgrid.best_score_


# SVM 

def trainSVC(xTrain, yTrain, cv):
    """ train Support Vector Machine Classifier
    Parameter(s):   xTrain:                 (pd.DataFrame) feature data set for training
                    yTrain:                 (pd.DataFrame) labels data set for training
                    cv:                     (ndArray) The testing set indices for the number of splits.
    Returns:        trainedSVC:             trained model
                    SVCgrid.best_params_:   best parameters found by grid search
                    SVCgrid:                best parameter input for model training
                    param_grid:             complet grid wherein to find optimum
                    SVCgrid.best_score_:    best score of all folds 
    """
    svc = SVC(kernel='rbf', probability=True)
    
    param_grid = {"C": [10 ** i for i in range(0, 5)],
                    "gamma": [10 ** i for i in range(-6, 0)]}
    
    scorer = metrics.make_scorer(metrics.balanced_accuracy_score)
    SVCgrid = GridSearchCV(svc, param_grid, scoring=scorer, error_score="raise",
                           return_train_score=True, n_jobs=2, cv=cv, verbose=1)
    
    # train model
    trainedSVC = SVCgrid.fit(xTrain, yTrain)
    print('Best Random Forest Classifier: \n', 'Parameters:', SVCgrid.best_params_,
           '\n balanced accuracy: ', SVCgrid.best_score_)
    return trainedSVC, SVCgrid.best_params_, SVCgrid, param_grid, SVCgrid.best_score_


# KNN
def trainKnn(xTrain, yTrain, cv, k_range=list):
    """ train k Nearest Neighbour Classifier
    Parameter(s):   xTrain:                 (pd.DataFrame) feature data set for training
                    yTrain:                 (pd.DataFrame) labels data set for training
                    cv:                     (ndArray) The testing set indices for the number of splits.
                    k_range:                (list) list of minumum to maximum number of neighbours to consider
    Returns:        trainedknn:             trained model
                    knngrid.best_params_:   best parameters found by grid search
                    knngrid:                best parameter input for model training
                    param_grid:             complete grid wherein to find optimum
                    knngrid.best_score_:    best score of all folds 
    """
    knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': k_range,
                    'weights': ('uniform', 'distance'),
                    }
    scorer = metrics.make_scorer(metrics.balanced_accuracy_score)
    # defining parameter range
    
    knngrid = GridSearchCV(knn, param_grid, scoring=scorer, error_score="raise",
                           return_train_score=True, n_jobs=2, cv=cv, verbose=1)

    trainedknn = knngrid.fit(xTrain, yTrain)
    print('Best Random Forest Classifier: \n', 'Parameters:', knngrid.best_params_,
           '\n balanced accuracy: ', knngrid.best_score_)
    return trainedknn, knngrid.best_params_, knngrid, param_grid, knngrid.best_score_


# saving and testing
def saveTrainedModel(model, filename):
    """ save trained model
    Parameter(s):   model:      (obj) trained Machine learning model
                    filename:   (str) name of the file where model is in saved
    Returns:        -
    """
    joblib.dump(model, f"{filename}.pkl", compress=3)

def testTrainedModel(xTest, yTest, model=None, savedModelfilename=str, scaledData=False, scaledDatafile=None):
    """ Test trained machine learning model
    Parameter(s):       xTest:                    (pd.DataFrame) feature data set for testing 
                        yTest:                    (pd.DataFrame) labels data set for testing
                        model=None:               (obj) machine learning model trained
                        savedModelfilename:       (str) filename/path for trained machine learning model
                        scaledData=False:         (Boolean) indicate if input data is scaled
                        scaledDatafile=None:      (str) filename/path for saved scaled feature dataset
    Returns:            pred                      (ndArray) array containing prediction of test set
                        balAcc                    (int) balanced accuracy metric on test set
                        pred_prob                 (ndArray)
    """
    if model == None:
        if savedModelfilename.find('.pkl') != -1:
            trainedModel = joblib.load(f"{savedModelfilename}")
        else:
            trainedModel = joblib.load(f"{savedModelfilename}.pkl")
    else:
        trainedModel = model

    if scaledData == True:
        pred = trainedModel.predict(xTest) 
    elif scaledDatafile != None:
        scale = joblib.load(scaledDatafile)
        # scale descriptors of the test set compounds
        xTest = scale.transform(xTest)
        # predict logBB class
        pred = trainedModel.predict(xTest)

    balAcc = metrics.balanced_accuracy_score(yTest, pred)
    evaluationReport = metrics.classification_report(yTest, pred)
    confusionMatrix = metrics.confusion_matrix(yTest, pred)

    predProb = trainedModel.predict_proba(xTest)
    return pred, balAcc, evaluationReport, confusionMatrix, predProb


def thresholdedAccuracy(yTest, pred, pred_prob, threshold=0.8):
    """ model's accuracy after certain prediction probability thershold
    Parameter(s):       yTest:      (pd.dataFrame) ground truth labels
                        pred:       (ndArray) predictions of the models
                        pred_prob:  (ndArray) probability of made predictions of the model
                        threshold:  (int) value of minimum certainty of predictions
    Returns:            threshPreds:(ndArray) array containing samples passing the threshold         
                        threshAcc:  (int) accuracy value of thresholded predictions
                        coverage:   (int) number of samples/original amount
    """
    # calc maximum predicted probability for each row (compound) and compare to the threshold
    threshPreds = np.amax(pred_prob, axis=1) > threshold
    threshAcc = accuracy_score(np.asarray(yTest)[threshPreds], pred[threshPreds])
    # calc coverage
    coverage = sum(threshPreds) / len(threshPreds)
    return threshPreds, threshAcc, coverage

def top100molecules(trainedModelFile):
    """ generate selection of most probable AHDL1 inhibitors
    Parameter(s):       trainedModelFile:   (str) filename of the trained model
    Returns:            top100Mols:         (pd.Dataframe) column of 100 SMILES
                                                corresponding to the highest prediction probability
    """
    molecules = importFiles(nrFiles=1)

    writeNewMolfile(molecules, filename='untestedMolFile.txt')
    allDescrs = createDescriptorDf(filename='untestedMolFile.txt')
    x = generateMorganFingerprint()

    x, _ = CombineDescriptorsAndFigerprints(x,allDescrs)

    xScale = scaleData(x,scaletype='standardize')
    smiles = xScale["SMILES"]
    xScaleNoSMILES = xScale.drop(columns=["SMILES"])
    trainedModel = joblib.load(f"{trainedModelFile}")
    predProb = trainedModel.predict_proba(xScaleNoSMILES)
    # pred_prob = trainedModel.predict_proba(untestedDataFile)
    predProbDf = pd.DataFrame(predProb)
    predProbDf.insert(loc=0, column='SMILES', value=smiles)
    predProbDfSort = predProbDf.sort_values(by=1, ascending=False)
    top100Mols = predProbDfSort.head(100)
    return top100Mols

